from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from torch.amp import GradScaler, autocast

from stf.compat import load_legacy_checkpoint
from stf.engine.base import BaseEngine
from stf.io import save_prediction_image, save_show_image
from stf.logging import Tracker


class TrainEngine(BaseEngine):
    def __init__(self, experiment, config_path: str, output_dir: str | None = None, resume_from: str | None = None):
        super().__init__(experiment, config_path, output_dir, mode="train")
        self.train_loader = experiment.data.train_dataloader
        self.val_loader = experiment.data.val_dataloader

        if self.train_loader is None:
            raise ValueError("train_dataloader is required for training")

        self.use_mixed_precision = experiment.train.use_mixed_precision
        self.precision = str(getattr(experiment.train, "precision", "fp16")).lower()
        if self.precision not in {"fp16", "bf16"}:
            raise ValueError(
                f"Unsupported train.precision={self.precision}, expected one of ['fp16', 'bf16']"
            )
        self.amp_enabled = self.use_mixed_precision
        self.amp_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        self.non_blocking_transfer = bool(getattr(experiment.train, "non_blocking_transfer", False))
        self.train_log_interval = max(1, int(getattr(experiment.train, "train_log_interval", 1)))
        self.use_channels_last = bool(getattr(experiment.train, "use_channels_last", False))
        self.compile_model = bool(getattr(experiment.train, "compile_model", False))
        self.compile_mode = str(getattr(experiment.train, "compile_mode", "max-autotune"))
        self.compile_dynamic = bool(getattr(experiment.train, "compile_dynamic", False))

        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.compile_model:
            self.model = torch.compile(
                self.model,
                mode=self.compile_mode,
                dynamic=self.compile_dynamic,
            )

        self.optimizer = self._build_optimizer(experiment.optimizer)
        self.scheduler = self._build_scheduler(experiment.scheduler)
        self.scaler = GradScaler(
            device="cuda",
            enabled=self.amp_enabled and self.precision == "fp16",
        )
        self.use_ema = experiment.train.use_ema
        self.ema = EMA(self.model, beta=0.995, update_every=1).to(self.device) if self.use_ema else None
        self.current_epoch = 0
        self.current_train_step = 0
        self.current_val_step = 0

        checkpoint_path = resume_from or experiment.resume_from
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _build_optimizer(self, optimizer):
        if callable(optimizer):
            return optimizer(params=self.model.parameters())
        return optimizer

    def _build_scheduler(self, scheduler):
        if scheduler is None:
            return None
        if callable(scheduler):
            return scheduler(self.optimizer)
        return scheduler

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        state = load_legacy_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            ema=self.ema,
            device=str(self.device),
            strict=False,
        )
        if self.ema is not None and not state["has_ema"]:
            self.ema.ema_model.load_state_dict(self.model.state_dict(), strict=False)
        epoch = state["epoch"]
        self.current_epoch = (epoch + 1) if epoch is not None else 0
        self.current_train_step = self.current_epoch * len(self.train_loader)
        if self.val_loader is not None and self.experiment.train.val_interval > 0:
            self.current_val_step = (self.current_epoch // self.experiment.train.val_interval) * len(self.val_loader)
        self.txt_logger.info(f"Resumed training from checkpoint: {checkpoint_path}, epoch={self.current_epoch}")

    def _prepare_train_inputs(self, batch):
        move = self._move_batch_tensor
        return [
            move(batch["coarse_img_01"]),
            move(batch["coarse_img_02"]),
            move(batch["fine_img_01"]),
            move(batch["fine_img_02"]),
        ]

    def _prepare_sample_inputs(self, batch):
        move = self._move_batch_tensor
        return [
            move(batch["coarse_img_01"]),
            move(batch["coarse_img_02"]),
            move(batch["fine_img_01"]),
        ]

    def _move_batch_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device, non_blocking=self.non_blocking_transfer)
        if self.use_channels_last and tensor.ndim == 4:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _save_batch_images(self, outputs, batch):
        if not self.experiment.io.save_images and not self.experiment.io.show_images:
            return

        keys = batch.get("key", [str(i) for i in range(outputs.shape[0])])
        dataset_name = batch.get("dataset_name", ["dataset"])[0]
        normalize_scale = batch.get("normalize_scale", [1.0])[0]
        normalize_mode = batch.get("normalize_mode", [2])[0]
        if hasattr(normalize_scale, "item"):
            normalize_scale = normalize_scale.item()
        if hasattr(normalize_mode, "item"):
            normalize_mode = normalize_mode.item()

        for idx in range(outputs.shape[0]):
            key = keys[idx] if isinstance(keys, (list, tuple)) else str(keys)
            pred = outputs[idx : idx + 1]
            if self.experiment.io.save_images:
                save_dir = self.run_dirs["images"] / dataset_name / f"epoch_{self.current_epoch}" / "pred"
                save_prediction_image(pred, save_dir, f"{key}.tif", normalize_scale, normalize_mode)
            if self.experiment.io.show_images:
                show_dir = self.run_dirs["images"] / dataset_name / f"epoch_{self.current_epoch}" / "show"
                show_tensors = [
                    batch["coarse_img_01"][idx : idx + 1],
                    batch["coarse_img_02"][idx : idx + 1],
                    batch["fine_img_01"][idx : idx + 1],
                    batch["fine_img_02"][idx : idx + 1],
                ]
                save_show_image(
                    show_tensors,
                    pred,
                    show_dir,
                    f"{key}.png",
                    normalize_mode,
                    show_bands=self.experiment.io.show_bands,
                )

    def _run_train_epoch(self):
        self.model.train()
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        tracker = Tracker("loss")
        for iter_idx, batch in enumerate(self.train_loader):
            inputs = self._prepare_train_inputs(batch)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                loss = self.model(*inputs)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.experiment.train.grad_clip_norm)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update()

            loss_value = float(loss.item())
            tracker.update("loss", loss_value)
            self.backend_logger.add_scalar("loss/train_step", loss_value, self.current_train_step)
            self.current_train_step += 1

            if iter_idx % self.train_log_interval == 0:
                self.txt_logger.info(
                    f"train epoch={self.current_epoch} iter={iter_idx} loss={loss_value:.4e}"
                )

        epoch_loss = tracker.results["loss"]
        self.backend_logger.add_scalar("loss/train_epoch", epoch_loss, self.current_epoch)
        self.txt_logger.info(f"train epoch={self.current_epoch} avg_loss={epoch_loss:.4e}")
        return epoch_loss

    def _run_val_epoch(self):
        if self.val_loader is None:
            return None

        sampler_model = self.ema.ema_model if self.ema is not None else self.model
        sampler_model.eval()

        tracker = Tracker("loss", *[metric.__name__ for metric in self.metrics])
        for iter_idx, batch in enumerate(self.val_loader):
            sample_inputs = self._prepare_sample_inputs(batch)
            gt = self._move_batch_tensor(batch["fine_img_02"])

            with torch.no_grad(), autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                outputs = sampler_model.sample(*sample_inputs)
                loss = F.mse_loss(outputs, gt)

            loss_value = float(loss.item())
            tracker.update("loss", loss_value)
            self.backend_logger.add_scalar("loss/val_step", loss_value, self.current_val_step)

            pred_for_metrics = (outputs.float() + 1.0) / 2.0
            gt_for_metrics = (gt.float() + 1.0) / 2.0
            ref_for_metrics = (self._move_batch_tensor(batch["fine_img_01"]).float() + 1.0) / 2.0
            for metric in self.metrics:
                num_params = len(inspect.signature(metric.forward).parameters)
                if num_params >= 3:
                    value = float(
                        metric(gt_for_metrics, pred_for_metrics, ref_for_metrics).item()
                    )
                else:
                    value = float(metric(gt_for_metrics, pred_for_metrics).item())
                tracker.update(metric.__name__, value)
                self.backend_logger.add_scalar(f"metric/val_step/{metric.__name__}", value, self.current_val_step)

            self._save_batch_images(outputs, batch)
            self.current_val_step += 1
            self.txt_logger.info(f"val epoch={self.current_epoch} iter={iter_idx} loss={loss_value:.4e}")

        for key, value in tracker.results.items():
            self.backend_logger.add_scalar(f"metric/val_epoch/{key}", value, self.current_epoch)

        msg = ", ".join([f"{k}={v:.4f}" for k, v in tracker.results.items()])
        self.txt_logger.info(f"val epoch={self.current_epoch} {msg}")
        return tracker.results

    def _save_checkpoint(self):
        data = {
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.ema is not None:
            data["ema"] = self.ema.state_dict()

        checkpoint_path = self.run_dirs["checkpoints"] / f"model_epoch_{self.current_epoch}.pth"
        torch.save(data, checkpoint_path)
        self.txt_logger.info(f"Saved checkpoint: {checkpoint_path}")

    def run(self) -> Path:
        max_epochs = self.experiment.train.max_epochs
        val_interval = max(self.experiment.train.val_interval, 1)
        save_interval = max(self.experiment.train.save_interval, 1)

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            self._run_train_epoch()

            val_results = None
            if self.val_loader is not None and (epoch + 1) % val_interval == 0:
                val_results = self._run_val_epoch()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = None if val_results is None else val_results.get("loss")
                    if metric is not None:
                        self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint()

        self.close()
        return self.run_dirs["root"]
