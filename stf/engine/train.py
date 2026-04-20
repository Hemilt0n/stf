from __future__ import annotations

import csv
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from torch.amp import GradScaler, autocast

from stf.compat import load_legacy_checkpoint
from stf.engine.base import BaseEngine
from stf.io import save_prediction_image, save_show_image, save_trust_map_image
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
        self.grad_accum_steps = int(getattr(experiment.train, "grad_accum_steps", 1))
        self.non_blocking_transfer = bool(getattr(experiment.train, "non_blocking_transfer", False))
        self.train_log_interval = max(1, int(getattr(experiment.train, "train_log_interval", 1)))
        self.use_channels_last = bool(getattr(experiment.train, "use_channels_last", False))
        self.compile_model = bool(getattr(experiment.train, "compile_model", False))
        self.compile_mode = str(getattr(experiment.train, "compile_mode", "max-autotune"))
        self.compile_dynamic = bool(getattr(experiment.train, "compile_dynamic", False))
        self.fine_t1_noise_warmup_epochs = int(getattr(experiment.train, "fine_t1_noise_warmup_epochs", 0))
        self.fine_t1_noise_warmup_steps = int(getattr(experiment.train, "fine_t1_noise_warmup_steps", 0))
        self.fine_t1_noise_power = float(getattr(experiment.train, "fine_t1_noise_power", 4.0))
        self.fine_t1_noise_std = float(getattr(experiment.train, "fine_t1_noise_std", 1.0))
        self.fine_t1_noise_alpha_tail = float(getattr(experiment.train, "fine_t1_noise_alpha_tail", 0.0))
        self.val_step_log_keys = bool(getattr(experiment.train, "val_step_log_keys", False))
        self.val_step_log_max_keys = max(1, int(getattr(experiment.train, "val_step_log_max_keys", 8)))
        self.val_step_save_csv = bool(getattr(experiment.train, "val_step_save_csv", False))
        self.val_trust_log_stats = bool(getattr(experiment.train, "val_trust_log_stats", False))
        self.val_trust_save_max = max(0, int(getattr(experiment.train, "val_trust_save_max", 0)))
        if self.grad_accum_steps <= 0:
            raise ValueError("train.grad_accum_steps must be >= 1")
        if self.fine_t1_noise_warmup_epochs < 0:
            raise ValueError("train.fine_t1_noise_warmup_epochs must be >= 0")
        if self.fine_t1_noise_warmup_steps < 0:
            raise ValueError("train.fine_t1_noise_warmup_steps must be >= 0")
        if self.fine_t1_noise_power <= 0:
            raise ValueError("train.fine_t1_noise_power must be > 0")
        if self.fine_t1_noise_std < 0:
            raise ValueError("train.fine_t1_noise_std must be >= 0")
        if not (0.0 <= self.fine_t1_noise_alpha_tail < 1.0):
            raise ValueError("train.fine_t1_noise_alpha_tail must be in [0, 1)")
        if self.fine_t1_noise_warmup_steps > 0:
            self.fine_t1_noise_total_steps = self.fine_t1_noise_warmup_steps
        else:
            self.fine_t1_noise_total_steps = self.fine_t1_noise_warmup_epochs * len(self.train_loader)
        if self.fine_t1_noise_alpha_tail > 0.0 and self.fine_t1_noise_total_steps <= 0:
            raise ValueError(
                "train.fine_t1_noise_alpha_tail > 0 requires "
                "train.fine_t1_noise_warmup_epochs > 0 or train.fine_t1_noise_warmup_steps > 0"
            )
        self._last_fine_t1_noise_alpha = 0.0

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
        if self.fine_t1_noise_total_steps > 0:
            self.txt_logger.info(
                "Enabled fine_t1 noise warmup: "
                f"total_steps={self.fine_t1_noise_total_steps}, "
                f"power={self.fine_t1_noise_power:.2f}, std={self.fine_t1_noise_std:.2f}, "
                f"alpha_tail={self.fine_t1_noise_alpha_tail:.4f}"
            )
        if self.grad_accum_steps > 1:
            self.txt_logger.info(
                f"Enabled gradient accumulation: grad_accum_steps={self.grad_accum_steps}"
            )

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
        coarse_img_01 = move(batch["coarse_img_01"])
        coarse_img_02 = move(batch["coarse_img_02"])
        fine_img_01 = move(batch["fine_img_01"])
        fine_img_02 = move(batch["fine_img_02"])

        fine_t1_noise_alpha = self._get_fine_t1_noise_alpha()
        if fine_t1_noise_alpha > 0:
            noise = torch.randn_like(fine_img_01) * self.fine_t1_noise_std
            fine_img_01 = fine_img_01 * (1.0 - fine_t1_noise_alpha) + noise * fine_t1_noise_alpha
        self._last_fine_t1_noise_alpha = fine_t1_noise_alpha
        if hasattr(self, "backend_logger") and self.backend_logger is not None:
            self.backend_logger.add_scalar(
                "train/fine_t1_noise_alpha",
                fine_t1_noise_alpha,
                self.current_train_step,
            )

        return [
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            fine_img_02,
        ]

    @staticmethod
    def _compute_fine_t1_noise_alpha(
        global_step: int,
        warmup_steps: int,
        power: float,
        alpha_tail: float = 0.0,
    ) -> float:
        if warmup_steps <= 0:
            return 0.0
        progress = max(0.0, min(1.0, max(float(global_step), 0.0) / float(warmup_steps)))
        alpha = alpha_tail + (1.0 - alpha_tail) * (1.0 - (progress**power))
        return max(0.0, min(1.0, alpha))

    def _get_fine_t1_noise_alpha(self) -> float:
        return self._compute_fine_t1_noise_alpha(
            global_step=self.current_train_step,
            warmup_steps=self.fine_t1_noise_total_steps,
            power=self.fine_t1_noise_power,
            alpha_tail=self.fine_t1_noise_alpha_tail,
        )

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

    def _save_batch_images(self, outputs, batch, trust_observability=None):
        if (
            not self.experiment.io.save_images
            and not self.experiment.io.show_images
            and self.val_trust_save_max <= 0
        ):
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
            if (
                trust_observability is not None
                and self.val_trust_save_max > 0
                and self._val_trust_maps_saved < self.val_trust_save_max
            ):
                trust_map = trust_observability.get("trust_map")
                change_map = trust_observability.get("change_map")
                if trust_map is not None:
                    trust_dir = self.run_dirs["images"] / dataset_name / f"epoch_{self.current_epoch}" / "trust"
                    change_slice = None if change_map is None else change_map[idx : idx + 1]
                    save_trust_map_image(
                        trust_map[idx : idx + 1],
                        trust_dir,
                        f"{key}.png",
                        change_tensor=change_slice,
                    )
                    self._val_trust_maps_saved += 1

    @staticmethod
    def _batch_field_to_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().flatten().tolist()
        if isinstance(value, (list, tuple)):
            return [item.item() if isinstance(item, torch.Tensor) else item for item in value]
        return [value]

    def _debug_batch_keys(self, batch, max_items: int | None = None) -> tuple[str, str]:
        sample_idxs = [str(v) for v in self._batch_field_to_list(batch.get("sample_idx"))]
        keys = [str(v) for v in self._batch_field_to_list(batch.get("key"))]

        if max_items is not None and len(sample_idxs) > max_items:
            sample_idx_preview = sample_idxs[:max_items] + ["..."]
        else:
            sample_idx_preview = sample_idxs
        if max_items is not None and len(keys) > max_items:
            key_preview = keys[:max_items] + ["..."]
        else:
            key_preview = keys
        return ",".join(sample_idx_preview), ",".join(key_preview)

    def _run_train_epoch(self):
        self.model.train()
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        tracker = Tracker("loss")
        self.optimizer.zero_grad(set_to_none=True)
        num_iters = len(self.train_loader)
        for iter_idx, batch in enumerate(self.train_loader):
            inputs = self._prepare_train_inputs(batch)

            with autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                loss = self.model(*inputs)
                loss_for_backward = loss / float(self.grad_accum_steps)
            if self.scaler.is_enabled():
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            should_step = ((iter_idx + 1) % self.grad_accum_steps == 0) or ((iter_idx + 1) == num_iters)
            if should_step:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.experiment.train.grad_clip_norm)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update()

            loss_value = float(loss.item())
            tracker.update("loss", loss_value)
            self.backend_logger.add_scalar("loss/train_step", loss_value, self.current_train_step)
            self.current_train_step += 1

            if iter_idx % self.train_log_interval == 0:
                if self.fine_t1_noise_total_steps > 0:
                    self.txt_logger.info(
                        f"train epoch={self.current_epoch} iter={iter_idx} "
                        f"loss={loss_value:.4e} fine_t1_noise_alpha={self._last_fine_t1_noise_alpha:.4f}"
                    )
                else:
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
        metric_names = [metric.__name__ for metric in self.metrics]
        trust_metric_names = ["trust_mean", "trust_changed", "trust_unchanged"]
        debug_writer = None
        debug_file = None
        debug_path = None
        self._val_trust_maps_saved = 0
        if self.val_step_save_csv:
            debug_dir = self.run_dirs["root"] / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"val_step_epoch_{self.current_epoch:04d}.csv"
            debug_file = debug_path.open("w", newline="")
            debug_writer = csv.DictWriter(
                debug_file,
                fieldnames=[
                    "epoch",
                    "iter",
                    "val_step",
                    "loss",
                    "sample_idx",
                    "key",
                    *metric_names,
                    *(trust_metric_names if self.val_trust_log_stats else []),
                ],
            )
            debug_writer.writeheader()
        for iter_idx, batch in enumerate(self.val_loader):
            sample_inputs = self._prepare_sample_inputs(batch)
            gt = self._move_batch_tensor(batch["fine_img_02"])

            with torch.no_grad(), autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                self._clear_trust_observability(sampler_model)
                outputs = sampler_model.sample(*sample_inputs)
                loss = F.mse_loss(outputs, gt)

            loss_value = float(loss.item())
            tracker.update("loss", loss_value)
            self.backend_logger.add_scalar("loss/val_step", loss_value, self.current_val_step)

            pred_for_metrics = (outputs.float() + 1.0) / 2.0
            gt_for_metrics = (gt.float() + 1.0) / 2.0
            ref_for_metrics = (self._move_batch_tensor(batch["fine_img_01"]).float() + 1.0) / 2.0
            metric_values = {}
            for metric in self.metrics:
                num_params = len(inspect.signature(metric.forward).parameters)
                if num_params >= 3:
                    value = float(
                        metric(gt_for_metrics, pred_for_metrics, ref_for_metrics).item()
                    )
                else:
                    value = float(metric(gt_for_metrics, pred_for_metrics).item())
                tracker.update(metric.__name__, value)
                metric_values[metric.__name__] = value
                self.backend_logger.add_scalar(f"metric/val_step/{metric.__name__}", value, self.current_val_step)

            trust_values = {}
            trust_observability = self._get_trust_observability(sampler_model)
            if self.val_trust_log_stats and trust_observability is not None:
                for name in trust_metric_names:
                    value = float(trust_observability[name])
                    tracker.update(name, value)
                    trust_values[name] = value
                    self.backend_logger.add_scalar(f"trust/val_step/{name}", value, self.current_val_step)

            self._save_batch_images(outputs, batch, trust_observability=trust_observability)
            sample_idx_full, key_full = self._debug_batch_keys(batch, max_items=None)
            if debug_writer is not None:
                debug_writer.writerow(
                    {
                        "epoch": self.current_epoch,
                        "iter": iter_idx,
                        "val_step": self.current_val_step,
                        "loss": f"{loss_value:.6e}",
                        "sample_idx": sample_idx_full,
                        "key": key_full,
                        **{name: f"{metric_values[name]:.6e}" for name in metric_names},
                        **{name: f"{trust_values[name]:.6e}" for name in trust_values},
                    }
                )
            sample_idx_preview, key_preview = self._debug_batch_keys(
                batch, max_items=self.val_step_log_max_keys
            )
            self.current_val_step += 1
            if self.val_step_log_keys:
                trust_msg = ""
                if trust_values:
                    trust_msg = " " + " ".join(
                        [f"{name}={value:.4f}" for name, value in trust_values.items()]
                    )
                self.txt_logger.info(
                    "val "
                    f"epoch={self.current_epoch} iter={iter_idx} loss={loss_value:.4e} "
                    f"sample_idx=[{sample_idx_preview}] key=[{key_preview}]{trust_msg}"
                )
            else:
                trust_msg = ""
                if trust_values:
                    trust_msg = " " + " ".join(
                        [f"{name}={value:.4f}" for name, value in trust_values.items()]
                    )
                self.txt_logger.info(
                    f"val epoch={self.current_epoch} iter={iter_idx} loss={loss_value:.4e}{trust_msg}"
                )

        if debug_file is not None:
            debug_file.close()
            self.txt_logger.info(f"Saved val-step debug csv: {debug_path}")

        for key, value in tracker.results.items():
            if key in trust_metric_names:
                self.backend_logger.add_scalar(f"trust/val_epoch/{key}", value, self.current_epoch)
            else:
                self.backend_logger.add_scalar(f"metric/val_epoch/{key}", value, self.current_epoch)

        msg = ", ".join([f"{k}={v:.4f}" for k, v in tracker.results.items()])
        self.txt_logger.info(f"val epoch={self.current_epoch} {msg}")
        return tracker.results

    @staticmethod
    def _trust_observable_candidates(model):
        candidates = [model]
        inner_model = getattr(model, "model", None)
        if inner_model is not None and inner_model is not model:
            candidates.append(inner_model)
        return candidates

    def _clear_trust_observability(self, model):
        for candidate in self._trust_observable_candidates(model):
            clear_fn = getattr(candidate, "clear_last_trust_observability", None)
            if callable(clear_fn):
                clear_fn()

    def _get_trust_observability(self, model):
        for candidate in self._trust_observable_candidates(model):
            get_fn = getattr(candidate, "get_last_trust_observability", None)
            if callable(get_fn):
                state = get_fn()
                if state is not None:
                    return state
        return None

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

    @staticmethod
    def _bytes_to_mib(num_bytes: int) -> float:
        return float(num_bytes) / (1024.0 * 1024.0)

    def _log_peak_memory_stats(self) -> None:
        torch.cuda.synchronize(self.device)
        device_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
        total_mem_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        peak_alloc_bytes = torch.cuda.max_memory_allocated(self.device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
        current_alloc_bytes = torch.cuda.memory_allocated(self.device)
        current_reserved_bytes = torch.cuda.memory_reserved(self.device)

        peak_alloc_ratio = (peak_alloc_bytes / total_mem_bytes) * 100.0
        peak_reserved_ratio = (peak_reserved_bytes / total_mem_bytes) * 100.0
        peak_remaining_by_reserved = max(total_mem_bytes - peak_reserved_bytes, 0)
        peak_remaining_by_alloc = max(total_mem_bytes - peak_alloc_bytes, 0)

        self.txt_logger.info(
            "gpu memory summary: "
            f"total={self._bytes_to_mib(total_mem_bytes):.2f} MiB, "
            f"peak_allocated={self._bytes_to_mib(peak_alloc_bytes):.2f} MiB ({peak_alloc_ratio:.2f}%), "
            f"peak_reserved={self._bytes_to_mib(peak_reserved_bytes):.2f} MiB ({peak_reserved_ratio:.2f}%), "
            f"remaining_by_peak_reserved={self._bytes_to_mib(peak_remaining_by_reserved):.2f} MiB, "
            f"remaining_by_peak_allocated={self._bytes_to_mib(peak_remaining_by_alloc):.2f} MiB, "
            f"current_allocated={self._bytes_to_mib(current_alloc_bytes):.2f} MiB, "
            f"current_reserved={self._bytes_to_mib(current_reserved_bytes):.2f} MiB"
        )

    def run(self) -> Path:
        max_epochs = self.experiment.train.max_epochs
        val_interval = int(self.experiment.train.val_interval)
        save_interval = int(self.experiment.train.save_interval)
        enable_val = self.val_loader is not None and val_interval > 0
        enable_save = save_interval > 0
        torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            self._run_train_epoch()

            val_results = None
            if enable_val and (epoch + 1) % val_interval == 0:
                val_results = self._run_val_epoch()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = None if val_results is None else val_results.get("loss")
                    if metric is not None:
                        self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            if enable_save and (epoch + 1) % save_interval == 0:
                self._save_checkpoint()

        self._log_peak_memory_stats()
        self.close()
        return self.run_dirs["root"]
