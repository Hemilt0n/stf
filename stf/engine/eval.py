from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from ema_pytorch import EMA

from stf.compat import load_legacy_checkpoint
from stf.engine.base import BaseEngine
from stf.io import save_prediction_image, save_show_image
from stf.logging import Tracker


class EvalEngine(BaseEngine):
    def __init__(self, experiment, config_path: str, checkpoint_path: str, output_dir: str | None = None):
        super().__init__(experiment, config_path, output_dir, mode="eval")
        self.loader = experiment.data.test_dataloader or experiment.data.val_dataloader
        if self.loader is None:
            raise ValueError("test_dataloader or val_dataloader is required for eval")

        self.ema = EMA(self.model, beta=0.995, update_every=1).to(self.device)
        state = load_legacy_checkpoint(
            checkpoint_path,
            self.model,
            ema=self.ema,
            device=str(self.device),
            strict=False,
        )
        self.use_ema = state["has_ema"]
        self.checkpoint_path = checkpoint_path

    def _prepare_sample_inputs(self, batch):
        return [
            batch["coarse_img_01"].to(self.device),
            batch["coarse_img_02"].to(self.device),
            batch["fine_img_01"].to(self.device),
        ]

    def run(self) -> tuple[Path, dict[str, float]]:
        sampler_model = self.ema.ema_model if self.use_ema else self.model
        sampler_model.eval()

        tracker = Tracker("loss", *[metric.__name__ for metric in self.metrics])
        step = 0
        for batch in self.loader:
            sample_inputs = self._prepare_sample_inputs(batch)
            gt = batch.get("fine_img_02")
            with torch.no_grad():
                outputs = sampler_model.sample(*sample_inputs)

            if gt is not None:
                gt = gt.to(self.device)
                loss = F.mse_loss(outputs, gt)
                tracker.update("loss", float(loss.item()))

                pred_for_metrics = (outputs + 1.0) / 2.0
                gt_for_metrics = (gt + 1.0) / 2.0
                for metric in self.metrics:
                    tracker.update(metric.__name__, float(metric(gt_for_metrics, pred_for_metrics).item()))

            if self.experiment.io.save_images or self.experiment.io.show_images:
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
                        save_prediction_image(
                            pred,
                            self.run_dirs["images"] / dataset_name / "pred",
                            f"{key}.tif",
                            normalize_scale,
                            normalize_mode,
                        )
                    if self.experiment.io.show_images and gt is not None:
                        save_show_image(
                            [
                                batch["coarse_img_01"][idx : idx + 1],
                                batch["coarse_img_02"][idx : idx + 1],
                                batch["fine_img_01"][idx : idx + 1],
                                batch["fine_img_02"][idx : idx + 1],
                            ],
                            pred,
                            self.run_dirs["images"] / dataset_name / "show",
                            f"{key}.png",
                            normalize_mode,
                            show_bands=self.experiment.io.show_bands,
                        )
            step += 1

        results = tracker.results
        for key, value in results.items():
            self.backend_logger.add_scalar(f"metric/eval/{key}", value, step)

        self.close()
        return self.run_dirs["root"], results
