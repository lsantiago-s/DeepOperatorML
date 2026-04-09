from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.modules.models.config import DONTrainConfig, DataConfig, ExperimentConfig, PathConfig, WandbConfig

logger = logging.getLogger(__name__)


def _serialize_for_wandb(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return _serialize_for_wandb(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _serialize_for_wandb(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_wandb(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _metric_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "metric"


class WandbLogger:
    def __init__(
        self,
        cfg: WandbConfig,
        *,
        data_cfg: DataConfig,
        train_cfg: DONTrainConfig,
        exp_cfg: ExperimentConfig,
        path_cfg: PathConfig,
    ) -> None:
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.exp_cfg = exp_cfg
        self.path_cfg = path_cfg
        self.run = None

    def start(self) -> None:
        if not self.cfg.enabled:
            return

        import wandb

        run_name = self.cfg.name or f"{self.data_cfg.problem}-{self.path_cfg.experiment_version}"
        tags = list(self.cfg.tags)
        for auto_tag in (
            self.data_cfg.problem,
            str(self.exp_cfg.model.strategy.name),
            str(self.exp_cfg.model.branch.architecture),
            str(self.exp_cfg.model.trunk.architecture),
        ):
            if auto_tag not in tags:
                tags.append(auto_tag)

        config_payload = {
            "data": _serialize_for_wandb(self.data_cfg),
            "train": _serialize_for_wandb(self.train_cfg),
            "experiment": _serialize_for_wandb(self.exp_cfg),
            "paths": {
                "outputs_path": str(self.path_cfg.outputs_path),
                "checkpoints_path": str(self.path_cfg.checkpoints_path),
                "metrics_path": str(self.path_cfg.metrics_path),
                "plots_path": str(self.path_cfg.plots_path),
            },
        }

        self.run = wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity,
            group=self.cfg.group,
            job_type=self.cfg.job_type,
            mode=self.cfg.mode,
            name=run_name,
            tags=tags,
            notes=self.cfg.notes,
            dir=str(self.path_cfg.outputs_path),
            config=config_payload,
            settings=wandb.Settings(
                console="off",
                x_disable_stats=True,
                x_disable_meta=True,
                x_disable_machine_info=True,
            ),
        )
        self.run.summary["outputs_path"] = str(self.path_cfg.outputs_path)
        self.run.summary["experiment_version"] = self.path_cfg.experiment_version
        self.run.summary["dataset_version"] = self.data_cfg.dataset_version
        logger.info("Initialized W&B run '%s' in project '%s'.", run_name, self.cfg.project)

    def log_epoch(
        self,
        *,
        phase: str,
        phase_index: int,
        phase_epoch: int,
        global_epoch: int,
        learning_rate: float,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        if self.run is None:
            return

        payload: dict[str, Any] = {
            "phase": phase,
            "phase_index": phase_index,
            "phase_epoch": phase_epoch,
            "global_epoch": global_epoch,
            "learning_rate": learning_rate,
        }

        for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metrics.items():
                payload[f"{split_name}/{_metric_slug(key)}"] = value

        self.run.log(payload, step=global_epoch)

    def finish(self, *, training_history_path: Path | None = None) -> None:
        if self.run is None:
            return

        if training_history_path is not None and training_history_path.exists():
            try:
                import wandb

                self.run.log({"artifacts/training_history": wandb.Image(str(training_history_path))})
                self.run.summary["training_history_path"] = str(training_history_path)
            except Exception as exc:
                logger.warning("Failed to upload training history image to W&B: %s", exc)

        self.run.summary["metrics_path"] = str(self.path_cfg.metrics_path)
        self.run.summary["plots_path"] = str(self.path_cfg.plots_path)
        self.run.finish()
        self.run = None
