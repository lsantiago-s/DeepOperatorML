import torch
import yaml
from dataclasses import dataclass
from src.modules.models.config.train_config import TrainConfig
from src.modules.models.config.data_config import DataConfig

@dataclass
class FNOTrainConfig(TrainConfig):
    """Aggregate training configuration."""
    precision: str
    device: str | torch.device
    seed: int
    model_name: str

    @classmethod
    def from_config_files(cls, exp_cfg_path: str, train_cfg_path: str, data_cfg: DataConfig):
        del data_cfg
        with open(exp_cfg_path, "r", encoding="utf-8") as f:
            exp_cfg = yaml.safe_load(f)
        with open(train_cfg_path, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)

        device = torch.device(exp_cfg["device"])
        dtype = getattr(torch, exp_cfg["precision"])

        return cls(
            precision=dtype,
            device=device,
            seed=int(train_cfg.get("seed", 42)),
            model_name=str(train_cfg.get("model_name", "fno")),
        )
