import torch
from dataclasses import dataclass
from src.modules.models.config.train_config import TrainConfig

@dataclass
class FNOTrainConfig(TrainConfig):
    """Aggregate training configuration."""
    precision: str
    device: str | torch.device
    seed: int
    model_name: str