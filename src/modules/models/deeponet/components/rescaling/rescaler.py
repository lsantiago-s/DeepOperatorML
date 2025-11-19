import torch
from src.modules.models.deeponet.components.rescaling.config import DONRescalingConfig

class Rescaler(torch.nn.Module):
    def __init__(self, config: DONRescalingConfig):
        super().__init__()
        self.scale = config.embedding_dimension ** config.exponent

    def __str__(self) -> str:
        return f"Scaler: {self.scale}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
