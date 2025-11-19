import torch
from typing import Protocol, TYPE_CHECKING
if TYPE_CHECKING:
    from src.modules.models.config.don_config import DeepONetConfig

class OutputHandler(Protocol):
    num_channels: int
    def adjust_dimensions(self, config: 'DeepONetConfig'): ...

    def combine(self, branch_out: torch.Tensor,
                trunk_out: torch.Tensor) -> torch.Tensor: ...
