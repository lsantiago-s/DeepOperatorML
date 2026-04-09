from __future__ import annotations
import torch
from typing import Any, Type, Optional
from dataclasses import dataclass, fields, is_dataclass
from src.modules.models.deeponet.components.bias.config import DONBiasConfig
from src.modules.models.deeponet.components.rescaling.config import DONRescalingConfig
from src.modules.models.deeponet.components.output_handler.config import DONOutputConfig
from src.modules.models.deeponet.components.branch.config import DONBranchConfig, DONBranchConfig
from src.modules.models.deeponet.components.trunk.config import DONTrunkConfig, DONTrunkConfig
from src.modules.models.deeponet.training_strategies.config import DONStrategyConfig, VanillaConfig, TwoStepConfig, PODConfig

@dataclass
class DeepONetConfig:
    branch: DONBranchConfig
    trunk: DONTrunkConfig
    bias: DONBiasConfig
    output: DONOutputConfig
    rescaling: DONRescalingConfig
    strategy: DONStrategyConfig

    def __post_init__(self):
        if self.strategy is not None:
            self.strategy = self._convert_strategy(self.strategy)

    @classmethod
    def _convert_strategy(cls, strategy_data: Any):
        if isinstance(strategy_data, dict):
            name = strategy_data["name"]

            strategy_class: Type[DONStrategyConfig] = {
                "vanilla": VanillaConfig,
                "pod": PODConfig,
                "two_step": TwoStepConfig
            }[name]

            valid_fields = {f.name for f in fields(strategy_class)}

            filtered = {k: v for k, v in strategy_data.items()
                        if k in valid_fields}
            return strategy_class(**filtered)
        return strategy_data

    @classmethod
    def from_dict(cls, data: dict):

        if 'dtype' in data and isinstance(data['dtype'], str):
            data['dtype'] = getattr(torch, data['dtype'].split('.')[-1])

        converted = {}
        for field in fields(cls):
            if field.name not in data:
                continue

            value = data[field.name]

            if is_dataclass(field.type) and isinstance(value, dict):
                converted[field.name] = field.type.from_dict(value)
            elif (field.type is Optional[DONStrategyConfig] and isinstance(value, dict)):
                converted[field.name] = cls._convert_strategy(value)
            else:
                converted[field.name] = value

        return cls(**converted)
