from src.modules.models.config import DataConfig, TrainConfig
from src.modules.models.config.don_train_config import DONTrainConfig
from src.modules.models.config.fno_train_config import FNOTrainConfig
from src.modules.pipe.don_train import deeponet_train
from src.modules.pipe.fno_train import fno_train

def train_model(data_cfg: DataConfig, train_cfg: TrainConfig) -> None:
    if isinstance(train_cfg, DONTrainConfig):
        deeponet_train(data_cfg=data_cfg, train_cfg=train_cfg)
    elif isinstance(train_cfg, FNOTrainConfig):
        fno_train(data_cfg=data_cfg, train_cfg=train_cfg)
    else:
        raise NotImplementedError(f"Model {type(train_cfg).__name__} not implemented yet.")
    
def get_train_config_class(model_name: str) -> type[TrainConfig]:
    if model_name == 'don':
        return DONTrainConfig
    elif model_name == 'fno':
        return FNOTrainConfig
    else:
        raise NotImplementedError(f"Model {model_name} not implemented yet.")