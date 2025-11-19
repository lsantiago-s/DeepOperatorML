import os
import argparse
import yaml
from src import train_model
from src import test_model
from src.modules.models.config.fno_train_config import FNOTrainConfig
from src.modules.pipe.logging import configure_logging
from src.modules.models.config import DataConfig, DONTrainConfig, TestConfig
from src.modules.models.config import validator as validation
from src.modules.pipe.logging import configure_logging
from src.modules.pipe.train import get_train_config_class
    
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", help="Type problem to be solved.")
    parser.add_argument("--train-config-path", default="./configs/training/config_don_train.yaml",
                        help="Path to training config file.")
    parser.add_argument("--test",   action="store_true",
                        help="Skip training and only test.")
    args = parser.parse_args()

    problem_path = os.path.join("./configs/problems/", args.problem)
    train_config_path = args.train_config_path
    experiment_config_path = os.path.join(
        problem_path, "config_experiment.yaml")
    problem_test_config_path = os.path.join(problem_path, "config_test.yaml")

    data_cfg = DataConfig.from_experiment_config(
        problem=args.problem,
        exp_cfg=yaml.safe_load(open(experiment_config_path))
    )
    if get_train_config_class(yaml.safe_load(open(experiment_config_path))['model']) is DONTrainConfig:
        train_cfg = DONTrainConfig.from_config_files(
            exp_cfg_path=experiment_config_path,
            train_cfg_path=train_config_path,
            data_cfg=data_cfg
        )
    elif get_train_config_class(yaml.safe_load(open(experiment_config_path))['model']) is FNOTrainConfig:
        train_cfg = FNOTrainConfig.from_config_files(
            exp_cfg_path=experiment_config_path,
            train_cfg_path=train_config_path,
            data_cfg=data_cfg
        )
        pass
    else:
        raise NotImplementedError(f"Model {yaml.safe_load(open(experiment_config_path))['model']} not implemented yet.")

    validation.validate_train_config(train_cfg)
    validation.validate_config_compatibility(data_cfg, train_cfg)

    if args.test:
        test_cfg = TestConfig.from_config_files(
            test_cfg_path=problem_test_config_path)

        experiment_path = test_cfg.output_path / args.problem / \
            test_cfg.experiment_version / 'experiment_config.yaml'

        if test_cfg.experiment_version is None:
            raise AttributeError(
                "Experiment version was not set in 'config.test.yaml' file.")
        with open(experiment_path) as file:
            exp_dict = yaml.safe_load(file)

        test_model(test_cfg_base=test_cfg,
                   exp_cfg_dict=exp_dict, data_cfg=data_cfg)

    else:
        train_model(data_cfg=data_cfg, train_cfg=train_cfg)


if __name__ == '__main__':
    configure_logging()
    main()
