from __future__ import annotations
import time
import numpy
import logging
from src.modules.pipe.saving import Saver
from src.modules.models.deeponet.deeponet_factory import DeepONetFactory
from src.modules.models.deeponet.dataset import preprocessing_utils as dtl
from src.modules.models.tools.metrics.errors import ERROR_METRICS
from src.modules.models.config import DataConfig, TestConfig
from src.modules.models.deeponet.dataset.deeponet_transform import DeepONetTransformPipeline

logger = logging.getLogger(__name__)


def inference(test_cfg: TestConfig, data_cfg: DataConfig):

    metric = ERROR_METRICS[test_cfg.metric]  # type: ignore
    model_params = test_cfg.checkpoint['model']  # type: ignore

    _, _, test_data = dtl.get_split_data(data=data_cfg.data,
                                         split_indices=data_cfg.split_indices,
                                         features_keys=data_cfg.features,
                                         targets_keys=data_cfg.targets
                                         )
    stats = dtl.get_stats(data=data_cfg.scalers,
                          keys=data_cfg.features + data_cfg.targets)

    transform_pipeline = DeepONetTransformPipeline(
        config=test_cfg.transforms)  # type: ignore

    transform_pipeline.set_branch_stats(
        stats=stats[data_cfg.features[0]]
    )
    transform_pipeline.set_trunk_stats(
        stats=stats[data_cfg.features[1]]
    )
    transform_pipeline.set_target_stats(
        stats=stats[data_cfg.targets[0]]
    )

    test_transformed = dtl.get_transformed_data(
        data=test_data,                                        
        features_keys=data_cfg.features,                                        
        targets_keys=data_cfg.targets,                                        
        transform_pipeline=transform_pipeline
    )

    model = DeepONetFactory.create_for_inference(
        saved_config=test_cfg.model, state_dict=model_params).to(device=test_cfg.device)  # type: ignore

    y_truth = test_transformed[data_cfg.targets[0]]

    start = time.perf_counter()

    y_pred = model(
        test_transformed[data_cfg.features[0]],
        test_transformed[data_cfg.features[1]]
    ).to(test_cfg.device)

    duration = time.perf_counter() - start

    errors = {}
    times = {}

    y_truth_normalized = y_truth
    y_pred_normalized = y_pred

    abs_error_normalized = metric(
        y_truth_normalized - y_pred_normalized
    ).detach().cpu().numpy()
    norm_truth_normalized = metric(y_truth_normalized).detach().cpu().numpy()

    errors['Normalized Error'] = {}
    errors['Physical Error'] = {}
    times['inference_time'] = duration

    normalized_relative_error = abs_error_normalized / numpy.maximum(
        norm_truth_normalized,
        1e-14,
    )
    for i, label in enumerate(data_cfg.targets_labels):
        errors['Normalized Error'][label] = normalized_relative_error[i]

    if test_cfg.transforms is not None and test_cfg.transforms.target.normalization is not None:
        y_truth_physical = transform_pipeline.inverse_transform(
            tensor=y_truth_normalized,
            component='target',
        )
        y_pred_physical = transform_pipeline.inverse_transform(
            tensor=y_pred_normalized,
            component='target',
        )
    else:
        y_truth_physical = y_truth_normalized
        y_pred_physical = y_pred_normalized

    abs_error_physical = metric(
        y_truth_physical - y_pred_physical
    ).detach().cpu().numpy()
    norm_truth_physical = metric(y_truth_physical).detach().cpu().numpy()
    physical_relative_error = abs_error_physical / numpy.maximum(
        norm_truth_physical,
        1e-14,
    )
    for i, label in enumerate(data_cfg.targets_labels):
        errors['Physical Error'][label] = physical_relative_error[i]

    normalized_msg = '\n'.join(
        f"{label}: {errors['Normalized Error'][label]:.3%}"
        for label in data_cfg.targets_labels
    )
    physical_msg = '\n'.join(
        f"{label}: {errors['Physical Error'][label]:.3%}"
        for label in data_cfg.targets_labels
    )
    
    
    logger.info(
        "Test error (normalized space):\n%s\n"
        "Test error (physical space):\n%s\n"
        "computed in %.3f ms.",
        normalized_msg,
        physical_msg,
        duration * 1000,
    )

    data_to_plot = {**{i: j for i, j in data_cfg.data.items()},
                    'predictions': y_pred_physical.detach().cpu().numpy(),
                    'branch_output': model.branch(test_transformed[data_cfg.features[0]]).detach().numpy(),
                    'trunk_output': model.trunk(test_transformed[data_cfg.features[1]]).detach().numpy(),
                    'bias': model.bias.bias.detach().numpy()
                    }

    saver = Saver()

    saver.save_errors(
        file_path=test_cfg.output_path / test_cfg.problem /  # type: ignore
        test_cfg.experiment_version / 'metrics' / 'test_metrics.yaml',
        errors=errors
    )
    saver.save_time(
        file_path=test_cfg.output_path / test_cfg.problem /  # type: ignore
        test_cfg.experiment_version / 'metrics' / 'test_time.yaml',
        times=times
    )

    numpy.savez(test_cfg.output_path / test_cfg.problem /  # type: ignore
                test_cfg.experiment_version / 'aux' / 'output_data.npz', **data_to_plot)

    logger.info(
        f"Saved to {test_cfg.output_path / str(test_cfg.problem) / str(test_cfg.experiment_version)}"
    )
