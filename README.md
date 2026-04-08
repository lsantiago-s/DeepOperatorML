# DeepOperatorML

## Description

This project aims to develop a framework for solving problems (e.g., PDEs) using an Operator Learning approach. The framework currently provides interfaces for data processing, a modular DeepONet architecture and a wide variety of optimization schemes. The user can define a new custom problem using the interface and solve it using one of the implemented models.

## Project Overview

- **Goal**: To provide a data-driven solver aggregating **state of the art** machine learning algorithms.

- **Approach**: Employment of Operator learning schemes with for multi-output operators. Currently, vanilla DeepONets [Lu, et al., (2019)](https://arxiv.org/abs/1910.03193), POD-DeepONets [Lu, et al., (2021)](https://arxiv.org/abs/2111.05512) and Two-step DeepONet [Lee & Shin, (2023)](https://arxiv.org/abs/2309.01020).
  
- **Implementation**: The model is developed in Python using PyTorch. Support for other tensor backend libraries (e.g., JAX, TensorFlow 2.x) are to be implemented in the future.

## Supported Problem Pipelines

The repository currently contains runnable end-to-end pipelines for:

- `kelvin`
- `rajapakse_fixed_material`
- `ground_vibration`
- `multilayer_horizontal_rocking`

## Common Workflow

Use this sequence for any supported problem:

1. Generate raw data:

```bash
python3 gen_data.py --problem <problem_name>
```

2. Preprocess into DeepONet-ready artifacts:

```bash
python3 preprocess_data.py --problem <problem_name>
```

3. Update `configs/problems/<problem_name>/config_experiment.yaml`:
Set `dataset_version` to the generated folder name under `data/processed/<problem_name>/`.
Get the latest one with:

```bash
ls -1t data/processed/<problem_name> | head -n 1
```

4. Train:

```bash
python3 main.py --problem <problem_name> --train-config-path ./configs/training/config_don_train.yaml
```

5. Test:
Set `experiment_version` in `configs/problems/<problem_name>/config_test.yaml` to the trained run folder under `output/<problem_name>/`.
Then run:

```bash
python3 main.py --problem <problem_name> --test
```

## Problem-by-Problem Instructions

### `kelvin`

Files:
- `configs/problems/kelvin/datagen.yaml`
- `configs/problems/kelvin/config_preprocessing.yaml`
- `configs/problems/kelvin/config_experiment.yaml`
- `configs/problems/kelvin/config_test.yaml`

Commands:

```bash
python3 gen_data.py --problem kelvin
python3 preprocess_data.py --problem kelvin
python3 main.py --problem kelvin --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem kelvin --test
```

### `rajapakse_fixed_material`

Recommended raw-data config:
- `configs/problems/rajapakse_fixed_material/datagen_paper_baseline.yaml`

Alternative:
- `configs/problems/rajapakse_fixed_material/datagen.yaml` (edit `data_filename` first if needed)

Commands:

```bash
python3 gen_data.py --problem rajapakse_fixed_material --config ./configs/problems/rajapakse_fixed_material/datagen_paper_baseline.yaml
python3 preprocess_data.py --problem rajapakse_fixed_material
python3 main.py --problem rajapakse_fixed_material --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem rajapakse_fixed_material --test
```

### `ground_vibration`

`ground_vibration` datagen does not numerically integrate inside this repo. It assembles the dataset from external files configured in `configs/problems/ground_vibration/datagen.yaml`:

- `real_influence_matrix_data_path`
- `imag_influence_matrix_data_path`
- `pde_params_data_path`
- `mesh_params_data_path`

Make sure those files exist before running.

Commands:

```bash
python3 gen_data.py --problem ground_vibration
python3 preprocess_data.py --problem ground_vibration
python3 main.py --problem ground_vibration --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem ground_vibration --test
```

### `multilayer_horizontal_rocking`

This problem uses the legacy solver executable configured by `executable_path` in:

- `configs/problems/multilayer_horizontal_rocking/datagen.yaml`
- `configs/problems/multilayer_horizontal_rocking/datagen_paper_baseline.yaml`

Ensure that executable exists and is runnable on your OS before generating data.

Commands:

```bash
python3 gen_data.py --problem multilayer_horizontal_rocking
python3 preprocess_data.py --problem multilayer_horizontal_rocking
python3 main.py --problem multilayer_horizontal_rocking --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem multilayer_horizontal_rocking --test
```

Paper-baseline generation variant:

```bash
python3 gen_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/datagen_paper_baseline.yaml
```

## Outputs

After preprocessing:
- `data/processed/<problem_name>/<dataset_version>/`

After training/testing:
- `output/<problem_name>/<experiment_version>/`

This output folder contains checkpoints, metrics and problem-dependent plots.
