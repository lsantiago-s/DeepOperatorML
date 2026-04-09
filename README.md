# DeepOperatorML

## Description

This project provides operator-learning pipelines (primarily DeepONet) for mechanics-inspired surrogate problems.

## Scope

Runnable end-to-end pipelines are available for:

- `kelvin`
- `rajapakse_fixed_material`
- `ground_vibration`
- `multilayer_horizontal_rocking`

`rajapakse_homogeneous` is currently out of benchmark scope.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Common Commands

### 1) Generate raw data

```bash
python3 gen_data.py --problem <problem_name>
```

Optional explicit datagen config:

```bash
python3 gen_data.py --problem <problem_name> --config <datagen_yaml>
```

When a problem provides dataset plotting support, `gen_data.py` also writes sanity-check `.png` plots next to the generated raw dataset file.

### 2) Preprocess

```bash
python3 preprocess_data.py --problem <problem_name>
```

Optional explicit preprocessing config:

```bash
python3 preprocess_data.py --problem <problem_name> --config <config_preprocessing_yaml>
```

### 3) Train

```bash
python3 main.py --problem <problem_name> --train-config-path ./configs/training/config_don_train.yaml
```

### 4) Test

```bash
python3 main.py --problem <problem_name> --test
```

## Problem-Specific Pipelines

### kelvin

```bash
python3 gen_data.py --problem kelvin
python3 preprocess_data.py --problem kelvin
python3 main.py --problem kelvin --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem kelvin --test
```

### rajapakse_fixed_material

Paper-baseline datagen config:

```bash
python3 gen_data.py --problem rajapakse_fixed_material --config ./configs/problems/rajapakse_fixed_material/datagen_paper_baseline.yaml
python3 preprocess_data.py --problem rajapakse_fixed_material
python3 main.py --problem rajapakse_fixed_material --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem rajapakse_fixed_material --test
```

### ground_vibration

Datagen uses externally provided matrices/parameters configured in `configs/problems/ground_vibration/datagen.yaml`.
The learned operator is the full surface influence matrix map
`(c11, c13, c33, c44, rho, eta, a0) -> U`, where
`a0 = omega b / cS` and `cS = sqrt(c44 / rho)`.

If the external `params_array.csv` still stores `omega` instead of `a0`,
the generator derives `a0` automatically using `strip_half_width`.

To create the external CSV/JSON bundle expected by `gen_data.py` on Linux
without MATLAB, use:

```bash
./.venv/bin/python scripts/ground_vibration/generate_external_dataset.py \
  --out-dir ./data/raw/ground_vibration/influence_dataset_N100_samples100_csv \
  --n-samples 100 \
  --n-points 100 \
  --half-span 2.0 \
  --damping 0.01
```

```bash
python3 gen_data.py --problem ground_vibration
python3 preprocess_data.py --problem ground_vibration
python3 main.py --problem ground_vibration --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem ground_vibration --test
```

### multilayer_horizontal_rocking

This pipeline is aligned with the bi-material full-influence formulation:

- input branch: `s = [p^(1), p^(2), a0]` (15 dimensions)
- output target: full complex matrix `U in C^(2M x 2M)` with blocks `Uxx, Uxz, Uzx, Uzz`

Requires legacy executable path configured in datagen YAML.

Paper baseline (Labaki et al. anisotropy cases, medium 1 fixed as case A, medium 2 in `{A,B,C}`):

```bash
python3 gen_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/datagen_paper_baseline.yaml
python3 preprocess_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/config_preprocessing.yaml
python3 main.py --problem multilayer_horizontal_rocking --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem multilayer_horizontal_rocking --test
```

Paper damping study dataset (Labaki et al. damping-style setup, `DB_DAMPING`):

```bash
python3 gen_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/datagen_paper_damping.yaml
python3 preprocess_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/config_preprocessing_paper_damping.yaml
python3 main.py --problem multilayer_horizontal_rocking --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem multilayer_horizontal_rocking --test
```

Key multilayer paper-style outputs produced during test:

- `plots/paper_alignment/formulation_alignment.yaml`
- `plots/paper_alignment/block_mean_heatmaps.png`
- `plots/paper_profiles/dynamic_compliance_proxies.png`
- `plots/paper_profiles/paper_case_reference_compliances.png`
- `plots/paper_profiles/paper_case_prediction_compliances.png`
- `plots/prediction_heatmaps/sample_*_full_matrix_heatmaps.png`

## Paper-Ready Benchmark Orchestration

Manifest:

- `configs/benchmarks/paper_ready_benchmark.yaml`

Run one full benchmark problem (with config freezing):

```bash
python3 scripts/benchmark/run_benchmark_pipeline.py \
  --manifest ./configs/benchmarks/paper_ready_benchmark.yaml \
  --problem ground_vibration \
  --model-track don \
  --stage all \
  --freeze-dataset-version \
  --freeze-experiment-version
```

Run all enabled problems:

```bash
python3 scripts/benchmark/run_benchmark_pipeline.py --manifest ./configs/benchmarks/paper_ready_benchmark.yaml --stage all
```

Dry-run commands only:

```bash
python3 scripts/benchmark/run_benchmark_pipeline.py --manifest ./configs/benchmarks/paper_ready_benchmark.yaml --stage all --dry-run
```

## Benchmark Aggregation and Validation

```bash
python3 scripts/benchmark/aggregate_benchmark_reports.py \
  --manifest ./configs/benchmarks/paper_ready_benchmark.yaml \
  --track don
```

Use `--strict` to fail on validation issues:

```bash
python3 scripts/benchmark/aggregate_benchmark_reports.py --manifest ./configs/benchmarks/paper_ready_benchmark.yaml --track don --strict
```

Generated outputs (default):

- `output/benchmark_reports/paper_ready_benchmark/global_error_table.csv`
- `output/benchmark_reports/paper_ready_benchmark/global_timing_table.csv`
- `output/benchmark_reports/paper_ready_benchmark/figure_index.csv`
- `output/benchmark_reports/paper_ready_benchmark/validation_report.yaml`
- per-problem bundles under `output/benchmark_reports/paper_ready_benchmark/bundles/`

## Required Metrics Contract Per Test Run

Every benchmark run should produce:

- `metrics/baseline_performance_report.yaml`
- `metrics/baseline_performance_table.csv`
- `metrics/timing_comparison_report.yaml`
- `plots/performance_tracking/*.png`

## Kelvin Operator Track (`q(x) -> u`)

Added configs for operator-mode Kelvin (fixed material, variable distributed load profile):

- `configs/problems/kelvin/datagen_operator.yaml`
- `configs/problems/kelvin/config_preprocessing_operator.yaml`
- `configs/problems/kelvin/config_experiment_operator.yaml`
- `configs/problems/kelvin/config_test_operator.yaml`

Run with explicit config paths:

```bash
python3 gen_data.py --problem kelvin --config ./configs/problems/kelvin/datagen_operator.yaml
python3 preprocess_data.py --problem kelvin --config ./configs/problems/kelvin/config_preprocessing_operator.yaml
python3 main.py --problem kelvin \
  --train-config-path ./configs/training/config_don_train.yaml \
  --experiment-config-path ./configs/problems/kelvin/config_experiment_operator.yaml \
  --test-config-path ./configs/problems/kelvin/config_test_operator.yaml
python3 main.py --problem kelvin --test \
  --experiment-config-path ./configs/problems/kelvin/config_experiment_operator.yaml \
  --test-config-path ./configs/problems/kelvin/config_test_operator.yaml
```

## FNO Track Status

`config_fno_train.yaml` and benchmark manifest hooks are present, but `src/modules/pipe/fno_train.py` is scaffold-only and currently raises `NotImplementedError`.

## Coaraci GPU Templates

See:

- `scripts/hpc/coaraci/README.md`
- `scripts/hpc/coaraci/env_coaraci.sh`
- `scripts/hpc/coaraci/train.slurm`
- `scripts/hpc/coaraci/test.slurm`
- `scripts/hpc/coaraci/full_pipeline.slurm`

Adjust SLURM account/partition/module settings to your approved project before submitting.
