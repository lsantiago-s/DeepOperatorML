# DeepOperatorML

## Description

This project provides operator-learning pipelines (primarily DeepONet) for mechanics-inspired surrogate problems.

## Scope

Runnable end-to-end pipelines are available for:

- `kelvin`
- `rajapakse_fixed_material`
- `ground_vibration`
- `vertical_layered_soil`

`rajapakse_homogeneous` is currently out of benchmark scope.

## Install

```bash
python3 -m pip install -e .
```

Python 3.11 or newer is required.

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

Native library compatibility note (Linux clusters):

- If you hit `GLIBC_2.34 not found` for `axsgrsce.so`, rebuild the library against glibc 2.28 (or older) before running datagen.
- See [`src/problems/rajapakse_fixed_material/libs/README.md`](src/problems/rajapakse_fixed_material/libs/README.md) for cluster-native and containerized rebuild paths, plus symbol compatibility checks.

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

### vertical_layered_soil

This pipeline is aligned with the vertical layered full-influence formulation (`N` finite layers over a half-space):

- input branch: `s = [p^(1), ..., p^(N+1), a0]`
- input dimension: `8(N+1)` (includes finite-layer thicknesses and excludes half-space thickness)
- output target: full complex matrix `U in C^(2M x 2M)` with blocks `Uxx, Uxz, Uzx, Uzz`

Requires legacy executable path configured in datagen YAML.

Paper baseline (`N=2`, Labaki et al. vertical layered cases A/B/C):

```bash
python3 gen_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/datagen_paper_baseline.yaml
python3 preprocess_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/config_preprocessing.yaml
python3 main.py --problem vertical_layered_soil --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem vertical_layered_soil --test
```

Paper damping study dataset (vertical layered damping variant):

```bash
python3 gen_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/datagen_paper_damping.yaml
python3 preprocess_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/config_preprocessing_paper_damping.yaml
python3 main.py --problem vertical_layered_soil --train-config-path ./configs/training/config_don_train.yaml
python3 main.py --problem vertical_layered_soil --test
```

Run raw-data sanity plots (after `gen_data`):

```bash
python3 src/problems/vertical_layered_soil/sanity_plots.py \
  --raw-data ./data/raw/vertical_layered_soil/vertical_layered_soil_paper_baseline_v3.npz \
  --output-dir ./output/vertical_layered_soil/data_sanity
```

Key vertical layered paper-style outputs produced during test:

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
