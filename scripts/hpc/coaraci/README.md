# Coaraci GPU Templates

These templates provide a reproducible baseline to run one benchmark problem on Coaraci GPU nodes.

## Queue Profiles

The submit scripts encode the queue limits you provided:

- `par48-x`: `1 node / 48 cores`, max walltime `72:00:00`, max active jobs `3`
- `par480-x`: `10 nodes / 480 cores`, max walltime `48:00:00`, max active jobs `2`
- `gpu-x`: `1 node / 3 gpus`, max walltime `168:00:00`, max active jobs `3`

## Files
- `env_coaraci.sh`: environment/bootstrap hook (modules, venv/conda activation).
- `train.slurm`: train DeepONet (or FNO once implemented) on a selected problem.
- `test.slurm`: run test/inference for an already trained experiment.
- `full_pipeline.slurm`: optional end-to-end `gen -> preprocess -> train -> test`.
- `queue_profiles.sh`: shared queue settings and limit checks.
- `submit_train.sh`: queue-aware `sbatch` wrapper for training.
- `submit_test.sh`: queue-aware `sbatch` wrapper for testing.
- `submit_pipeline.sh`: queue-aware `sbatch` wrapper for benchmark pipeline.

## Usage
1. Copy and edit `env_coaraci.sh` for your project allocation, module names, and Python environment.
2. Submit with one of the wrappers:

```bash
scripts/hpc/coaraci/submit_train.sh --queue gpu-x --account <project_account> --problem ground_vibration
scripts/hpc/coaraci/submit_test.sh --queue gpu-x --account <project_account> --problem ground_vibration
scripts/hpc/coaraci/submit_pipeline.sh --queue par48-x --account <project_account> --problem kelvin
```

Vertical layered benchmark run (paper baseline from manifest):

```bash
scripts/hpc/coaraci/submit_pipeline.sh --queue gpu-x --account <project_account> --problem vertical_layered_soil
```

Ground-vibration benchmark run:

```bash
scripts/hpc/coaraci/submit_pipeline.sh --queue gpu-x --account <project_account> --problem ground_vibration
```

When the external ground-vibration CSV bundle is missing, the benchmark
pipeline now generates it automatically on the cluster before calling
`gen_data.py`. For `ground_vibration` with `--stage gen` or `--stage all`,
`submit_pipeline.sh` now defaults to `--cpus-per-task 16` unless you
override it explicitly.

Direct train/test for vertical layered soil:

```bash
scripts/hpc/coaraci/submit_train.sh --queue gpu-x --account <project_account> --problem vertical_layered_soil
scripts/hpc/coaraci/submit_test.sh --queue gpu-x --account <project_account> --problem vertical_layered_soil
```

3. Validate generated `sbatch` command before submission:

```bash
scripts/hpc/coaraci/submit_pipeline.sh --queue par480-x --problem kelvin --dry-run
```

## Vertical Layered Notes
- The benchmark manifest points `vertical_layered_soil` to `datagen_paper_baseline.yaml`.
- If you want damping-study data (`datagen_paper_damping.yaml`) in pipeline mode, create a manifest copy with that datagen config and pass it via `--manifest`.
- Vertical layered test generates additional paper-style plots under `plots/paper_alignment/`, `plots/paper_profiles/`, and `plots/prediction_heatmaps/`.

The wrappers automatically:
- enforce queue walltime limits
- request resources based on the selected queue profile
- check your active jobs against the queue limit (`squeue`), unless `--skip-limit-check` is provided

## Artifact Sync
Each script writes logs under `output/hpc_logs/` and keeps normal project outputs in `output/<problem>/<experiment_version>/`.

Use your preferred sync command after run completion, for example:

```bash
rsync -av output/<problem>/ <local-machine>:<target-dir>/
```

## Rajapakse GLIBC Compatibility

If `rajapakse_fixed_material` fails with:

`/lib64/libc.so.6: version 'GLIBC_2.34' not found`

rebuild `axsgrsce.so` on a glibc-compatible Linux first:

```bash
cd src/problems/rajapakse_fixed_material/libs
chmod +x build_linux_library.sh check_glibc_compat.sh
RAJAPAKSE_TARGET_GLIBC=2.28 ./build_linux_library.sh
```

This script also syncs the rebuilt `.so` to `rajapakse_homogeneous/libs/`.

Full details and container fallback are documented in:

- `src/problems/rajapakse_fixed_material/libs/README.md`
