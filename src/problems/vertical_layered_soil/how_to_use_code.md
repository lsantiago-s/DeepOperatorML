# Vertical Layered Soil (full multilayer formulation)

Este problema segue a formulação de solo estratificado vertical com entrada em perfil de propriedades no eixo de profundidade e aprendizado da matriz de influência completa.

- entrada branch: `xb = [c11(z), c12(z), c13(z), c33(z), c44(z), rho(z), eta(z), a0]`
- dimensão da entrada: `7*Nz + 1` (com `Nz = profile_encoding.num_points`)
- entrada trunk: `xt = (r_m, s1^n, s2^n)`
- saída: canais complexos `[Uxx, Uxz, Uzx, Uzz]` por ponto de `xt`, equivalentes à matriz `U ∈ C^(2M x 2M)`

## Entrada do solver Fortran

O solver legado lê `INPUT.TXT` no formato:

```text
omegai omegaf omegainc
N Nload B M outputfilename
CODExx CODExf CODEfx CODEff
(N+1 linhas de propriedades)
c11 c12 c13 c33 c44 eta rho h
```

Nesta pipeline:

- `N` = número de camadas finitas
- linha `N+1` = semi-espaço (usa `h = 0`)
- frequência normalizada: `a0 = omega * a / cS_ref`
- conversão usada no gerador: `omega = a0 * cS_ref / a_ref`, com `cS_ref = sqrt(c44^(N+1)/rho^(N+1))`
- perfis `theta(z)` são amostrados em grade uniforme `z in [0, z_max]` (configurável em `profile_encoding`)

## Canais e montagem da matriz U

Mapeamento padrão:

- `Uxx <- URFx`
- `Uzx <- UZFx`
- `Uxz <- URMy`
- `Uzz <- UZMy`

Matriz final:

```text
U = [[Uxx, Uxz],
     [Uzx, Uzz]]
```

## Geração de dataset

Random multilayer:

```bash
python3 gen_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/datagen.yaml
python3 preprocess_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/config_preprocessing.yaml
```

Paper baseline (casos A/B/C, N=2):

```bash
python3 gen_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/datagen_paper_baseline.yaml
python3 preprocess_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/config_preprocessing.yaml
```

Paper damping variant:

```bash
python3 gen_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/datagen_paper_damping.yaml
python3 preprocess_data.py --problem vertical_layered_soil --config ./configs/problems/vertical_layered_soil/config_preprocessing_paper_damping.yaml
```

Sanity plots para o dataset bruto:

```bash
python3 src/problems/vertical_layered_soil/sanity_plots.py \
  --raw-data ./data/raw/vertical_layered_soil/vertical_layered_soil_paper_baseline_v4_profile_operator.npz \
  --output-dir ./output/vertical_layered_soil/data_sanity
```

## Artefatos `.npz`

- `xb`: `(num_samples, 7*Nz + 1)` (perfil em profundidade + `a0`)
- `xt`: `(M*M, 3)` com colunas `(r, s1, s2)`
- `g_u`: `(num_samples, M*M, 4)` complexo (`Uxx,Uxz,Uzx,Uzz`)
- `g_u_full`: `(num_samples, (2M)^2)` complexo (compatibilidade/diagnóstico)
- `g_u_blocks`: `(num_samples, M, M, 4)` complexo (`Uxx,Uxz,Uzx,Uzz`)
- `a0`, `omega`, `properties`, `profiles`, `z`, `r`, `s1`, `s2`, `paper_case_label`

## Visualizações e relatórios

No teste, são gerados:

- `plots/paper_alignment/formulation_alignment.yaml`
- `plots/paper_alignment/block_mean_heatmaps.png`
- `plots/paper_profiles/dynamic_compliance_proxies.png`
- `plots/paper_profiles/paper_case_reference_compliances.png`
- `plots/paper_profiles/paper_case_prediction_compliances.png`
- `plots/prediction_heatmaps/sample_*_full_matrix_heatmaps.png`

Contrato de benchmark:

- `metrics/baseline_performance_report.yaml`
- `metrics/baseline_performance_table.csv`
- `metrics/timing_comparison_report.yaml`
- `plots/performance_tracking/*.png`
