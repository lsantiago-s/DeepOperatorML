# Multilayer Horizontal Rocking (formulação bi-material completa)

Este problema foi ajustado para a formulação de **solo não homogêneo bi-material** com aprendizado da matriz de influência completa:

- entrada: `s = [p^(1), p^(2), a0]` com **15 dimensões**
- saída: matriz complexa `U ∈ C^(2M x 2M)` com blocos `Uxx, Uxz, Uzx, Uzz`

## Entrada do solver Fortran

O solver legado ainda lê `INPUT.TXT` no formato:

```text
omegai omegaf omegainc
N Nload B M outputfilename
CODExx CODExf CODEfx CODEff
(N+1 linhas de propriedades)
c11 c12 c13 c33 c44 eta rho h
```

Para a formulação bi-material desta pipeline:

- `N = 1` (duas linhas de propriedades: meio 1 e meio 2)
- o campo `h` é mantido por compatibilidade numérica do solver (não entra na branch do modelo)
- a frequência normalizada `a0` é amostrada e convertida para `omega` via
  `omega = a0 * cS1 / a_ref`, `cS1 = sqrt(c44^(1)/rho^(1))`

## Canais e montagem da matriz U

A montagem padrão é:

- `Uxx <- URFx`
- `Uzx <- UZFx`
- `Uxz <- URMy`
- `Uzz <- UZMy`

com

```text
U = [[Uxx, Uxz],
     [Uzx, Uzz]]
```

## Geração de dataset

Seleção do solver:

- `executable_path: auto` escolhe o binário nativo da plataforma em `libs/`
- Linux: `multilayer_linux.exe`
- macOS: `multilayer.exe`
- Windows: `HORROCK_190615.exe`

Para recompilar o solver no Linux:

```bash
./src/problems/multilayer_horizontal_rocking/libs/build_linux_solver.sh
```

Baseline principal:

```bash
python3 gen_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/datagen.yaml
python3 preprocess_data.py --problem multilayer_horizontal_rocking
```

Baseline paper-oriented:

```bash
python3 gen_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/datagen_paper_baseline.yaml
python3 preprocess_data.py --problem multilayer_horizontal_rocking
```

Baseline damping (Labaki et al., Sec. 4.4):

```bash
python3 gen_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/datagen_paper_damping.yaml
python3 preprocess_data.py --problem multilayer_horizontal_rocking
```

Nos modos `paper_case`, os casos seguem a Tabela 2 do artigo:

- `A`: meio1=A, meio2=A
- `B`: meio1=A, meio2=B
- `C`: meio1=A, meio2=C
- `D`: meio1=A, meio2=D
- `DB_DAMPING`: meio1=D, meio2=B

## Artefatos salvos em `.npz`

- `xb`: `(num_samples, 15)`
- `xt`: `((2M)^2, 2)` (índices normalizados receptor/fonte)
- `g_u`: `(num_samples, (2M)^2)` complexo (matriz `U` achatada)
- `g_u_blocks`: `(num_samples, M, M, 4)` complexo (`Uxx,Uxz,Uzx,Uzz`)
- `a0`, `omega`, `properties`

## Plotagem e relatórios

No teste, a visualização do problema gera:

- `plots/paper_alignment/formulation_alignment.yaml`
- `plots/paper_alignment/block_mean_heatmaps.png`
- `plots/paper_profiles/dynamic_compliance_proxies.png`
- `plots/paper_profiles/paper_case_reference_compliances.png`
- `plots/paper_profiles/paper_case_prediction_compliances.png`
- `plots/prediction_heatmaps/sample_*_full_matrix_heatmaps.png`

E o contrato de benchmark:

- `metrics/baseline_performance_report.yaml`
- `metrics/baseline_performance_table.csv`
- `metrics/timing_comparison_report.yaml`
- `plots/performance_tracking/*.png`
