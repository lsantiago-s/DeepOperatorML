# Multilayer Horizontal Rocking (estado atual no repositório)

Este problema usa o solver Fortran em `libs/` para gerar respostas complexas de deslocamento em solo estratificado transversalmente isotrópico.

## Formato de entrada (`INPUT.TXT`)
```
omegai omegaf omegainc
N Nload B M outputfilename
CODExx CODExf CODEfx CODEff
(N+1 linhas de propriedades)
c11 c12 c13 c33 c44 eta rho hn
```

- `N`: número de camadas (o solver lê `N+1` linhas, sendo a última o semi-espaço).
- `Nload`: interface onde a carga é aplicada (`1` = superfície).
- `B`: raio interno da carga anelar (`B=0` para carga circular).
- `M`: discretização radial (afeta loops `I` e `J` no solver).
- `CODE..`: flags das quatro componentes calculadas.

## Saídas
No código Fortran atual (`PRINCIPAL.FOR`), o formato de saída é por blocos:
- `SAIDA_URFx_W.out`
- `SAIDA_UZFx_W.out`
- `SAIDA_URMy_W.out`
- `SAIDA_UZMy_W.out`
- `SAIDA_UZZ_W.out` (novo canal vertical para carga vertical anular)

Cada arquivo contém blocos:
```
OMEGA= ...
Nrec= ...
real imag
real imag
...
```

Também existem arquivos legados no repositório (`data/raw/.../v1/SAIDA_*_.out`) sem o bloco `Nrec`; o gerador Python atual suporta ambos os formatos.

## Geração de dataset para ML (implementado)
Use o pipeline do projeto:
1. `python3 gen_data.py --problem multilayer_horizontal_rocking`
2. `python3 preprocess_data.py --problem multilayer_horizontal_rocking`

Para usar um arquivo de configuração alternativo (por exemplo, baseline do paper):
1. `python3 gen_data.py --problem multilayer_horizontal_rocking --config ./configs/problems/multilayer_horizontal_rocking/datagen_paper_baseline.yaml`
2. `python3 preprocess_data.py --problem multilayer_horizontal_rocking`

O gerador Python:
- amostra propriedades por camada + frequência;
- cria `INPUT.TXT` por amostra;
- executa o solver;
- parseia cinco saídas complexas (`URFx`, `UZFx`, `URMy`, `UZMy`, `UZZ`);
- salva `xb`, `xt`, `g_u` em `.npz`.

## Plotagem para comparação com o paper
Após treinar/testar (`main.py --problem multilayer_horizontal_rocking --mode test`), a
plotagem específica do problema gera:

- `plots/paper_alignment/paper_baseline_compatibility.yaml`
- `plots/paper_alignment/paper_ratio_coverage.png`
- `plots/paper_profiles/frequency_sweep_by_case.png`
- `plots/paper_profiles/radial_profiles_omega_*.png`
- `plots/prediction_heatmaps/sample_*_truth_pred_heatmaps.png`

Esses gráficos seguem o estilo de comparação por caso/camada/frequência do paper
(A/B/C e varredura em frequência), usando as saídas disponíveis no solver atual
(`URFx`, `UZFx`, `URMy`, `UZMy`, `UZZ`).

### Observação importante de modelagem
O paper **Vertical Vibrations of an Elastic Foundation... (2014)** foca em resposta
vertical da placa (`w`, `M_r`, `Q`). O código Fortran desta pasta agora inclui também
o canal vertical do meio (`UZZ`), além dos canais de **horizontal + rocking** (`Fx` e `My`).

Ou seja, a comparação implementada aqui fica mais próxima do problema vertical no nível
do operador do solo, mas ainda não reproduz literalmente as grandezas da placa
(`w`, `M_r`, `Q`) sem adicionar o módulo variacional da placa do paper.

## Observações importantes
- O binário `multilayer.exe` em `libs/` é macOS arm64.
- O binário `HORROCK_190615.exe` é Windows.
- Em Linux (cluster), compile os `.FOR` com `gfortran`.
- Faça validação física antes de treinar (caso de referência homogêneo, comparação com literatura).
