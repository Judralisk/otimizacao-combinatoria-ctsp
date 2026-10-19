# CTSP com metaheurísticas híbridas

Solução do **Clustered Traveling Salesman Problem (CTSP)** com um algoritmo híbrido:
**GRASP** para gerar a população inicial e **VNS + Simulated Annealing + Algoritmo Genético** cooperando a cada iteração.

No CTSP as cidades estão divididas em clusters e a rota precisa visitar todas as cidades de um cluster
em sequência antes de passar para o próximo. O objetivo é minimizar a distância total do ciclo.

![Rota inicial do GRASP vs. rota final do híbrido](docs/rota_100C9.png)

## Como funciona

| Etapa | O que faz |
|---|---|
| **Penalização** | Cada aresta entre clusters diferentes recebe `+M` na matriz de custo. Com `M > n · max(d)`, toda rota viável custa menos que qualquer inviável, então o ótimo do problema penalizado é o ótimo do CTSP. |
| **GRASP** | Construção gulosa-aleatória: a próxima cidade é sorteada na RCL (custo ≤ `c_min + α(c_max − c_min)`), com α sorteado em `{0.25, 0.5, 0.75}`. |
| **VNS** | Três vizinhanças em sequência — troca, inserção e 2-opt — até nenhuma melhorar. Avaliação por delta vetorizada (O(n²) por vizinhança). |
| **Simulated Annealing** | Movimentos aleatórios de inserção com delta O(1), critério de Metropolis e resfriamento geométrico. A temperatura inicial é calibrada pelos próprios dados. |
| **Algoritmo Genético** | Melhor metade da população como pais, crossover de 2 cortes (OX) e de 1 corte, mutação por troca adjacente e seleção elitista sem duplicatas. |
| **Híbrido** | A cada iteração: VNS e SA em cada solução, GA na população; as melhores soluções distintas seguem. Para em `Max_itters` ou após 30 iterações sem melhora. |

## Estrutura

```
ctsp/
├── utils.py           leitura, matriz penalizada, calc_of, viabilidade
├── grasp.py           soluções iniciais
├── neighborhoods.py   troca, inserção e 2-opt (avaliação por delta)
├── vns.py | sa.py | ga.py
├── hybrid.py          loop principal (run_hybrid)
└── plots.py           rota e convergência
instancias/            conjuntos de instâncias (sintéticas e baseadas na TSPLIB)
data/                  instâncias pequenas para testes rápidos
scripts/instances.py   conversor GTSPLIB → CSV, k-means sobre TSPLIB e gerador sintético
tests/test_ctsp.py     testes de correção (deltas vs. força bruta, big-M, viabilidade...)
main.py                experimento completo: várias runs por instância + estatísticas
CTSP_metaheuristicas.ipynb   passo a passo com gráficos
```

## Como rodar

```bash
pip install -r requirements.txt

# experimento: todas as instâncias de uma pasta, 10 runs por instância
python main.py --data instancias/sinteticas --runs 10 --itters 150

# só algumas instâncias, com limite de tempo por run
python main.py --data instancias/sinteticas --instances CTSP_50C4_1.csv CTSP_50C4_2.csv --runs 5 --time-limit 60

# testes
pip install pytest && python -m pytest -q
```

Saídas em `results/`:
- `runs.csv` — uma linha por (instância, run): viabilidade, OF, tempo, ordem dos clusters e a solução
- `summary.csv` — por instância: melhor, média, desvio padrão, mediana da OF, % de runs viáveis e tempo médio.
  Se a pasta de instâncias tiver um `bks.csv` (`Instance,BKS_literature`), inclui o gap % em relação à melhor solução conhecida.

Uso como biblioteca:

```python
from ctsp import read_tsp_file, build_cost_matrix, run_hybrid, calc_of

coords, cluster, distance = read_tsp_file('instancias/sinteticas/CTSP_50C4_1.csv')
cost, M = build_cost_matrix(distance, cluster)
solution, _, history = run_hybrid(cost, {'Max_itters': 50, 'M': M}, seed=0)
print(calc_of(solution, distance))
```

## Instâncias

Formato CSV: `X, Y, cluster, 0, 1, ..., n-1` (as colunas numéricas são a matriz de distâncias).
Os conjuntos disponíveis estão descritos em [`instancias/README.md`](instancias/README.md).

Para gerar novas:

```bash
# converter uma instância da GTSPLIB (.txt com GTSP_SET_SECTION)
python scripts/instances.py gtsp 25rat99.txt 25rat99.csv

# instância CTSP a partir de um .tsp da TSPLIB, com clusters por k-means
python scripts/instances.py tsplib eil101.tsp --k 50 --out 50eil101.csv

# instâncias sintéticas CTSP_<n>C<k>_<id>
python scripts/instances.py synthetic --n 200 --k 15 --quantity 10 --out data/
```

O conversor GTSPLIB arredonda as distâncias para o inteiro mais próximo (convenção EUC_2D da TSPLIB), que é como as
melhores soluções da literatura são calculadas — sem isso o gap não é comparável. Use `--no-round` para desligar.

## Resultados

Configuração padrão (`Solutions_Quantity = 10`, `Max_itters = 150`, parada após 30 iterações sem melhora), uma semente por run,
executado em uma máquina com 2 vCPUs (Python 3.13). A coluna **Melhora sobre o GRASP** compara a melhor solução final
com a melhor solução da população inicial do GRASP na mesma semente. Os dados de cada run estão em [`docs/resultados/`](docs/resultados/).

**Instâncias sintéticas** — 10 instâncias por grupo, 5 runs cada (50 runs por grupo):

| Grupo | Runs viáveis | Melhora sobre o GRASP | CV entre runs | Tempo médio |
|---|---|---|---|---|
| 10C2 | 100% | 5,6% | 0,00% | 1,2 s |
| 50C4 | 100% | 31,2% | 0,19% | 4,0 s |
| 100C9 | 100% | 25,4% | 0,22% | 8,6 s |
| 200C15 | 100% | 31,9% | 0,22% | 23,4 s |
| 200C25 | 100% | 37,3% | 0,08% | 22,1 s |

Nas 10C2 o ótimo foi calculado por enumeração exata: o algoritmo encontrou o ótimo em **50 de 50 runs**.
O CV (coeficiente de variação da OF entre as runs de uma mesma instância, em média) abaixo de 0,25% indica que
o resultado praticamente não depende da semente.

**Instâncias baseadas na TSPLIB** — distância euclidiana sem arredondamento:

| Instância | n | k | Runs | Viáveis | OF inicial (GRASP) | Melhor OF | OF média ± desvio | Melhora | Tempo médio |
|---|---|---|---|---|---|---|---|---|---|
| 5eil51 | 51 | 5 | 10 | 100% | 786,7 | 439,2 | 440,6 ± 1,7 | 43,9% | 4,2 s |
| 15berlin52 | 52 | 15 | 10 | 100% | 11 708,4 | 8 051,2 | 8 185,2 ± 170,0 | 29,9% | 4,4 s |
| 25rat99 | 99 | 25 | 10 | 100% | 2 156,6 | 1 294,1 | 1 328,0 ± 26,3 | 38,4% | 8,8 s |
| 50eil101 | 101 | 50 | 10 | 100% | 1 134,0 | 672,6 | 684,9 ± 8,9 | 39,6% | 8,8 s |
| 10gil262 | 262 | 10 | 10 | 100% | 7 873,6 | 2 617,5 | 2 666,2 ± 38,3 | 66,1% | 42,9 s |
| 50gil262 | 262 | 50 | 10 | 100% | 5 106,9 | 2 539,1 | 2 602,1 ± 50,7 | 48,7% | 40,8 s |
| 50rat783* | 783 | 50 | 5 | 100% | 26 194,9 | 11 426,7 | 12 157,3 ± 414,6 | 53,6% | 180 s |
| 100pr1002* | 1002 | 100 | 5 | 100% | 719 074,6 | 442 189,8 | 448 868,3 ± 4 113,2 | 37,6% | 180 s |

\* Com limite de 180 s por run (`--time-limit 180`). Todas as runs dessas duas pararam pelo limite de tempo
(~44 e ~30 iterações), antes do critério de convergência — o que explica a dispersão maior entre runs.

Observações:
- Todas as runs terminaram com rota viável; com a penalização big-M na matriz, a solução inicial do GRASP já foi viável em todas elas.
- A melhora pequena nas 10C2 é esperada: com 10 cidades o GRASP já chega perto do ótimo.
- As instâncias TSPLIB `reconstruidas/` não são as mesmas da literatura (ver [`instancias/README.md`](instancias/README.md)),
  então esses valores não devem ser comparados com resultados publicados.

Para reproduzir:

```bash
python main.py --data instancias/sinteticas --runs 5
python main.py --data instancias/tsplib/originais --runs 10
python main.py --data instancias/tsplib/reconstruidas --instances 25rat99.csv 50eil101.csv 10gil262.csv 50gil262.csv --runs 10
python main.py --data instancias/tsplib/reconstruidas --instances 50rat783.csv 100pr1002.csv --runs 5 --time-limit 180
```

## Reprodutibilidade

Cada run usa uma semente (`seed = nº da run`) aplicada em `random` e `numpy`, então o mesmo comando gera os mesmos resultados.
Como as metaheurísticas são estocásticas, os resultados devem ser lidos pela distribuição das runs (média, desvio, mediana),
não por uma execução isolada.
