# Instâncias

Todas no formato CSV do projeto: `X, Y, cluster, 0, 1, ..., n-1` (as colunas numéricas são a matriz de distâncias).

| Pasta | Conteúdo | Clusters |
|---|---|---|
| `sinteticas/` | 50 instâncias `CTSP_<n>C<k>_<id>`: 10C2, 50C4, 100C9, 200C15 e 200C25 (10 de cada) | Definidos na geração |
| `tsplib/originais/` | `5eil51`, `15berlin52` | Partição fixa, usada nos experimentos do projeto |
| `tsplib/reconstruidas/` | `25rat99`, `50eil101`, `10gil262`, `50gil262`, `50rat783`, `100pr1002` | k-means (`scripts/instances.py tsplib`, semente 0) |
| `tsplib/fonte/` | Arquivos `.tsp` da TSPLIB usados como base | — |

Os nomes seguem o padrão `<nº de clusters><instância TSPLIB>`.

## Instâncias sintéticas

As coordenadas são uniformes em [0, 1]², mas a distância **não** é a euclidiana de (X, Y): o número do cluster entra como uma terceira coordenada,

$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (c_i - c_j)^2}$$

Ou seja, ir do cluster 0 para o 8 custa ~8, e do 0 para o 1 custa ~1: a **numeração dos clusters entra no custo**,
o que favorece rotas que visitam os clusters em ordem numérica. Vale ter isso em mente ao interpretar os resultados.

## Instâncias baseadas na TSPLIB

Coordenadas da [TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), distância euclidiana **sem arredondamento**.

As instâncias de `reconstruidas/` têm clusters gerados por k-means neste projeto. Elas **não são as mesmas** instâncias
de mesmo nome publicadas na literatura de CTSP, então não devem ser comparadas com as melhores soluções conhecidas (BKS) desses trabalhos.

Para gerar outras instâncias (ou as mesmas com distâncias arredondadas pela convenção EUC_2D, opção `--round`):

```bash
python scripts/instances.py tsplib instancias/tsplib/fonte/eil101.tsp --k 50 --out minha_instancia.csv
```
