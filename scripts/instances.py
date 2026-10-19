'''
GERAÇÃO E CONVERSÃO DE INSTÂNCIAS

Todas as instâncias são salvas no formato CSV usado no projeto:
    X, Y, cluster, 0, 1, ..., n-1   (as colunas numéricas são a matriz de distâncias euclidianas)

Uso:
    # converter uma instância da GTSPLIB (.txt) para CSV
    python scripts/instances.py gtsp data/raw/25rat99.txt data/25rat99.csv

    # gerar instâncias sintéticas no estilo CTSP_<n>C<k>_<id>
    python scripts/instances.py synthetic --n 50 --k 4 --quantity 10 --out data/
'''

import argparse
import os
import numpy as np
import pandas as pd


def to_dataframe(coords, cluster, round_distances=False):
    """
    Monta o DataFrame no formato do projeto (coordenadas, cluster e matriz de distâncias).

    round_distances=True arredonda para o inteiro mais próximo, como manda a convenção EUC_2D da TSPLIB.
    As melhores soluções conhecidas (BKS) da literatura são calculadas assim, então o arredondamento
    é necessário para o gap em relação à literatura ser comparável.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=2))
    if round_distances:
        distance = np.floor(distance + 0.5)

    df = pd.DataFrame({'X': coords[:, 0], 'Y': coords[:, 1], 'cluster': cluster})
    df = pd.concat([df, pd.DataFrame(distance, columns=[str(i) for i in range(len(coords))])], axis=1)
    return df


def generate_csv(input_file, output_file, round_distances=True):
    """
    Converte uma instância da GTSPLIB (seções NODE_COORD_SECTION e GTSP_SET_SECTION) para CSV.

    Na GTSP_SET_SECTION cada linha é: <id do cluster> <cidade> <cidade> ... -1
    (as cidades são numeradas a partir de 1 no arquivo e a partir de 0 no CSV)
    """
    coords = {}
    cluster = {}
    section = None

    with open(input_file) as f:
        tokens_by_line = [line.split() for line in f if line.strip()]

    set_tokens = []
    for tokens in tokens_by_line:
        head = tokens[0].upper().rstrip(':')
        if head in ('NODE_COORD_SECTION', 'GTSP_SET_SECTION'):
            section = head
            continue
        if head == 'EOF':
            break
        if ':' in ' '.join(tokens) and section is None:
            continue  ## linhas de cabeçalho (NAME: ..., DIMENSION: ...)

        if section == 'NODE_COORD_SECTION':
            node, x, y = int(tokens[0]), float(tokens[1]), float(tokens[2])
            coords[node - 1] = (x, y)
        elif section == 'GTSP_SET_SECTION':
            set_tokens.extend(int(t) for t in tokens)

    ## percorre os tokens da seção de clusters: id, cidades..., -1
    i = 0
    while i < len(set_tokens):
        cluster_id = set_tokens[i]
        i += 1
        while set_tokens[i] != -1:
            cluster[set_tokens[i] - 1] = cluster_id
            i += 1
        i += 1

    n = len(coords)
    coords_array = np.array([coords[c] for c in range(n)])
    cluster_array = np.array([cluster[c] for c in range(n)])

    to_dataframe(coords_array, cluster_array, round_distances).to_csv(output_file, index=False)
    return output_file


def generate_synthetic_instance(n, k, seed=None, spread=0.08):
    """
    Gera uma instância sintética: k centros aleatórios no quadrado [0, 1]² e as n cidades
    distribuídas em volta deles (distribuição normal), cada uma rotulada com o cluster do seu centro.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1, 0.9, size=(k, 2))

    ## garante ao menos 2 cidades por cluster e distribui o resto aleatoriamente
    cluster = np.concatenate([np.repeat(np.arange(k), 2), rng.integers(0, k, size=n - 2 * k)])
    rng.shuffle(cluster)

    coords = centers[cluster] + rng.normal(0, spread, size=(n, 2))
    return to_dataframe(coords, cluster)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p_gtsp = sub.add_parser('gtsp', help='converte instância GTSPLIB para CSV')
    p_gtsp.add_argument('input_file')
    p_gtsp.add_argument('output_file')
    p_gtsp.add_argument('--no-round', action='store_true',
                        help='não arredondar as distâncias (por padrão segue a convenção EUC_2D da TSPLIB)')

    p_syn = sub.add_parser('synthetic', help='gera instâncias sintéticas')
    p_syn.add_argument('--n', type=int, required=True, help='número de cidades')
    p_syn.add_argument('--k', type=int, required=True, help='número de clusters')
    p_syn.add_argument('--quantity', type=int, default=1)
    p_syn.add_argument('--seed', type=int, default=0)
    p_syn.add_argument('--out', default='data')

    args = parser.parse_args()

    if args.command == 'gtsp':
        generate_csv(args.input_file, args.output_file, round_distances=not args.no_round)
        print(f"Salvo em {args.output_file}")
    else:
        os.makedirs(args.out, exist_ok=True)
        for idx in range(1, args.quantity + 1):
            df = generate_synthetic_instance(args.n, args.k, seed=args.seed + idx)
            path = os.path.join(args.out, f"CTSP_{args.n}C{args.k}_{idx}.csv")
            df.to_csv(path, index=False)
            print(f"Salvo em {path}")
