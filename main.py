'''
MAIN CODE

Roda o algoritmo híbrido em todas as instâncias (.csv) de uma pasta, várias vezes por instância
(uma semente por run), e salva:

- results/runs.csv    -> uma linha por (instância, run)
- results/summary.csv -> estatísticas por instância (melhor, média, desvio padrão, % de runs viáveis, tempo)

Se existir um arquivo data/bks.csv com as colunas Instance,BKS_literature (OF da melhor solução conhecida na literatura),
o resumo também traz o gap percentual: 100 * (OF - BKS_literature) / BKS_literature.

Uso:
    python main.py                               # todas as instâncias de data/, 10 runs
    python main.py --data data --runs 30 --itters 150
    python main.py --instances CTSP_50C4_1.csv --runs 5
'''

import os
import time
import argparse
import numpy as np
import pandas as pd

from ctsp import (read_tsp_file, build_cost_matrix, calc_of, check_solution_viability,
                  run_hybrid, save_data)


def summarize(runs_df, bks_file=None):
    """Estatísticas descritivas por instância (as metaheurísticas são estocásticas: uma run só não diz muito)."""
    summary = runs_df.groupby('Instance').agg(
        Runs=('OF_bks', 'size'),
        Feasible_rate=('Feasible', 'mean'),
        OF_best=('OF_bks', 'min'),
        OF_mean=('OF_bks', 'mean'),
        OF_std=('OF_bks', 'std'),
        OF_median=('OF_bks', 'median'),
        Time_mean=('Elapsed_time', 'mean'),
    ).reset_index()

    if bks_file and os.path.exists(bks_file):
        bks = pd.read_csv(bks_file)
        summary = summary.merge(bks, on='Instance', how='left')
        summary['Gap_best_%'] = 100 * (summary['OF_best'] - summary['BKS_literature']) / summary['BKS_literature']
        summary['Gap_mean_%'] = 100 * (summary['OF_mean'] - summary['BKS_literature']) / summary['BKS_literature']

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data', help='pasta com as instâncias .csv')
    parser.add_argument('--instances', nargs='*', help='rodar só estas instâncias (nomes dos arquivos)')
    parser.add_argument('--runs', type=int, default=10, help='runs independentes por instância')
    parser.add_argument('--itters', type=int, default=150, help='Max_itters do híbrido')
    parser.add_argument('--solutions', type=int, default=10, help='Solutions_Quantity')
    parser.add_argument('--M', type=float, default=None, help='penalização (padrão: n * max distância)')
    parser.add_argument('--time-limit', type=float, default=None, help='limite de tempo por run, em segundos')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()

    # Listar os arquivos CSV de instância
    csv_files = args.instances or sorted(f for f in os.listdir(args.data)
                                         if f.endswith('.csv') and f != 'bks.csv')
    os.makedirs(args.out, exist_ok=True)
    runs_file = os.path.join(args.out, 'runs.csv')
    if os.path.exists(runs_file):
        os.remove(runs_file)

    params = {'Max_itters': args.itters, 'Solutions_Quantity': args.solutions, 'time_limit': args.time_limit}

    for instance in csv_files:
        coords, cluster, tsp_distance = read_tsp_file(os.path.join(args.data, instance))

        ## matriz penalizada: d + M quando a aresta troca de cluster
        tsp_distance_punished, M = build_cost_matrix(tsp_distance, cluster, args.M)
        params['M'] = M

        for run in range(args.runs):
            # computando o tempo de início do algoritmo
            start_time = time.time()

            best_known_solution, best_known_of_punished, history = run_hybrid(
                tsp_distance_punished, params, seed=run)

            elapsed_time = time.time() - start_time

            ## of sem a punição
            real_of = calc_of(best_known_solution, tsp_distance)
            is_valid = check_solution_viability(best_known_solution, cluster)
            cluster_order = cluster[best_known_solution].tolist()

            print(f"Instance: {instance} | run {run + 1}/{args.runs} | "
                  f"Solução válida? {is_valid} | OF: {real_of:.4f} | {elapsed_time:.1f}s")

            data = {
                'Instance': [instance],
                'Run': [run],
                'Seed': [run],
                'Feasible': [is_valid],
                'OF_bks': [real_of],
                'OF_punished': [best_known_of_punished],
                'Elapsed_time': [elapsed_time],
                'Itters_done': [len(history) - 1],
                'Cluster_order': [cluster_order],
                'BKS': [best_known_solution],
            }
            save_data(pd.DataFrame(data), runs_file)

    runs_df = pd.read_csv(runs_file)
    summary = summarize(runs_df, os.path.join(args.data, 'bks.csv'))
    summary.to_csv(os.path.join(args.out, 'summary.csv'), index=False)

    print("\n-------------------------------------")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == '__main__':
    main()
