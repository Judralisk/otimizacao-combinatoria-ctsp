'''
UTILIDADES

- leitura das instâncias
- matriz de custo com penalização de troca de cluster
- cálculo da função objetivo
- verificação de viabilidade
- salvando os resultados
'''

import os
import numpy as np
import pandas as pd


def read_tsp_file(filename):
    """
    Lê uma instância no formato CSV usado no projeto:
    colunas X, Y, cluster e depois a matriz de distâncias (0, 1, ..., n-1).

    Retorno:
    - coords: array (n, 2) com as coordenadas
    - cluster: array (n,) com o cluster de cada cidade
    - tsp_distance: matriz de distâncias (n, n)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    data = pd.read_csv(filename)
    coords = data[['X', 'Y']].to_numpy(dtype=float)
    cluster = data['cluster'].to_numpy()
    tsp_distance = data.drop(columns=['X', 'Y', 'cluster']).to_numpy(dtype=float)

    return coords, cluster, tsp_distance


def build_cost_matrix(tsp_distance, cluster, M=None):
    """
    Monta a matriz de custo penalizada: C[i, j] = d[i, j] + M se i e j estão em clusters diferentes.

    Com a penalidade dentro da matriz, a OF penalizada vira só a soma das arestas do ciclo,
    e todas as metaheurísticas trabalham em cima de uma única matriz.

    Sobre o M:
    num ciclo viável cada cluster é visitado em um bloco só, então existem exatamente k trocas
    de cluster (k = nº de clusters). Qualquer rota inviável tem pelo menos k + 1 trocas.
    Se M > n * max(d), qualquer rota viável tem OF penalizada menor que qualquer inviável,
    ou seja, o ótimo do problema penalizado é o ótimo do CTSP (big-M exato).

    Parâmetros:
    - tsp_distance: matriz de distâncias original
    - cluster: array com o cluster de cada cidade
    - M: constante de penalização (None -> n * max(d), que garante a propriedade acima)

    Retorno:
    - cost_matrix: matriz penalizada
    - M: valor de M utilizado
    """
    n = tsp_distance.shape[0]
    if M is None:
        M = n * tsp_distance.max()

    different_cluster = cluster[:, None] != cluster[None, :]
    cost_matrix = tsp_distance + M * different_cluster

    return cost_matrix, M


def calc_of(solution, distance_matrix):
    """
    Calcula a função objetivo (comprimento do ciclo, voltando para a cidade inicial).

    Se `distance_matrix` for a matriz penalizada, retorna a OF penalizada;
    se for a matriz original, retorna a distância real.
    """
    solution = np.asarray(solution)
    ## next_cities é a solução "rodada" uma posição: cada cidade aponta para a próxima, e a última volta pra primeira
    next_cities = np.roll(solution, -1)
    return float(distance_matrix[solution, next_cities].sum())


def check_solution_viability(solution, cluster_array):
    """
    Verifica se a solução é viável, ou seja, se cada cluster é visitado em um único bloco contíguo.

    A verificação é CÍCLICA: [1, 0, 0, 0, 1, 1] é viável, pois a rota volta da última para a primeira
    cidade e o cluster 1 fica contíguo no ciclo (é só uma rotação de [1, 1, 1, 0, 0, 0]).

    Parâmetros:
    - solution: lista de cidades na ordem de visitação (ex: [0, 1, 2, 3, 4])
    - cluster_array: array onde o valor de cada índice representa o número do cluster
      ao qual a cidade correspondente pertence.

    Retorno:
    - True se a solução é viável, False caso contrário.
    """
    cluster_order = np.asarray(cluster_array)[np.asarray(solution)]
    clusters_quantity = len(np.unique(cluster_order))

    ## com um cluster só não existe troca
    if clusters_quantity == 1:
        return True

    ## quantidade de trocas de cluster no ciclo (incluindo a volta da última pra primeira)
    cluster_changes = np.sum(cluster_order != np.roll(cluster_order, -1))

    # viável <=> exatamente uma entrada e uma saída por cluster
    return bool(cluster_changes == clusters_quantity)


def is_permutation(solution, cities_quantity):
    """Confere se a solução visita todas as cidades exatamente uma vez."""
    return sorted(solution) == list(range(cities_quantity))


def save_data(data, filepath):
    file_exists = os.path.exists(filepath)
    if file_exists:
        data.to_csv(filepath, index=False, mode='a', header=False)
    else:
        data.to_csv(filepath, index=False)
