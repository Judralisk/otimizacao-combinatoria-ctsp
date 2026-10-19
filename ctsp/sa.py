'''
SIMULATED ANNEALING FUNCTIONS
'''

import math
import random
import numpy as np

from .utils import calc_of
from .neighborhoods import random_insertion_move, apply_insertion


def initial_temperature(solution, tsp_distance, M=None, acceptance_probability=0.5, samples=500):
    """
    Calibra a temperatura inicial a partir dos dados: amostra movimentos aleatórios e escolhe T_max
    tal que um movimento de piora "típico" seja aceito com a probabilidade desejada no começo:

        exp(-delta_tipico / T_max) = p   =>   T_max = -delta_tipico / ln(p)

    Com a penalização M na matriz, a maioria dos movimentos aleatórios de inserção quebra um cluster
    e tem delta da ordem de M (com k clusters, só ~1/k das inserções caem no mesmo bloco).
    Se esses deltas entrassem na conta, T_max ficaria na escala de M: o SA passaria quase todo o tempo
    aceitando rotas inviáveis e terminaria numa temperatura alta demais para refinar as distâncias.
    Por isso, quando M é informado, só entram os deltas < M / 2 (movimentos que não mudam a quantidade
    de trocas de cluster), e uso a MEDIANA, que é robusta a outliers.
    """
    positive_deltas = []
    for _ in range(samples):
        _, _, delta = random_insertion_move(solution, tsp_distance)
        if delta > 0 and (M is None or delta < M / 2):
            positive_deltas.append(delta)

    if not positive_deltas:
        return 1.0

    typical_delta = float(np.median(positive_deltas))
    return -typical_delta / math.log(acceptance_probability)


def SA(solution, tsp_distance, T_max=None, T_min=None, alpha=0.9, k_max=None, M=None):
    """
    Simulated Annealing com vizinhança de inserção.

    Algoritmo:
    - a cada temperatura T, faz k_max movimentos aleatórios de inserção
    - se o movimento melhora (delta <= 0) aceita sempre
    - se piora, aceita com probabilidade exp(-delta / T) (critério de Metropolis)
    - resfriamento geométrico: T = alpha * T, até T < T_min

    Parâmetros:
    - solution: solução inicial
    - tsp_distance: matriz de custos (penalizada)
    - T_max: temperatura inicial (None -> calibrada automaticamente)
    - T_min: temperatura final (None -> T_max / 1000)
    - alpha: taxa de resfriamento
    - k_max: movimentos por temperatura (None -> número de cidades)
    - M: penalização usada na matriz (só para calibrar T_max)

    Retorno:
    - best_SOL, best_OF
    """
    current_sol = list(solution)
    current_of = calc_of(current_sol, tsp_distance)

    best_SOL = current_sol[:]
    best_OF = current_of

    if T_max is None:
        T_max = initial_temperature(current_sol, tsp_distance, M)
    if T_min is None:
        T_min = T_max / 1000
    if k_max is None:
        k_max = len(current_sol)

    T = T_max

    while T > T_min:
        for _ in range(k_max):
            # sorteia um vizinho de inserção e calcula o delta sem recalcular a OF inteira
            i, k, delta_E = random_insertion_move(current_sol, tsp_distance)

            if delta_E <= 0 or random.random() < math.exp(-delta_E / T):
                current_sol = apply_insertion(current_sol, i, k)
                current_of += delta_E

                if current_of < best_OF - 1e-12:
                    best_SOL = current_sol[:]
                    best_OF = current_of

        T = alpha * T

    ## recalcula a of do zero para não acumular erro de ponto flutuante dos deltas
    return best_SOL, calc_of(best_SOL, tsp_distance)
