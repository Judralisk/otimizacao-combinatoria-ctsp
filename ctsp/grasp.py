'''
GRASP para gerar as soluções iniciais
'''

import random
import numpy as np


def generate_initial_solutions_GRASP(tsp_distance, n_solutions, alphas):
    """
    Gera n soluções iniciais usando a fase construtiva do GRASP.

    A cada passo, a partir da cidade atual:
    - calcula o custo para todas as cidades ainda não visitadas
    - monta a RCL (Lista de Candidatos Restritos) com as cidades de custo <= c_min + alpha * (c_max - c_min)
    - sorteia a próxima cidade dentro da RCL

    alpha = 0 -> totalmente guloso (vizinho mais próximo)
    alpha = 1 -> totalmente aleatório

    Parâmetros:
    - tsp_distance: matriz de custos do problema (pode ser a penalizada).
    - n_solutions: número de soluções iniciais a serem geradas.
    - alphas: lista de valores de alpha para controlar aleatoriedade e gulodice.

    Retorno:
    - solutions: lista com n soluções iniciais geradas.
    """
    # Quantidade de cidades
    cities_quantity = tsp_distance.shape[0]

    solutions = []

    # Iterando sobre o número de soluções a serem geradas
    for _ in range(n_solutions):
        # Selecionando o alpha aleatoriamente
        alpha = random.choice(alphas)

        ## cidade inicial aleatória para diversificar as soluções
        current_city = random.randrange(cities_quantity)
        solution = [current_city]

        ## máscara das cidades ainda não visitadas
        not_visited = np.ones(cities_quantity, dtype=bool)
        not_visited[current_city] = False

        # Construção do caminho até que todas as cidades sejam visitadas
        while len(solution) < cities_quantity:
            candidates = np.flatnonzero(not_visited)
            candidates_cost = tsp_distance[current_city, candidates]

            c_min = candidates_cost.min()
            c_max = candidates_cost.max()

            # Calculando o L_value com base no alpha
            L_value = c_min + alpha * (c_max - c_min)

            # Construção da RCL
            rcl = candidates[candidates_cost <= L_value]

            # Escolha aleatória dentro da RCL
            current_city = int(random.choice(rcl))
            solution.append(current_city)
            not_visited[current_city] = False

        solutions.append(solution)

    return solutions


def generate_initial_population(solutions_quantity, cities_quantity):
    """
    Gera soluções iniciais totalmente aleatórias (permutações das cidades).
    Útil como baseline para comparar com o GRASP.
    """
    population = []
    for _ in range(solutions_quantity):
        s = list(range(cities_quantity))
        random.shuffle(s)
        population.append(s)

    return population
