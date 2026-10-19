'''
GENETIC ALGORITHM FUNCTIONS
'''

import random
import numpy as np

from .utils import calc_of


def crossover_2_cuts(parent1, parent2, cities_quantity):
    """
    Crossover de dois cortes (variação do Order Crossover - OX):
    - o filho herda do parent1 o trecho entre os dois cortes, na mesma posição
    - as posições restantes são preenchidas com as cidades que faltam, na ordem em que aparecem no parent2
    """
    cut_1, cut_2 = sorted(random.sample(range(1, cities_quantity), 2))

    # initialize the sibling (-1 marca posição vazia, já que 0 também é uma cidade)
    sibling = [-1] * cities_quantity
    sibling[cut_1:cut_2] = parent1[cut_1:cut_2]
    already_in_sibling = set(sibling[cut_1:cut_2])

    posicao = 0  # index that we'll itter to put the cities
    for cidade in parent2:
        if cidade in already_in_sibling:
            continue
        while cut_1 <= posicao < cut_2:
            posicao += 1
        sibling[posicao] = cidade
        posicao += 1

    return sibling


def crossover_1_cut(solution_1, solution_2, cities_quantity):
    """
    Crossover de um corte:
    - o filho herda as primeiras cidades da solução 1 (até o corte), na mesma ordem
    - completa com as cidades que faltam na ordem em que aparecem na solução 2
    """
    crossover_cut = random.randrange(1, cities_quantity - 1)

    first_n_cities = list(solution_1[:crossover_cut])
    used_cities = set(first_n_cities)

    ## atualizando os valores da solução 2, tirando as cidades que já foram utilizadas (mantendo a ordem!)
    solution_2_remaining = [cidade for cidade in solution_2 if cidade not in used_cities]

    # como estou utilizando o tipo lista, o operador "+" concatena as listas
    crossover_result = first_n_cities + solution_2_remaining

    return crossover_result


def mutation(solution):
    """Troca uma cidade de posição com a vizinha (a última troca com a primeira, já que a rota é um ciclo)."""
    solution = list(solution)
    chosen_index = random.randrange(len(solution))
    next_index = (chosen_index + 1) % len(solution)
    solution[chosen_index], solution[next_index] = solution[next_index], solution[chosen_index]
    return solution


def selection(solutions_list, distance_matrix, number_of_siblings):
    """
    Seleciona as `number_of_siblings` melhores soluções DISTINTAS (elitismo).

    Retorno:
    - best_solutions: lista das melhores soluções
    - best_costs: OF de cada uma
    """
    costs = []
    filtered_solutions = set()
    for solution in solutions_list:
        this_solution = tuple(solution)
        # caso essa solução já esteja adicionada na seleção, ele adiciona um custo infinito para que ela não seja selecionada mais que uma vez quando utilizamos o np.argsort na seleção das menores funções custo
        if this_solution in filtered_solutions:
            costs.append(np.inf)
            continue
        costs.append(calc_of(solution, distance_matrix))
        filtered_solutions.add(this_solution)

    best_indexes = np.argsort(costs, kind='stable')[:number_of_siblings]  ## this function returns the indices that would sort the values
    best_solutions = [list(solutions_list[i]) for i in best_indexes]
    best_costs = [costs[i] for i in best_indexes]
    return best_solutions, best_costs


def GA(Initial_Solutions_List, tsp_distance, Mutation_probability, generate_siblings_quantity, output_solutions_quantity):
    '''
    Parâmetros:

    - Initial_Solutions_List: soluções iniciais dadas
    - tsp_distance: matriz de custos (penalizada)
    - Mutation_probability: probabilidade de o filho receber uma mutação
    - generate_siblings_quantity: quantidade total de filhos que serão gerados no passo de reprodução
    - output_solutions_quantity: quantas soluções serão retornadas no final do algoritmo

    Algoritmo:
    Dada a quantidade inicial de soluções:
    - são escolhidos os pais (a melhor metade)
    - crossover dos pais, alternando 2 cortes e 1 corte (até gerar "generate_siblings_quantity" filhos)
    - mutação com probabilidade "Mutation_probability"
    - seleção das melhores soluções entre pais + filhos
    '''
    cities_quantity = len(Initial_Solutions_List[0])
    initial_solutions_quantity = len(Initial_Solutions_List)

    ## vou escolher metade das soluções atuais para reproduzir (a melhor metade)
    ## A FUNÇÃO SELECTION COMO FOI DEFINIDA SERVE PARA ESSE ESCOLHER OS PAIS TAMBÉM!!
    parents_list, _ = selection(Initial_Solutions_List, tsp_distance, max(2, initial_solutions_quantity // 2))
    parents_quantity = len(parents_list)

    siblings_list = []

    ## CROSSOVER DOS PAIS (enquanto não tivermos a quantidade de filhos pré-determinada, o algoritmo gera filhos com crossover de 2 e 1 cortes e para assim que tivermos a quantidade de filhos determinada)
    while len(siblings_list) < generate_siblings_quantity:
        for i in range(parents_quantity):
            ## o último pai cruza com o primeiro
            parent_1 = parents_list[i]
            parent_2 = parents_list[(i + 1) % parents_quantity]

            # crossover com 2 cortes
            if len(siblings_list) < generate_siblings_quantity:
                siblings_list.append(crossover_2_cuts(parent_1, parent_2, cities_quantity))

            ## crossover com 1 corte
            if len(siblings_list) < generate_siblings_quantity:
                siblings_list.append(crossover_1_cut(parent_1, parent_2, cities_quantity))

    ## MUTATION
    for i in range(len(siblings_list)):
        if random.random() <= Mutation_probability:
            siblings_list[i] = mutation(siblings_list[i])

    ## adicionando os parents pq eles também podem ser selecionados caso a of deles seja melhor
    siblings_list = [list(s) for s in Initial_Solutions_List] + siblings_list

    ## SELECTION (vai retornar a mesma quantidade de soluções de entrada!!!)
    solutions_list, of_list = selection(siblings_list, tsp_distance, output_solutions_quantity)

    return solutions_list, of_list
