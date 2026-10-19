'''
ALGORITMO HÍBRIDO: GRASP + (VNS, SA, GA)

- GRASP gera a população inicial
- a cada iteração, cada solução passa pelo VNS e pelo SA, e a população inteira passa pelo GA
- junta tudo (população + VNS + SA + GA) e mantém as `Solutions_Quantity` melhores soluções distintas
'''

import time
import random
import numpy as np

from .grasp import generate_initial_solutions_GRASP
from .vns import VNS
from .sa import SA
from .ga import GA, selection


DEFAULT_PARAMS = {
    'Solutions_Quantity': 10,
    'Max_itters': 150,
    'alphas': [0.25, 0.5, 0.75],
    ## GA PARAMS
    'Mutation_probability': 0.10,
    'generate_siblings_quantity': 76,
    ## VNS / SA PARAMS
    'VNS_itters': 10,
    'SA_alpha': 0.9,
    ## penalização usada na matriz (o SA usa para calibrar a temperatura inicial)
    'M': None,
    ## critério de parada extra: para se a melhor OF não melhorar por N iterações seguidas (None = desligado)
    'Max_itters_without_improvement': 30,
    ## limite de tempo em segundos (None = desligado)
    'time_limit': None,
}


def set_seed(seed):
    """Fixa a semente dos dois geradores usados (random e numpy) para a run ser reprodutível."""
    random.seed(seed)
    np.random.seed(seed)


def run_hybrid(tsp_distance, params=None, seed=None, verbose=False):
    """
    Roda o algoritmo híbrido uma vez.

    Parâmetros:
    - tsp_distance: matriz de custos (penalizada)
    - params: dicionário de parâmetros (o que não for passado usa DEFAULT_PARAMS)
    - seed: semente da run
    - verbose: printa a melhor OF a cada iteração

    Retorno:
    - best_SOL: melhor solução encontrada
    - best_OF: OF (penalizada) da melhor solução
    - history: melhor OF ao final de cada iteração (curva de convergência)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    if seed is not None:
        set_seed(seed)

    start_time = time.time()
    Solutions_Quantity = p['Solutions_Quantity']

    ## GERANDO AS SOLUÇÕES INICIAIS
    solutions_list = generate_initial_solutions_GRASP(tsp_distance, Solutions_Quantity, p['alphas'])
    solutions_list, of_list = selection(solutions_list, tsp_distance, Solutions_Quantity)

    # Inicializando a melhor OF e solução
    best_SOL, best_OF = solutions_list[0], of_list[0]
    history = [best_OF]
    itters_without_improvement = 0

    itters = 0
    while itters < p['Max_itters']:
        VNS_solutions_list = []
        SA_Solutions_list = []

        for solution in solutions_list:
            ### Rodando a parte do VNS
            VNS_solution, _ = VNS(solution, tsp_distance, p['VNS_itters'])
            VNS_solutions_list.append(VNS_solution)

            ### Rodando a parte do SA
            SA_Solution, _ = SA(solution, tsp_distance, alpha=p['SA_alpha'], M=p['M'])
            SA_Solutions_list.append(SA_Solution)

        ### Rodando a parte do GA
        ## GA já recebe como entrada todas as soluções
        GA_solutions, _ = GA(solutions_list, tsp_distance, p['Mutation_probability'],
                             p['generate_siblings_quantity'], Solutions_Quantity)

        # atualizando a lista de soluções com as melhores encontradas na iteração + as que já tinham
        ## (a seleção recalcula a OF de cada solução, então OF e solução nunca se desencontram)
        all_solutions = solutions_list + VNS_solutions_list + SA_Solutions_list + GA_solutions
        solutions_list, of_list = selection(all_solutions, tsp_distance, Solutions_Quantity)

        # atualizando a melhor resposta
        if of_list[0] < best_OF - 1e-12:
            best_OF = of_list[0]
            best_SOL = solutions_list[0]
            itters_without_improvement = 0
        else:
            itters_without_improvement += 1

        history.append(best_OF)
        itters += 1

        if verbose:
            print(f"itter {itters:>4} | best_OF: {best_OF:.4f}")

        ## critérios de parada extras
        if (p['Max_itters_without_improvement'] is not None
                and itters_without_improvement >= p['Max_itters_without_improvement']):
            break
        if p['time_limit'] is not None and time.time() - start_time > p['time_limit']:
            break

    return best_SOL, best_OF, history
