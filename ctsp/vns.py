'''
VNS FUNCTIONS
'''

from .utils import calc_of
from .neighborhoods import neighbors_change_index, neighbors_insertion, neighbors_2opt


def VNS(solution, tsp_distance, itters=10):
    '''
    Busca em vizinhança variável com três vizinhanças (troca, inserção e 2-opt).

    Algoritmo:

    - Inicializa a of da solução de entrada
    - Testa a primeira vizinhança (troca)
    - Se não tiver melhoria testa a segunda vizinhança (inserção)
    - Se não tiver melhoria testa a terceira vizinhança (2-opt)
    - Se alguma das vizinhanças melhorou, volta para a primeira e repete (até `itters` vezes)
    - Se nenhuma vizinhança melhorou, a solução é ótimo local das três e o loop para
    - Retorna a melhor solução e a sua of

    Parâmetros:
    - solution: solução de entrada
    - tsp_distance: matriz de custos (penalizada)
    - itters: quantidade máxima de ciclos troca -> inserção -> 2-opt
    '''
    solution = list(solution)
    of = calc_of(solution, tsp_distance)

    while itters > 0:

        ## primeira vizinhança: troca de uma cidade aleatória com todas as outras (best improvement)
        neighbor, neighbor_of = neighbors_change_index(solution, tsp_distance)

        ## a segunda vizinhança só será chamada quando nenhum vizinho da primeira obtiver melhora na of
        call_second_neighbor = neighbor_of >= of

        if not call_second_neighbor:
            solution, of = neighbor, neighbor_of

        else:
            ## segunda vizinhança: inserção (best improvement)
            neighbor, delta = neighbors_insertion(solution, tsp_distance)
            has_improvement = delta < -1e-12

            if has_improvement:
                solution = neighbor
                of = of + delta
            else:
                ## terceira vizinhança: 2-opt (desfaz cruzamentos)
                neighbor, delta = neighbors_2opt(solution, tsp_distance)
                has_improvement = delta < -1e-12

                if has_improvement:
                    solution = neighbor
                    of = of + delta
                else:
                    # nenhuma das três vizinhanças melhorou: ótimo local, para o loop
                    break

        itters -= 1

    ## recalcula a of do zero para não acumular erro de ponto flutuante dos deltas
    return (solution, calc_of(solution, tsp_distance))
