'''
VIZINHANÇAS

Gerar todos os vizinhos e recalcular a OF de cada um do zero custa O(n) por vizinho; na inserção
são n * (n - 1) vizinhos, ou seja, O(n³) por chamada.

Aqui a avaliação é feita pela variação (delta) da OF, vetorizada com numpy:
só as arestas que mudam entram na conta, o que deixa a vizinhança de inserção em O(n²).
'''

import random
import numpy as np


def neighbors_change_index(solution, cost_matrix):
    """
    Vizinhança de troca (swap): escolhe uma cidade aleatória e troca de posição com todas as outras.

    Parâmetros:
    - solution: lista representando a sequência de cidades.
    - cost_matrix: matriz de custos.

    Retorno:
    - best_neighbor: melhor vizinho encontrado
    - best_neighbor_of: OF do melhor vizinho
    """
    tour = np.asarray(solution)
    n = len(tour)

    # Select a random city
    random_city = random.randrange(n)
    others = np.delete(np.arange(n), random_city)

    ## monta todos os vizinhos de uma vez: cada linha é a solução com a troca (random_city, i)
    neighbors = np.tile(tour, (n - 1, 1))
    rows = np.arange(n - 1)
    neighbors[rows, random_city] = tour[others]
    neighbors[rows, others] = tour[random_city]

    ## OF de todos os vizinhos vetorizada (cada vizinho tem só 4 arestas diferentes, mas em O(n²) total já fica rápido)
    neighbors_of = cost_matrix[neighbors, np.roll(neighbors, -1, axis=1)].sum(axis=1)

    best = int(np.argmin(neighbors_of))
    return neighbors[best].tolist(), float(neighbors_of[best])


def neighbors_insertion(solution, cost_matrix):
    """
    Vizinhança de inserção: remove a cidade da posição i e reinsere em outra posição do ciclo.
    Avalia todos os movimentos pelo delta e retorna o melhor (best improvement).

    Para remover a cidade c (entre p e q) e inserir entre a e b:
        delta = [C(a,c) + C(c,b) - C(a,b)] - [C(p,c) + C(c,q) - C(p,q)]

    Retorno:
    - best_neighbor: melhor vizinho
    - best_delta: variação da OF em relação à solução de entrada (negativo = melhoria)
    """
    tour = np.asarray(solution)
    n = len(tour)

    ## custo de remover cada cidade da rota
    prev_city = np.roll(tour, 1)
    next_city = np.roll(tour, -1)
    removal_gain = (cost_matrix[prev_city, tour] + cost_matrix[tour, next_city]
                    - cost_matrix[prev_city, next_city])

    ## reduced[i] = rota sem a posição i (matriz n x (n-1))
    k = np.arange(n - 1)
    reduced_idx = k[None, :] + (k[None, :] >= np.arange(n)[:, None])
    reduced = tour[reduced_idx]
    reduced_prev = np.roll(reduced, 1, axis=1)

    city = tour[:, None]
    insertion_cost = (cost_matrix[reduced_prev, city] + cost_matrix[city, reduced]
                      - cost_matrix[reduced_prev, reduced])

    delta = insertion_cost - removal_gain[:, None]

    ## reinserir no mesmo lugar (entre p e q) devolve a própria solução, então não conta como vizinho
    same_place = (reduced_prev == prev_city[:, None]) & (reduced == next_city[:, None])
    delta[same_place] = np.inf

    i, k_best = np.unravel_index(np.argmin(delta), delta.shape)
    best_neighbor = reduced[i].tolist()
    best_neighbor.insert(int(k_best), int(tour[i]))

    return best_neighbor, float(delta[i, k_best])


def random_insertion_move(solution, cost_matrix):
    """
    Sorteia UM movimento de inserção e calcula o delta em O(1). Usado no Simulated Annealing.

    Retorno:
    - i: posição da cidade removida
    - k: posição de inserção na rota reduzida (sem a cidade i)
    - delta: variação da OF
    """
    n = len(solution)
    i = random.randrange(n)
    k = random.randrange(n - 1)

    city = solution[i]
    p = solution[i - 1]
    q = solution[(i + 1) % n]

    ## posição k da rota reduzida corresponde à posição k (antes de i) ou k + 1 (depois de i) na rota original
    def reduced_at(pos):
        pos = pos % (n - 1)
        return solution[pos] if pos < i else solution[pos + 1]

    a = reduced_at(k - 1)
    b = reduced_at(k)

    if a == p and b == q:
        return i, k, 0.0

    delta = ((cost_matrix[a, city] + cost_matrix[city, b] - cost_matrix[a, b])
             - (cost_matrix[p, city] + cost_matrix[city, q] - cost_matrix[p, q]))

    return i, k, float(delta)


def apply_insertion(solution, i, k):
    """Aplica o movimento de inserção (remove a posição i e insere na posição k da rota reduzida)."""
    neighbor = solution[:]
    city = neighbor.pop(i)
    neighbor.insert(k, city)
    return neighbor


def neighbors_2opt(solution, cost_matrix):
    """
    Vizinhança 2-opt: remove duas arestas (a, b) e (c, d) e reconecta como (a, c) e (b, d),
    invertendo o trecho entre b e c. É a vizinhança que desfaz cruzamentos na rota.

        delta = C(a,c) + C(b,d) - C(a,b) - C(c,d)        (vale para matriz simétrica)

    Avalia todos os pares (i, j) vetorizado e retorna o melhor (best improvement).

    Retorno:
    - best_neighbor: melhor vizinho
    - best_delta: variação da OF (negativo = melhoria)
    """
    tour = np.asarray(solution)
    n = len(tour)

    a = tour[:, None]
    b = np.roll(tour, -1)[:, None]
    c = tour[None, :]
    d = np.roll(tour, -1)[None, :]

    delta = cost_matrix[a, c] + cost_matrix[b, d] - cost_matrix[a, b] - cost_matrix[c, d]

    ## só pares com i < j - 1 (arestas não adjacentes); o par (0, n-1) também é adjacente no ciclo
    i_idx, j_idx = np.triu_indices(n, k=2)
    valid = ~((i_idx == 0) & (j_idx == n - 1))
    i_idx, j_idx = i_idx[valid], j_idx[valid]

    pair_delta = delta[i_idx, j_idx]
    best = int(np.argmin(pair_delta))
    i, j = int(i_idx[best]), int(j_idx[best])

    best_neighbor = tour.tolist()
    best_neighbor[i + 1:j + 1] = best_neighbor[i + 1:j + 1][::-1]

    return best_neighbor, float(pair_delta[best])
