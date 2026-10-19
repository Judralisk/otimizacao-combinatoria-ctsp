'''
Testes de correção das funções principais.  Rodar com:  python -m pytest -q
'''

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ctsp import (build_cost_matrix, calc_of, check_solution_viability, is_permutation,
                  generate_initial_solutions_GRASP, neighbors_insertion, neighbors_change_index,
                  crossover_2_cuts, crossover_1_cut, mutation, VNS, SA, GA, run_hybrid, neighbors_2opt)
from ctsp.neighborhoods import random_insertion_move, apply_insertion
from scripts.instances import generate_synthetic_instance


def small_instance(n=30, k=4, seed=1):
    df = generate_synthetic_instance(n, k, seed=seed)
    cluster = df['cluster'].to_numpy()
    distance = df.drop(columns=['X', 'Y', 'cluster']).to_numpy()
    cost, _ = build_cost_matrix(distance, cluster)
    return distance, cluster, cost


def test_calc_of_closes_the_cycle():
    d = np.array([[0, 1, 5], [1, 0, 2], [5, 2, 0]], dtype=float)
    assert calc_of([0, 1, 2], d) == 1 + 2 + 5


def test_viability_is_cyclic():
    cluster = np.array([1, 0, 0, 0, 1, 1])
    assert check_solution_viability([0, 1, 2, 3, 4, 5], cluster)        # 1 | 0 0 0 | 1 1  -> ciclo contíguo
    assert not check_solution_viability([0, 1, 4, 2, 3, 5], cluster)    # 1 0 1 0 0 1      -> volta ao cluster


def test_big_M_feasible_always_beats_infeasible():
    distance, cluster, cost = small_instance()
    random.seed(0)
    feasible, infeasible = [], []
    for _ in range(2000):
        s = list(range(len(cluster)))
        random.shuffle(s)
        (feasible if check_solution_viability(s, cluster) else infeasible).append(calc_of(s, cost))
    ## monta algumas viáveis ordenando por cluster
    for _ in range(50):
        order = list(np.unique(cluster)); random.shuffle(order)
        s = []
        for c in order:
            cities = list(np.flatnonzero(cluster == c)); random.shuffle(cities); s += cities
        feasible.append(calc_of(s, cost))
    assert max(feasible) < min(infeasible)


def test_grasp_does_not_modify_input_and_returns_permutations():
    _, _, cost = small_instance()
    before = cost.copy()
    sols = generate_initial_solutions_GRASP(cost, 10, [0, 0.5, 1])
    assert np.array_equal(before, cost)
    assert all(is_permutation(s, cost.shape[0]) for s in sols)


def test_insertion_delta_matches_brute_force():
    _, _, cost = small_instance(n=15)
    random.seed(3)
    s = list(range(15)); random.shuffle(s)
    base = calc_of(s, cost)

    ## força bruta: todos os vizinhos de inserção
    brute_best = min(calc_of(apply_insertion(s, i, k), cost)
                     for i in range(15) for k in range(14) if apply_insertion(s, i, k) != s)
    neighbor, delta = neighbors_insertion(s, cost)
    assert np.isclose(calc_of(neighbor, cost), brute_best)
    assert np.isclose(base + delta, calc_of(neighbor, cost))

    for _ in range(500):
        i, k, d = random_insertion_move(s, cost)
        assert np.isclose(base + d, calc_of(apply_insertion(s, i, k), cost))


def test_swap_neighborhood_returns_correct_of():
    _, _, cost = small_instance()
    s = list(range(cost.shape[0]))
    for _ in range(20):
        neighbor, of = neighbors_change_index(s, cost)
        assert np.isclose(of, calc_of(neighbor, cost))


def test_crossovers_and_mutation_keep_permutation():
    random.seed(0)
    n = 20
    for _ in range(500):
        p1 = list(range(n)); random.shuffle(p1)
        p2 = list(range(n)); random.shuffle(p2)
        assert is_permutation(crossover_2_cuts(p1, p2, n), n)
        child = crossover_1_cut(p1, p2, n)
        assert is_permutation(child, n)
        assert is_permutation(mutation(p1), n)


def test_crossover_1_cut_preserves_order_of_parents():
    p1 = [5, 3, 8, 0, 1, 2, 4, 6, 7, 9]
    p2 = [9, 7, 6, 4, 2, 1, 0, 8, 3, 5]
    for seed in range(20):
        random.seed(seed)
        cut = random.randrange(1, 9)   # mesmo sorteio feito dentro do crossover
        random.seed(seed)
        child = crossover_1_cut(p1, p2, 10)
        assert child[:cut] == p1[:cut]
        assert child[cut:] == [c for c in p2 if c not in p1[:cut]]


def test_local_searches_never_worsen():
    _, _, cost = small_instance()
    random.seed(2)
    s = list(range(cost.shape[0])); random.shuffle(s)
    of = calc_of(s, cost)
    assert VNS(s, cost)[1] <= of + 1e-9
    assert SA(s, cost)[1] <= of + 1e-9
    _, ga_of = GA([s] * 4, cost, 0.1, 10, 4)
    assert ga_of[0] <= of + 1e-9


def test_hybrid_finds_feasible_solution():
    distance, cluster, cost = small_instance(n=40, k=5)
    sol, of, history = run_hybrid(cost, {'Max_itters': 15}, seed=42)
    assert is_permutation(sol, 40)
    assert check_solution_viability(sol, cluster)
    assert all(a >= b for a, b in zip(history, history[1:]))   # convergência monotônica
    ## reprodutibilidade com a mesma semente
    assert run_hybrid(cost, {'Max_itters': 15}, seed=42)[1] == of


def test_2opt_delta_matches_brute_force():
    _, _, cost = small_instance(n=15)
    random.seed(5)
    s = list(range(15)); random.shuffle(s)
    base = calc_of(s, cost)
    brute_best = np.inf
    for i in range(15):
        for j in range(i + 2, 15):
            if i == 0 and j == 14:
                continue
            nb = s[:]; nb[i + 1:j + 1] = nb[i + 1:j + 1][::-1]
            brute_best = min(brute_best, calc_of(nb, cost))
    neighbor, delta = neighbors_2opt(s, cost)
    assert np.isclose(calc_of(neighbor, cost), brute_best)
    assert np.isclose(base + delta, brute_best)
