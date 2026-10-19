'''
GRÁFICOS

- instância (cidades coloridas por cluster) e rota
- curva de convergência
'''

import numpy as np
import matplotlib.pyplot as plt


def plot_route(coords, cluster, solution=None, ax=None, title=None):
    """Plota as cidades coloridas pelo cluster e, se passada, a rota (fechando o ciclo)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    clusters = np.unique(cluster)
    cmap = plt.get_cmap('tab20' if len(clusters) > 10 else 'tab10')

    if solution is not None:
        route = np.append(solution, solution[0])
        ax.plot(coords[route, 0], coords[route, 1], color='0.55', lw=1, zorder=1)

    for idx, c in enumerate(clusters):
        mask = cluster == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=28, color=cmap(idx % cmap.N),
                   edgecolor='white', linewidth=0.5, zorder=2, label=f'cluster {c}')

    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)
    if len(clusters) <= 10:
        ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

    return ax


def plot_convergence(histories, ax=None, labels=None, title='Convergência'):
    """Plota a melhor OF por iteração (uma linha por run)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    for i, history in enumerate(histories):
        ax.plot(history, lw=1.5, alpha=0.8, label=None if labels is None else labels[i])

    ax.set_xlabel('iteração')
    ax.set_ylabel('melhor OF')
    ax.set_title(title, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    if labels is not None:
        ax.legend(frameon=False)

    return ax
