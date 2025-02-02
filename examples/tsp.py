from random import shuffle
from permutenm import GNMPermutationOptimizer

N = 5

cost_matrix = [
    [0, 2.24, 4.12, 6.40, 5.39],
    [2.24, 0, 3.16, 4.47, 3.16],
    [4.12, 3.16, 0, 3.16, 4.47],
    [6.40, 4.47, 3.16, 0, 3.16],
    [5.39, 3.16, 4.47, 3.16, 0],
]


def tsp_cost_function(permutation):
    cost = 0
    for i in range(N - 1):
        cost += cost_matrix[permutation[i]][permutation[i + 1]]
    cost += cost_matrix[permutation[-1]][permutation[0]]
    return cost


opt = GNMPermutationOptimizer(tsp_cost_function, verbose=True)

initial_pop = []
for _ in range(N + 1):
    permutation = list(range(N))
    shuffle(permutation)
    initial_pop.append(permutation)

result = opt.minimize(initial_pop)
print(f"Best permutation: {' '.join(map(str, result.best_member))}")
print(f"Best score: {result.best_score}")
