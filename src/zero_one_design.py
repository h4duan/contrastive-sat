import itertools
import numpy as np

def right_index(j):
    if j < 21:
        return j
    else:
        return j - 21


def generate_matrix():
    incidence_matrix = np.zeros((21, 21))
    difference_set = [0, 2, 7, 8, 11]
    elements = np.range(105)
    np.random.shuffle(elements)
    polarity = np.random.choice([-1, 1], 105, replace=True)
    elements = elements * polarity
    for i in range(21):
        for ij, j in enumerate(difference_set):
            incidence_matrix[i][right_index(j)] = elements[i * 5 + ij]
    return incidence_matrix


def generate_instance():
    incidence = generate_matrix()
    clauses = []
    need = np.zeros(21) + 2
    avai = np.zeros(21) + 2
    index_need = np.random.choice(range(21), 11, replace=False)
    index_avai = np.random.choice(range(21), 10, replace=False)
    for i in range(21):
        if i in index_need:
            need[i] += 1
        if i in index_avai:
            avai[i] += 1
    need = need.astype(int)
    avai = avai.astype(int)
    for r in range(21):
        subset_size = 5 - need[r] + 1
        row = incidence[r]
        clauses += list(map(list, itertools.combinations(row, subset_size)))
    for c in range(21):
        subset_size = 5 - need[r] + 1
        column = -incidence[:, c]
        clauses += list(map(list, itertools.combinations(column, subset_size)))
    return clauses

print(len(generate_instance()))

