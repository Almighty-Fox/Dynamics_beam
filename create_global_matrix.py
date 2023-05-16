import numpy as np


# резмерность глобальной матрицы жесткости (2 * maxnode, 2 * maxnode) поскольку в каждом узле по 2 DOF
def build_global_stiffness_matrix(elements, maxnode):  # составляем глобальную матрицу жесткости
    local_stiffness = []
    for el in elements:
        local_stiffness.append(np.zeros((2 * maxnode, 2 * maxnode)))
        row_start = col_start = 2 * el.id
        local_stiffness[el.id][row_start:row_start + 4, col_start:col_start + 4] = el.kel
        # print(local_stiffness[el.id])
        # print('---------------')

    global_stiffness = sum(local_stiffness)
    # print(global_stiffness)
    return global_stiffness


def build_global_mass_matrix(elements, maxnode):  # составляем глобальную матрицу жесткости
    local_mass = []
    for el in elements:
        local_mass.append(np.zeros((2 * maxnode, 2 * maxnode)))
        row_start = col_start = 2 * el.id
        local_mass[el.id][row_start:row_start + 4, col_start:col_start + 4] = el.mel
        # print(local_stiffness[el.id])
        # print('---------------')

    global_mass = sum(local_mass)
    # print(global_mass)
    return global_mass


def global_stiffness_matrix_with_GU(global_stiffness):
    global_stiffness[0, :] = global_stiffness[:, 0] = global_stiffness[1, :] = global_stiffness[:, 1] = 0
    global_stiffness[0, 0] = global_stiffness[1, 1] = 1
    return global_stiffness


def create_global_force(maxnode):
    global_force = np.zeros((2 * maxnode, 1))
    global_force[-2, 0] = 1

    return global_force
