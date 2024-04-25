import numpy as np
from scipy.linalg import eigh


# резмерность глобальной матрицы жесткости (2 * maxnode, 2 * maxnode) поскольку в каждом узле по 2 DOF
def build_global_stiffness_matrix(elements, maxnode):  # составляем глобальную матрицу жесткости
    local_stiffness = []
    for el in elements:
        local_stiffness.append(np.zeros((maxnode, maxnode)))
        row_start = 1 * el.id
        col_start = 1 * el.id
        local_stiffness[el.id][row_start:row_start + 2, col_start:col_start + 2] = el.kel

    global_stiffness = sum(local_stiffness)
    return global_stiffness


def global_stiffness_matrix_with_GU(global_stiffness):
    global_stiffness[0, :] = 0
    global_stiffness[:, 0] = 0
    global_stiffness[-1, :] = 0
    global_stiffness[:, -1] = 0
    global_stiffness[0, 0] = 1
    global_stiffness[-1, -1] = 1

    return global_stiffness


def create_ef_stiffness(global_mass, global_stiffness, al0):
    ef_stiffness = global_stiffness + al0 * global_mass
    return ef_stiffness


def build_global_mass_matrix(elements, maxnode):  # составляем глобальную матрицу жесткости
    local_mass = []
    for el in elements:
        local_mass.append(np.zeros((1 * maxnode, 1 * maxnode)))
        row_start = 1 * el.id
        col_start = 1 * el.id
        local_mass[el.id][row_start:row_start + 2, col_start:col_start + 2] = el.mel

    global_mass = sum(local_mass)
    return global_mass


def create_global_force(global_force, f_ampl):
    global_force[-4, 0] = f_ampl
    return global_force


def create_VI_force(point_bar, global_force, dis_i_bar, k_c):
    plus_slag = -dis_i_bar
    if plus_slag <= 0:
        plus_slag = 0

    VI_force = k_c * plus_slag ** (3 / 2)
    # VI_force = 0

    # print('vel_i_before = {}'.format(vel_i_before))
    # print('vel_i[(maxnode // 2) * 2, 0] = {}'.format(vel_i_bar))
    # print('plus_slag**(3/2) = {}'.format(plus_slag**(3/2)))
    # print('VI_force = {}'.format(VI_force))

    global_force[point_bar, 0] += VI_force

    return global_force, VI_force


def create_modal_matrix(K, M):
    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigh(K, M)

    # Perform mass-orthonormalization of eigenvectors
    eigenvectors_normalized = eigenvectors / np.sqrt(np.diag(np.dot(eigenvectors.T, np.dot(M, eigenvectors))))

    # Print the eigenvalues
    print("Eigenvalues:")
    print(eigenvalues)

    print("Eigenvalues (sqrt):")
    print(np.sqrt(np.array(eigenvalues)))

    # Print the eigenvectors
    print("Eigenvectors:")
    print(eigenvectors_normalized)

    # Verify mass-orthonormality of eigenvectors
    mass_orthonormality = np.dot(eigenvectors_normalized.T, np.dot(M, eigenvectors_normalized))
    print("Mass-Orthonormality:")
    print(mass_orthonormality)

    # Verify stiffness-orthonormality of eigenvectors
    stiffness_orthonormality = np.dot(eigenvectors_normalized.T, np.dot(K, eigenvectors_normalized))
    print("Stiffness-Orthonormality:")
    print(stiffness_orthonormality)

    return eigenvalues, eigenvectors_normalized
