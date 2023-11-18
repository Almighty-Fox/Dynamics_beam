import numpy as np
from scipy.linalg import eigh


# резмерность глобальной матрицы жесткости (2 * maxnode, 2 * maxnode) поскольку в каждом узле по 2 DOF
def build_global_stiffness_matrix(elements, maxnode):  # составляем глобальную матрицу жесткости
    local_stiffness = []
    for el in elements:
        local_stiffness.append(np.zeros((2 * maxnode, 2 * maxnode)))
        row_start = 2 * el.id
        col_start = 2 * el.id
        local_stiffness[el.id][row_start:row_start + 4, col_start:col_start + 4] = el.kel

    global_stiffness = sum(local_stiffness)
    return global_stiffness


def global_stiffness_matrix_with_GU(global_stiffness):
    global_stiffness[0, :] = 0
    global_stiffness[:, 0] = 0
    global_stiffness[1, :] = 0
    global_stiffness[:, 1] = 0
    global_stiffness[0, 0] = 1
    global_stiffness[1, 1] = 1

    return global_stiffness


def create_ef_stiffness(global_mass, global_stiffness, al0):
    ef_stiffness = global_stiffness + al0 * global_mass
    return ef_stiffness


def build_global_mass_matrix(elements, maxnode):  # составляем глобальную матрицу жесткости
    local_mass = []
    for el in elements:
        local_mass.append(np.zeros((2 * maxnode, 2 * maxnode)))
        row_start = 2 * el.id
        col_start = 2 * el.id
        local_mass[el.id][row_start:row_start + 4, col_start:col_start + 4] = el.mel

    global_mass = sum(local_mass)
    return global_mass


def create_global_force(global_force, f_ampl):
    global_force[-2, 0] = f_ampl
    return global_force


def create_VI_force(global_force, point_bar, delta, dis_i_bar, vel_i_bar, vel_i_before, k_c, restitution=0.7):
    plus_slag = -dis_i_bar - delta
    if plus_slag <= 0:
        plus_slag = 0
    # VI_force = k_c * plus_slag**(3/2) * (1 + 3 * (1 - restitution) / 2 / vel_i_before * vel_i_bar)
    VI_force = k_c * plus_slag ** (3 / 2)
    # VI_force = 0

    # print('vel_i_before = {}'.format(vel_i_before))
    # print('vel_i[(maxnode // 2) * 2, 0] = {}'.format(vel_i_bar))
    # print('plus_slag**(3/2) = {}'.format(plus_slag**(3/2)))
    # print('VI_force = {}'.format(VI_force))

    global_force[point_bar, 0] += VI_force

    return global_force, VI_force


def open_file_earthquake_data():
    file_name = r'D:\Beam\PEERNGARecords_Unscaled\RSN1106_KOBE_KJM000.AT2'
    with open(file_name, 'r') as cur_file:
        body = cur_file.readlines()

    # выделяем из файла значения количества точек и шага по времени
    print(body[:4])
    param = body[3].split(', ')
    points = int(param[0].split('=')[1])
    time_step = float((param[1].split('=')[1]).strip().split(' ')[0])

    # отбрасываем шапку тела файла
    body_points = list(map(lambda x: x.strip(), body[4:]))
    all_data = []
    for line in body_points:
        line_sep = list(map(float, line.split('  ')))
        all_data += line_sep

    print('Количетво точек совпадает с указанным в файле:', end=' ')
    print(len(all_data) == points)
    print('Количество значений: {}'.format(points))
    print('Time step: {}'.format(time_step))

    return time_step, all_data


def create_impulse_earthquake_data():  # создаем импульсное поле ускорений
    points = 75000
    impulse_period = 1e-3
    time_step = 1e-4
    ampl_impulse = 100
    impulse_point = int(impulse_period / time_step)
    x_impulse = np.linspace(0, impulse_period, impulse_point)
    omega_impulse = 2 * np.pi * (1 / (impulse_period * 2))
    y_impulse = -ampl_impulse * np.sin(omega_impulse * x_impulse)
    all_data = list(y_impulse) + [0] * (points - len(y_impulse))

    return time_step, all_data, impulse_period


# инерционная сила при землетрясении
def earthquake_force(global_force, earth_acceleration, dm):

    global_force[2::2, 0] = -earth_acceleration * dm

    return global_force


def create_modal_matrix(K, M):
    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigh(K, M)

    # Perform mass-orthonormalization of eigenvectors
    eigenvectors_normalized = eigenvectors / np.sqrt(np.diag(np.dot(eigenvectors.T, np.dot(M, eigenvectors))))

    # Print the eigenvalues
    print("Eigenvalues:")
    print(eigenvalues)

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
