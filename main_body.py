from class_beam_elements import Class_beam_elements
from class_nodes import Class_nodes
from create_global_matrix import build_global_stiffness_matrix
from create_global_matrix import build_global_mass_matrix
from create_global_matrix import global_stiffness_matrix_with_GU
from create_global_matrix import create_global_force
import numpy as np


def main_body_fun():
    # ---------определяем параметры материала образца----------
    a = 10e-3  # сторона квадратного сечения
    S = a * a  # площадь сечения балки
    I = a*a**3 / 12  # момент инерции сечения
    ro = 7850 * S  # погонная плотность
    E = 2e11  # модуль Юнга
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    MaxNode = 3 + 1  # количество узлов

    elements = [Class_beam_elements(i, L, MaxNode, E, I, ro) for i in range(MaxNode - 1)]  # создаем массив балочных элементов
    nodes = [Class_nodes(i) for i in range(MaxNode - 1)]  # создаем массив узлов

    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ
    global_mass = build_global_mass_matrix(elements, MaxNode)  # собирает глобальную ММ

    global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ и МЖ
    # print(global_stiffness)

    global_force = create_global_force(MaxNode)
    # print(global_force)

    stat_def = np.matmul(np.linalg.inv(global_stiffness), global_force)

    print(stat_def)



main_body_fun()