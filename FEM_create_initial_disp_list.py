from class_beam_elements import Class_beam_elements
from create_global_matrix import *
import numpy as np


def main_body_fun():
    # ---------определяем параметры материала образца----------
    a = 10e-3  # сторона квадратного сечения
    S = a * a  # площадь сечения балки
    I_inertia = a * a ** 3 / 12  # момент инерции сечения
    ro = 7850 * S  # погонная плотность
    E = 2e11  # модуль Юнга
    nu = 0.3  # коэффициент Пуассона
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    MaxNode = 32 + 1  # количество узлов
    dl = L / (MaxNode - 1)
    # -------------------------------------------------------

    elements = [Class_beam_elements(i, L, MaxNode, E, I_inertia, ro) for i in
                range(MaxNode - 1)]  # создаем массив балочных элементов

    # формируем матрицы для дифура
    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ (матрица жесткости)
    global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ в МЖ

    global_force = np.zeros((2 * MaxNode, 1))  # создаем размер глобального вектора сил
    global_force = create_global_force(global_force, f_ampl=1)  # создаем начальный вектор сил

    # проверочный для МЖ (матрица жесткости) статический расчет
    # с помощью него зададим начальные координаты
    start_def = np.matmul(np.linalg.inv(global_stiffness), global_force)

    start_def_ravel = np.concatenate(start_def).ravel().tolist()  # вектор столбец переводим в строку
    start_def_postup = start_def_ravel[::2]  # берем только поступательные координаты

    file_name = 'initial_disp_{}_nodes.txt'.format(MaxNode)
    with open(r'./initial_disp/' + file_name, 'w') as cur_file:
        for id, item in enumerate(start_def_postup):
            # write each item on a new line
            cur_file.write("%s\n" % item)
            if id % 10 == 0:
                print('Done {}%'.format(round(id / MaxNode * 100)))
    print('Done 100%')


main_body_fun()