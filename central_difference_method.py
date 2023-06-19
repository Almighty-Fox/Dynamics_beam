from class_beam_elements import Class_beam_elements
from class_nodes import Class_nodes
from create_global_matrix import *
import numpy as np
from matplotlib import pyplot as plt

# np.set_printoptions(precision=1)
def main_body_fun():
    # ---------определяем параметры материала образца----------
    a = 10e-3  # сторона квадратного сечения
    S = a * a  # площадь сечения балки
    I = a * (a ** 3) / 12  # момент инерции сечения
    ro = 7850 * S  # погонная плотность
    E = 2e11  # модуль Юнга
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    MaxNode = 4 + 1  # количество узлов
    # ---------определяем параметры временного шага----------
    # первая частота балки = 50 Гц, период = 0.02с
    dt = 1e-4
    t_end = 2
    # -------------------------------------------------------
    al0 = 1 / (dt ** 2)
    print(al0)
    al1 = 1 / (2 * dt)
    al2 = 2 * al0
    # -------------------------------------------------------

    elements = [Class_beam_elements(i, L, MaxNode, E, I, ro) for i in
                range(MaxNode - 1)]  # создаем массив балочных элементов
    # nodes = [Class_nodes(i) for i in range(MaxNode)]  # создаем массив узлов

    # формируем матрицы для дифура
    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ
    print(global_stiffness)
    print('----------------------------------')
    global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ в МЖ
    print(global_stiffness)
    print('----------------------------------')
    global_mass = build_global_mass_matrix(elements, MaxNode)  # собирает глобальную ММ
    print(global_mass)

    global_force = np.zeros((2 * MaxNode, 1))
    global_force = create_global_force(global_force, MaxNode, f_ampl=0)  # создаем вектор сил

    # проверочный для МЖ статический расчет
    # stat_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    # print(stat_def)

    # формируем эффективную матрицу жесткости
    ef_mass = al0 * global_mass
    ef_mass_inv = np.linalg.inv(ef_mass)

    dis_i = np.zeros((2 * MaxNode, 1))
    vel_i = np.zeros((2 * MaxNode, 1))
    acc_i = np.zeros((2 * MaxNode, 1))
    dis_i_1 = np.zeros((2 * MaxNode, 1))
    vel_i_1 = np.zeros((2 * MaxNode, 1))
    acc_i_1 = np.zeros((2 * MaxNode, 1))
    # acc_i[-2, 0] = 1e-3

    fig, ax = plt.subplots()

    time_disp = []
    time_lst = []

    # начинаем цикл по времени
    for t in np.arange(dt, t_end, dt):
        if t > (dt*5):
            global_force = create_global_force(global_force, MaxNode, f_ampl=1)
        print('Time = ', str(t))

        R_ef = global_force - np.matmul((global_stiffness - al2 * global_mass), dis_i) - al0 * np.matmul(global_mass, dis_i_1)

        # dis_i1 = np.matmul(np.linalg.inv(ef_stiffness), R_ef)
        dis_i1 = np.matmul(ef_mass_inv, R_ef)

        vel_i1 = 1/2/dt * (dis_i1 - dis_i_1)
        acc_i1 = 1/dt**2 * (dis_i_1 - 2*dis_i + dis_i1)


        # print(np.reshape(np.array(dis_i1), (1, -1)))

        dis_i_1 = dis_i.copy()
        vel_i_1 = vel_i.copy()
        acc_i_1 = acc_i.copy()
        dis_i = dis_i1.copy()
        vel_i = vel_i1.copy()
        acc_i = acc_i1.copy()

        # fig.suptitle('Время = ' + str(t) + ' c = ')

        lst_dis = [dis_i1[i * 2, 0] for i in range(MaxNode)]
        # ax.set_ylim([-3e-2, 3e-2])
        ax.plot(np.linspace(0, L, num=MaxNode), lst_dis, 'r', linewidth=1)
        plt.pause(0.02)
        ax.clear()

        # time_disp.append(dis_i1[-2, 0])
        # time_lst.append(t)
        # # print(time_disp)
        # # print(np.arange(0, t, dt))
        # # ax.set_ylim([-3e-2, 3e-2])
        # ax.plot(time_lst, time_disp, 'r', linewidth=1)
        # plt.pause(0.02)
        # ax.clear()


main_body_fun()
