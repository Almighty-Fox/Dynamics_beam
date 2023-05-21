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
    MaxNode = 30 + 1  # количество узлов
    dl = L
    # ---------определяем параметры временного шага----------
    # первая частота балки = 50 Гц, период = 0.02с
    dt = 1e-3
    t_end = 2
    # -------------------------------------------------------
    al0 = 1 / (dt ** 2)
    print(al0)
    al1 = 1 / (2 * dt)
    al2 = 2 * al0
    # -------------------------------------------------------

    # формируем матрицы для дифура
    global_stiffness = 2 * E * I / dl ** 3 \
                   * np.array(  # создаем матрицу жесткости для каждого элемента (Stiffness_matrix_beam_elements.jpg)
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 6, -3 * dl],
             [0, 0, -3 * dl, 2 * dl * dl]])

    global_mass = ro * dl / 420 \
                  * np.array(  # создаем матрицу масс для каждого элемента (Mass_matrix_beam_elements.jpg)
        [[156, 22 * dl, 54, -13 * dl],
         [22 * dl, 4 * dl * dl, 13 * dl, -3 * dl * dl],
         [54, 13 * dl, 156, -22 * dl],
         [-13 * dl, -3 * dl * dl, -22 * dl, 4 * dl * dl]])

    # global_mass = ro * dl / 420 \
    #               * np.array(  # создаем матрицу масс для каждого элемента (Mass_matrix_beam_elements.jpg)
    #     [[1, 0, 0, 0],
    #      [0, 1, 0, 0],
    #      [0, 0, 156, -22 * dl],
    #      [0, 0, -22 * dl, 4 * dl * dl]])

    global_force = np.zeros((4, 1))
    global_force[-2, 0] = 1
    # global_force[-1, 0] = 1

    # проверочный для МЖ статический расчет
    stat_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    print(stat_def)

    # формируем эффективную матрицу жесткости
    ef_mass = al0 * global_mass
    ef_mass_inv = np.linalg.inv(ef_mass)

    dis_i = np.zeros((4, 1))
    vel_i = np.zeros((4, 1))
    acc_i = np.zeros((4, 1))
    dis_i_1 = np.zeros((4, 1))
    vel_i_1 = np.zeros((4, 1))
    acc_i_1 = np.zeros((4, 1))

    # dis_i[-2, 0] = 1e-3
    # vel_i[-2, 0] = 2
    # acc_i[-2, 0] = 1

    # dis_i[-1, 0] = 1e-3
    # vel_i[-1, 0] = 2
    # acc_i[-1, 0] = 1
    # dis_i_1[-2, 0] = 1e-3 - dt * 2 + dt ** 2 / 2 * 1
    # dis_i_1[-1, 0] = 1e-3 - dt * 2 + dt ** 2 / 2 * 1

    fig, ax = plt.subplots()

    time_disp = []
    time_lst = []

    # начинаем цикл по времени
    for t in np.arange(dt, t_end, dt):
        if t > (dt*5):
            global_force[-2, 0] = 0
        #     global_force[-1, 0] = 0
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

        # lst_dis = [dis_i1[i * 2, 0] for i in range(MaxNode)]
        # ax.set_ylim([-3e-2, 3e-2])

        # ax.plot([0, 1], [dis_i1[0], dis_i1[-2]], 'r', linewidth=1)
        # plt.pause(0.02)
        # ax.clear()

        time_disp.append(dis_i1[-2, 0])
        time_lst.append(t)
        # print(time_disp)
        # print(np.arange(0, t, dt))
        # ax.set_ylim([-3e-2, 3e-2])
        ax.plot(time_lst, time_disp, 'r', linewidth=1)
        plt.pause(0.02)
        ax.clear()


main_body_fun()
