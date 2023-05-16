from class_beam_elements import Class_beam_elements
from class_nodes import Class_nodes
from create_global_matrix import *
import numpy as np
from matplotlib import pyplot as plt


def main_body_fun():
    # ---------определяем параметры материала образца----------
    a = 10e-3  # сторона квадратного сечения
    S = a * a  # площадь сечения балки
    I = a*a**3 / 12  # момент инерции сечения
    ro = 7850 * S  # погонная плотность
    E = 2e11  # модуль Юнга
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    MaxNode = 100 + 1  # количество узлов
    # ---------определяем параметры временного шага----------
    # первая частота балки = 50 Гц, период = 0.02с
    dt = 2e-5
    t_end = 2e-2
    # ---------определяем параметры метода Ньюмарка----------
    aa = dt/2
    bb = dt/2
    cc = dt
    # -------------------------------------------------------

    elements = [Class_beam_elements(i, L, MaxNode, E, I, ro) for i in range(MaxNode - 1)]  # создаем массив балочных элементов
    nodes = [Class_nodes(i) for i in range(MaxNode)]  # создаем массив узлов

    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ
    global_mass = build_global_mass_matrix(elements, MaxNode)  # собирает глобальную ММ

    global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ и МЖ
    # print(global_stiffness)

    global_force = create_global_force(MaxNode)
    # print(global_force)

    stat_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    # print(stat_def)

    ef_stiffness = create_ef_stiffness(dt, global_mass, global_stiffness)

    dis_i_1 = np.zeros((2*MaxNode, 1))
    dis_i = np.zeros((2 * MaxNode, 1))
    vel_i_1 = np.zeros((2 * MaxNode, 1))
    vel_i = np.zeros((2 * MaxNode, 1))
    acc_i_1 = np.zeros((2 * MaxNode, 1))
    acc_i = np.zeros((2 * MaxNode, 1))

    fig, ax = plt.subplots()

    # начинаем цикл по времени
    for t in np.arange(dt, t_end, dt):
        if t > (dt*5):
            global_force = create_global_force(MaxNode, f_ampl=0)
        print('Time = ', str(t))

        R_ef_2 = np.matmul(global_mass, (dis_i/2/dt**2 + dis_i_1/2/dt + (bb+cc)/(aa+bb) * (acc_i_1 - acc_i)))
        R_ef = global_force + R_ef_2

        dis_i1 = np.matmul(np.linalg.inv(ef_stiffness), R_ef)
        acc_i1 = (dis_i1 - dis_i) /2/dt**2 - vel_i_1/2/dt - (bb+cc)/(aa+bb)*(acc_i_1 - acc_i)
        vel_i1 = vel_i_1 + 2*dt*acc_i1 + 2*dt*(bb+cc)/(aa+bb)*(acc_i_1 - acc_i)

        # print(np.reshape(np.array(dis_i1), (1, -1)))

        dis_i_1 = dis_i
        dis_i = dis_i1
        vel_i_1 = vel_i
        vel_i = vel_i1
        acc_i_1 = acc_i
        acc_i = acc_i1

        # fig.suptitle('Время = ' + str(t) + ' c = ')
        lst_dis = [dis_i1[i*2, 0] for i in range(MaxNode)]
        ax.set_ylim([-3e-3, 3e-3])
        ax.plot(np.linspace(0, L, num=MaxNode), lst_dis, 'r', linewidth=1)
        plt.pause(0.1)
        ax.clear()


main_body_fun()