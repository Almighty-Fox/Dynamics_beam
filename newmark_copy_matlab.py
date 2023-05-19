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
    I = a * a ** 3 / 12  # момент инерции сечения
    ro = 7850 * S  # погонная плотность
    E = 2e11  # модуль Юнга
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    MaxNode = 100 + 1  # количество узлов
    # ---------определяем параметры временного шага----------
    # первая частота балки = 50 Гц, период = 0.02с
    dt = 2e-3
    t_end = 2
    nt = int(t_end / dt)
    # ---------определяем параметры метода Ньюмарка----------
    gamma = 1 / 2
    beta = 1 / 4
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt ** 2)
    a3 = 1 / (beta * dt)
    a4 = gamma / beta
    a5 = 1 / (2 * beta)
    a6 = (gamma / (2 * beta) - 1) * dt
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

    global_force = create_global_force(MaxNode, f_ampl=1)
    print(global_force)

    # проверочный для МЖ статический расчет
    # stat_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    # print(stat_def)

    dis_i = np.zeros((2 * MaxNode, nt))
    vel_i = np.zeros((2 * MaxNode, nt))
    # acc_i = np.zeros((2 * MaxNode, nt))
    acc_i = np.matmul(global_mass, global_force)

    Kcap = global_stiffness + a2 * global_mass

    a = a3 * global_mass
    b = a5 * global_mass

    fig, ax = plt.subplots()

    # начинаем цикл по времени
    for t in np.arange(dt, t_end, dt):
        print('Time = ', str(t))

        R_ef = global_force + np.matmul(a, vel_i) + np.matmul(b, acc_i)
        delx = np.matmul(Kcap, R_ef)
        delxdot = a1 * delx - a4 * vel_i - a6 * acc_i
        delx2dot = a2 * delx - a3 * vel_i - a5 * acc_i
        # print('R_ef')
        # print(R_ef)

        dis_i1 = dis_i + delx
        vel_i1 = vel_i + delxdot
        acc_i1 = acc_i + delx2dot



        # print(np.reshape(np.array(dis_i1), (1, -1)))

        dis_i = dis_i1.copy()
        vel_i = vel_i1.copy()
        acc_i = acc_i1.copy()

        # fig.suptitle('Время = ' + str(t) + ' c = ')
        lst_dis = [dis_i1[i * 2, 0] for i in range(MaxNode)]
        # ax.set_ylim([-3e-2, 3e-2])
        ax.plot(np.linspace(0, L, num=MaxNode), lst_dis, 'r', linewidth=1)
        plt.pause(0.7)
        ax.clear()


main_body_fun()
