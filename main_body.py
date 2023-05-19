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
    MaxNode = 3 + 1  # количество узлов
    # ---------определяем параметры временного шага----------
    # первая частота балки = 50 Гц, период = 0.02с
    dt = 1e-4
    t_end = 2
    # ---------определяем параметры метода Ньюмарка----------
    al = 0.25
    bet = 0.5
    # -------------------------------------------------------
    al0 = 1 / (al * dt ** 2)
    al1 = bet / (al * dt)
    al2 = 1 / (al * dt)
    al3 = 1 / 2 / dt - 1
    al4 = bet / al - 1
    al5 = dt / 2 * (bet / al - 2)
    al6 = dt * (1 - bet)
    al7 = bet * dt
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

    # проверочный для МЖ статический расчет
    # stat_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    # print(stat_def)

    # формируем эффективную матрицу жесткости
    ef_stiffness = create_ef_stiffness(global_mass, global_stiffness, al0)

    # print(ef_stiffness)
    # print(np.linalg.inv(ef_stiffness))
    ef_stiffness_inv = np.linalg.inv(ef_stiffness)
    print('ef_stiffness_inv')
    print(ef_stiffness_inv)

    dis_i = np.zeros((2 * MaxNode, 1))
    vel_i = np.zeros((2 * MaxNode, 1))
    acc_i = np.zeros((2 * MaxNode, 1))
    # acc_i[-2, 0] = 1e-3

    fig, ax = plt.subplots()

    # начинаем цикл по времени
    for t in np.arange(dt, t_end, dt):
        # if t > (dt*5):
        #     global_force = create_global_force(MaxNode, f_ampl=0)
        print('Time = ', str(t))

        R_ef_2 = np.matmul(global_mass, al0 * dis_i + al2 * vel_i + al3 * acc_i)
        R_ef = global_force + R_ef_2
        print('R_ef')
        print(R_ef)
        # dis_i1 = np.matmul(np.linalg.inv(ef_stiffness), R_ef)
        dis_i1 = np.matmul(ef_stiffness_inv, R_ef)
        print('dis_i1')
        print(dis_i1)
        # acc_i1 = 1 / al * (1/dt**2 * (dis_i1 - dis_i - vel_i*dt) - (1/2 - al)*acc_i)
        # vel_i1 = vel_i + ((1 - bet) * acc_i + bet * acc_i1) * dt
        acc_i1 = al0 * (dis_i1 - dis_i) - al2 * vel_i - al3 * acc_i
        vel_i1 = vel_i + al6 * acc_i + al7 * acc_i1

        # print(np.reshape(np.array(dis_i1), (1, -1)))

        dis_i = dis_i1.copy()
        vel_i = vel_i1.copy()
        acc_i = acc_i1.copy()

        # fig.suptitle('Время = ' + str(t) + ' c = ')
        lst_dis = [dis_i1[i * 2, 0] for i in range(MaxNode)]
        # ax.set_ylim([-3e-2, 3e-2])
        ax.plot(np.linspace(0, L, num=MaxNode), lst_dis, 'r', linewidth=1)
        plt.pause(2)
        ax.clear()


main_body_fun()
