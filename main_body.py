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
    nu = 0.3  # коэффициент Пуассона
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    MaxNode = 100 + 1  # количество узлов
    # ---------определяем параметры временного шага----------
    # первая частота балки = 8 Гц, период = 0.125с
    dt = 1e-3  # шаг по времени
    t_end = 200  # исследуемый интервал времени
    # ---------определяем параметры метода Ньюмарка----------
    gamma = 0.5
    beta = 0.5
    # -------------------------------------------------------

    elements = [Class_beam_elements(i, L, MaxNode, E, I, ro) for i in
                range(MaxNode - 1)]  # создаем массив балочных элементов

    # формируем матрицы для дифура
    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ
    print(global_stiffness)
    print('----------------------------------')
    global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ в МЖ
    print(global_stiffness)
    print('----------------------------------')
    global_mass = build_global_mass_matrix(elements, MaxNode)  # собирает глобальную ММ
    print(global_mass)
    # global_damping = build_global_damping_matrix(elements, MaxNode)  # собирает глобальную ММ
    # print(global_damping)
    global_damping = 0 * global_stiffness + 0 * global_mass
    print(global_damping)

    global_force = np.zeros((2 * MaxNode, 1))
    global_force = create_global_force(global_force, MaxNode, f_ampl=0)  # создаем вектор сил
    print(global_force)

    # проверочный для МЖ статический расчет
    # stat_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    # print(stat_def)

    # ------------Начало метода Ньюмарка----------------------------
    # MCK = global_mass + gamma * dt * global_damping + beta * dt**2 * global_stiffness

    dis_i = np.zeros((2 * MaxNode, 1))  # начальный вектор координат
    vel_i = np.zeros((2 * MaxNode, 1))  # начальный вектор скоростей
    # считаем начальный вектор ускорений
    acc_i = np.matmul(np.linalg.inv(global_mass),
                      (global_force - np.matmul(global_damping, vel_i) - np.matmul(global_stiffness, dis_i)))

    time_disp = [dis_i[(MaxNode // 2) * 2, 0]]
    time_disp_end = [dis_i[-2, 0]]
    time_force = [global_force[(MaxNode // 2) * 2, 0]]
    time_lst = [0]

    fig, axs = plt.subplots(3)
    plt.subplots_adjust(wspace=0.4, hspace=0.7)
    # fig.suptitle('ГРАФИКИ')
    axs[0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r', linewidth=1)
    # axs[0].axis([0, L * 1.1, -max(time_disp) * 1.1, max(time_disp) * 1.1])  # устанавливаем диапозон осей
    axs[1].plot(time_lst, time_disp, 'g', linewidth=1)
    axs[1].plot(time_lst, time_disp_end, 'g', linewidth=1)
    axs[2].plot(time_lst, time_force, 'g', linewidth=1)
    axs[0].set_title('Форма балки')
    axs[1].set_title('Временная з-ть узла балки')
    axs[2].set_title('VI force')
    plt.pause(2)
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()

    # delta = a/2 + 5e-4
    delta = 2e-5
    delta_original = delta

    k_c = 2 * E * 10e-3 ** 0.5 / 3 / (1 - nu ** 2)
    print('k_c = {}'.format(k_c))
    vel_i_before = vel_i[(MaxNode // 2) * 2, 0]
    dis_i_before = dis_i[(MaxNode // 2) * 2, 0]
    # -------------------------------------------------------------------------------------
    # начинаем цикл по времени
    t = 0
    # t_end = 0.07839
    try:
        while t < t_end:
            t += dt

            MCK = global_mass + gamma * dt * global_damping + beta * dt ** 2 * global_stiffness

            print('Time = ', str(t))
            if t < dt * 100:
                f_ampl = -6 * np.sin(2 * np.pi * 5 * t)
            else:
                f_ampl = 0
            global_force = create_global_force(global_force, MaxNode, f_ampl)
            global_force[(MaxNode // 2) * 2, 0] = 0

            if -dis_i[(MaxNode // 2) * 2, 0] - delta >= 0:
                delta = -dis_i_before * 0.999  # динамически двигаем барьер
                print('Действует сила')
                global_force = create_VI_force(global_force, MaxNode, delta, dis_i, vel_i, vel_i_before, k_c,
                                               restitution=0.7)
            else:
                delta = delta_original
                vel_i_before = vel_i[(MaxNode // 2) * 2, 0]
                dis_i_before = dis_i[(MaxNode // 2) * 2, 0]

            # print(vel_i[MaxNode+1, 0])
            vel_i1_pred = vel_i + (1 - gamma) * dt * acc_i
            dis_i1_pred = dis_i + dt * vel_i + (0.5 - beta) * dt ** 2 * acc_i

            acc_i1 = np.matmul(np.linalg.inv(MCK),
                               global_force - np.matmul(global_damping, vel_i1_pred) - np.matmul(global_stiffness,
                                                                                                 dis_i1_pred))

            vel_i1 = vel_i1_pred + gamma * dt * acc_i1
            dis_i1 = dis_i1_pred + beta * dt ** 2 * acc_i1

            dis_i = dis_i1.copy()
            vel_i = vel_i1.copy()
            acc_i = acc_i1.copy()

            time_disp.append(dis_i1[(MaxNode // 2) * 2, 0])
            time_disp_end.append(dis_i1[-2, 0])
            time_force.append(global_force[(MaxNode // 2) * 2, 0])
            time_lst.append(t)

            axs[0].set_title('Форма балки')
            axs[1].set_title('Временная з-ть узла балки. \n Черная - конец, зеленая - середина.', fontsize=10)
            axs[2].set_title('VI force')
            fig.suptitle('Время = ' + str('%.2f' % t)
                         + ' c = ' + str('%.2f' % (t * 1e3)) + ' мс = ' + str('%.2f' % (t * 1e6)) + ' мкс')
            axs[0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r',
                        linewidth=1)  # Положение балки
            axs[0].plot([L / 2], [dis_i1[(MaxNode // 2) * 2, 0]], 'go', markersize=4)  # Жирная точка середина балки
            axs[0].plot([L], [dis_i1[-2, 0]], 'ko', markersize=4)  # Жирная точка конца балки
            axs[0].plot([L / 2], [-delta], 'b^', markersize=7)  # Местоположение барьера
            scale = max(abs(min(time_disp_end)), abs(max(time_disp_end)), delta * 2)  # Масштаб графика формы балки
            axs[0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей

            axs[1].plot(time_lst, time_disp, 'g', linewidth=1)  # временная з-ть середины балки
            axs[1].plot(time_lst, time_disp_end, 'k', linewidth=1)  # временная з-ть конца балки
            axs[2].plot(time_lst, time_force, 'k', linewidth=1)  # временная з-ть силы VI
            plt.pause(0.0001)
            axs[0].clear()
            axs[1].clear()
            axs[2].clear()

            if 2 * dis_i[(MaxNode // 2) * 2, 0] <= -delta:
                dt = 1e-5
            else:
                dt = 1e-4

    except KeyboardInterrupt:
        return


main_body_fun()
