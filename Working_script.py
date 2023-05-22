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
    # первая частота балки = 8 Гц, период = 0.125с
    dt = 2e-3  # шаг по времени
    t_end = 20  # исследуемый интервал времени
    nt = int(t_end / dt)
    # ---------определяем параметры метода Ньюмарка----------
    beta1 = 0.5
    beta2 = 0.5
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

    global_force = create_global_force(MaxNode, f_ampl=0)  # создаем вектор сил
    print(global_force)

    # проверочный для МЖ статический расчет
    # stat_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    # print(stat_def)

    # ------------Начало метода Ньюмарка----------------------------
    MK = global_mass + 0.5 * beta2 * dt**2 * global_stiffness

    dis_i = np.zeros((2 * MaxNode, 1))  # начальный вектор координат
    vel_i = np.zeros((2 * MaxNode, 1))  # начальный вектор скоростей
    # считаем начальный вектор ускорений
    acc_i = np.matmul(np.linalg.inv(global_mass), (-np.matmul(global_stiffness, dis_i) + global_force))

    time_disp = [dis_i[-2, 0]]
    time_lst = [0]

    fig, axs = plt.subplots(2)
    plt.subplots_adjust(wspace=0.6, hspace=0.4)
    # fig.suptitle('ГРАФИКИ')
    axs[0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r', linewidth=1)
    axs[1].plot(time_lst, time_disp, 'g', linewidth=1)
    axs[0].set_title('Форма балки')
    axs[1].set_title('Временная з-ть узла балки')
    plt.pause(2)
    axs[0].clear()
    axs[1].clear()

    # начинаем цикл по времени
    for t in np.arange(dt, t_end, dt):
        print('Time = ', str(t))
        if (t > dt * 5) & (t < dt * 100):
            global_force = create_global_force(MaxNode, f_ampl=1)
        else:
            global_force = create_global_force(MaxNode, f_ampl=0)

        # global_force = create_global_force(MaxNode, f_ampl=np.sin(2*np.pi*15 * t))
        # global_force = create_global_force(MaxNode, f_ampl=0)

        print(vel_i[MaxNode+1, 0])

        acc_i1 = np.matmul(np.linalg.inv(MK), global_force - np.matmul(global_stiffness, dis_i + dt*vel_i + 0.5*(1-beta2)*dt**2*acc_i))

        vel_i1 = vel_i + dt*(1-beta1)*acc_i + dt*beta1*acc_i1
        dis_i1 = dis_i + dt*vel_i + (1-beta2)*0.5*dt**2*acc_i + 0.5*beta2*dt**2*acc_i1

        dis_i = dis_i1.copy()
        vel_i = vel_i1.copy()
        acc_i = acc_i1.copy()

        time_disp.append(dis_i1[-2, 0])
        time_lst.append(t)

        axs[0].set_title('Форма балки')
        axs[1].set_title('Временная з-ть узла балки')
        fig.suptitle('Время = ' + str('%.2f' % t)
                     + ' c = ' + str('%.2f' % (t * 1e3)) + ' мс = ' + str('%.2f' % (t * 1e6)) + ' мкс')
        axs[0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r', linewidth=1)
        axs[0].axis([0, L, -max(time_disp)*1.1, max(time_disp)*1.1])  # устанавливаем диапозон осей
        axs[1].plot(time_lst, time_disp, 'g', linewidth=1)
        plt.pause(0.01)
        axs[0].clear()
        axs[1].clear()




main_body_fun()
