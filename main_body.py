from class_beam_elements import Class_beam_elements
from create_global_matrix import *
import numpy as np
from matplotlib import pyplot as plt
from chart_script import *
import pandas as pd
import os


# np.set_printoptions(precision=1)
def main_body_fun(loc_bar=0.9):
    # ---------определяем параметры материала образца----------
    a = 10e-3  # сторона квадратного сечения
    S = a * a  # площадь сечения балки
    I_inertia = a * a ** 3 / 12  # момент инерции сечения
    ro = 7850 * S  # погонная плотность
    E = 2e11  # модуль Юнга
    nu = 0.3  # коэффициент Пуассона
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    MaxNode = 20 + 1  # количество узлов
    dl = L / (MaxNode - 1)
    # ---------определяем параметры временного шага----------
    # первая частота балки = 8 Гц, период = 0.125с
    dt = 2e-6  # шаг по времени
    t_end = 200  # исследуемый интервал времени
    # ---------определяем параметры метода Ньюмарка----------
    gamma = 0.5
    beta = 0.5
    # -------------------------------------------------------

    elements = [Class_beam_elements(i, L, MaxNode, E, I_inertia, ro) for i in
                range(MaxNode - 1)]  # создаем массив балочных элементов

    # формируем матрицы для дифура
    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ (матрица жесткости)
    global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ в МЖ
    # print(global_stiffness)

    global_mass = build_global_mass_matrix(elements, MaxNode)  # собирает глобальную ММ (матрица масс)
    # print(global_mass)

    global_force = np.zeros((2 * MaxNode, 1))  # создаем размер глобального вектора сил
    global_force = create_global_force(global_force, f_ampl=1)  # создаем начальный вектор сил

    eigenvalues, eigenvectors_normalized = create_modal_matrix(global_stiffness, global_mass)  # создаем модальную матрицу для перехода в модальные координаты для расчета возбуждаемый мод

    # ksi_list = np.array([0.03] * (2 * MaxNode))
    # list_diag_damping_modal = 2 * ksi_list * (eigenvalues ** 0.5)
    # global_damping_modal = np.diag(list_diag_damping_modal)
    # # global_damping = 0 * global_stiffness + 0 * global_mass  # глобальная МД (матрица демпфирования). Нулевое демпфирование
    # global_damping = np.dot(global_mass, np.dot(eigenvectors_normalized, np.dot(global_damping_modal, np.dot(eigenvectors_normalized.T, global_mass))))
    global_damping = 0 * global_stiffness + 0 * global_mass  # глобальная МД (матрица демпфирования). Нулевое демпфирование

    # проверочный для МЖ (матрица жесткости) статический расчет
    # с помощью него зададим начальные координаты
    start_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    print(start_def)
    print(np.concatenate(start_def).ravel().tolist())

    # Начало метода Ньюмарка
    # dis_i = np.zeros((2 * MaxNode, 1))  # начальный вектор координат
    dis_i = start_def.copy()  # начальный вектор координат
    vel_i = np.zeros((2 * MaxNode, 1))  # начальный вектор скоростей
    # modal_vel_i = np.zeros((2 * MaxNode, 1))  # начальный вектор модальных скоростей

    # считаем начальный вектор ускорений
    acc_i = np.matmul(np.linalg.inv(global_mass),
                      (global_force - np.matmul(global_damping, vel_i) - np.matmul(global_stiffness, dis_i)))

    # ------------------------------ параметры барьера ----------------------------------
    # loc_bar = 0.9  # местоположение барьера вдоль оси балки (от 0 до 1)
    point_bar = round((MaxNode - 1) * loc_bar) * 2  # номер эелемента в глобальном векторе сил, на который действует сила VI
    delta = 0  # зазор
    # delta = start_def[point_bar, 0] / 10  # зазор
    start_def_loc_bar = start_def[point_bar, 0]
    delta_original = delta  # динамически немного далее двигаем барьер
    # ---------------------------

    time_disp = [dis_i[point_bar, 0]]  # запоминаем з-ть коодинаты середины балки
    time_disp_end = [dis_i[-2, 0]]  # запоминаем з-ть коодинаты конца балки
    time_force = [global_force[point_bar, 0]]  # запоминаем з-ть VI силы
    time_lst = [0]  # массив времени
    en_kin_lst = []  # массив кинетической энергии балки
    modal_vel_time = [0]  # массив суммы квадратов модальный скоростей
    disp_modes = [0] * (2 * MaxNode)  # массив модальных перемещений
    vel_modes = [0] * (2 * MaxNode)  # массив модальных скоростей
    full_en_mode = [0] * (2 * MaxNode)  # массив полной энергии мод
    step_plot = 1000  # каждый 200ый шаг выводим графики
    number_mode_plot = 10  # количество мод, которое выводим на графиках

    fig, axs = plt.subplots(3, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.7)
    # fig.suptitle('ГРАФИКИ')
    axs[0][0].set_title('Beam shape')
    axs[1][0].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10)
    axs[2][0].set_title('VI force')
    axs[0][0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r', linewidth=1)
    scale = start_def[-2][0]  # Масштаб графика формы балки
    axs[0][0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей
    axs[1][0].plot(time_lst, time_disp, 'g', linewidth=1)
    axs[1][0].plot(time_lst, time_disp_end, 'k', linewidth=1)
    axs[2][0].plot(time_lst, time_force, 'k', linewidth=1)
    plt.pause(1)
    axs[0][0].clear()
    axs[1][0].clear()
    axs[2][0].clear()
    # main_chart_first_step(L, MaxNode, dis_i, start_def, time_lst, time_disp, time_disp_end, time_force)

    # ------- для вычисления силы VI ----------------------
    k_c = 2 * E * (10e-3 ** 0.5) / 3 / (1 - nu ** 2)  # константа в формуле силы VI
    print('k_c = {}'.format(k_c))
    vel_i_before = vel_i[point_bar, 0]
    dis_i_before = dis_i[point_bar, 0]
    # -------------------------------------------------------------------------------------
    # начинаем цикл по времени
    t = 0
    t_end = 0.5
    try:
        while t < t_end:
            t += dt

            MCK = global_mass + gamma * dt * global_damping + beta * dt ** 2 * global_stiffness

            # print('Time = ', str(t))

            # if True:
            # if t < 2e-3:
            # # if t < dt * 100:
            #     f_ampl = -600 * np.sin(2 * np.pi * 5 * t)
            #     # f_ampl = 1
            # else:
            #     f_ampl = 0

            global_force = create_global_force(global_force, f_ampl=0)
            global_force[point_bar, 0] = 0

            if -dis_i[point_bar, 0] - delta >= 0:
                # delta = -dis_i_before * 0.999  # динамически двигаем барьер
                # print('Действует сила')
                global_force = create_VI_force(global_force, point_bar, delta, dis_i[point_bar, 0], vel_i[point_bar, 0],
                                               vel_i_before, k_c, restitution=0.7)
            else:
                # delta = delta_original
                vel_i_before = vel_i[point_bar, 0]
                dis_i_before = dis_i[point_bar, 0]

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

            time_disp.append(dis_i1[point_bar, 0])
            time_disp_end.append(dis_i1[-2, 0])
            time_force.append(global_force[point_bar, 0])
            time_lst.append(t)

            if len(time_lst) % step_plot == 0:
                axs[0][0].set_title('Beam shape')
                axs[1][0].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10)
                axs[2][0].set_title('VI force')
                # fig.suptitle('Time = ' + str('%.2f' % t)
                #              + ' s = ' + str('%.2f' % (t * 1e3)) + ' ms = ' + str('%.2f' % (t * 1e6)) + ' µs')
                fig.suptitle('Time=' + str('%.2f' % t)
                             + 's=' + str('%.2f' % (t * 1e3)) + 'ms')
                axs[0][0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r',
                            linewidth=1)  # Положение балки
                axs[0][0].plot([L * (point_bar / 2) / (MaxNode - 1)], [dis_i1[point_bar, 0]], 'go', markersize=4)  # Жирная точка середина балки
                axs[0][0].plot([L], [dis_i1[-2, 0]], 'ko', markersize=4)  # Жирная точка конца балки
                axs[0][0].plot([L * (point_bar / 2) / (MaxNode - 1)], [-delta], 'b^', markersize=7)  # Местоположение барьера
                # scale = max(abs(min(time_disp_end)), abs(max(time_disp_end)), delta * 2)  # Масштаб графика формы балки
                scale = start_def[-2][0]  # Масштаб графика формы балки
                axs[0][0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей

                axs[1][0].plot(time_lst, time_disp, color='g', linewidth=1)  # временная з-ть середины балки
                axs[1][0].plot(time_lst, time_disp_end, color='k', linewidth=1)  # временная з-ть конца балки
                axs[1][0].plot(time_lst, [-delta] * len(time_lst), 'r--', linewidth=1)  # барьер

                axs[2][0].plot(time_lst, time_force, 'k', linewidth=1)  # временная з-ть силы VI
                # plt.pause(0.000001)
                # axs[0][0].clear()
                # axs[1][0].clear()
                # axs[2][0].clear()

            # ----------- выводим график полной энергии --------
            dis_i_transp = [disp for sublist in dis_i for disp in sublist]
            vel_i_transp = [vel for sublist in vel_i for vel in sublist]
            cur_kin_en = 1 / 2 * np.dot(vel_i_transp, np.dot(global_mass, vel_i)) + 1 / 2 * np.dot(dis_i_transp, np.dot(global_stiffness, dis_i))
            en_kin_lst.append(cur_kin_en)
            if len(time_lst) % step_plot == 0:
                # plt.figure(2)

                # plt.plot(time_lst[1:], en_kin_lst, color='k', linewidth=1)
                axs[0][1].set_title('Total energy')
                axs[0][1].plot(time_lst[1:], en_kin_lst, color='k', linewidth=1)
                # axs[0][1].clear()
            # # --------------------------------------------------------

            # # ----------- выводим график модальный скоростей --------
            # modal_vel_i = np.matmul(np.linalg.inv(eigenvectors_normalized), vel_i)
            # modal_vel_i_transp = [modal_vel for sublist in modal_vel_i for modal_vel in sublist]
            # # modal_vel_time.append(sum(list(map(lambda x: x**2, modal_vel_i_transp))) / 2)
            # modal_vel_time.append(modal_vel_i_transp[9])
            # plt.figure(3)
            # plt.plot(time_lst, modal_vel_time, color='k', linewidth=1)
            # # --------------------------------------------------------

            # ------заполняем массив амплитудами перемещений рассматриваемых мод------------
            modal_dis_i = np.matmul(np.linalg.inv(eigenvectors_normalized), dis_i)
            modal_dis_i_transp = [modal_dis for sublist in modal_dis_i for modal_dis in sublist]
            # for jj in range(len(modal_dis_i_transp)):
            #     disp_modes[jj] = max(disp_modes[jj], modal_dis_i_transp[jj])
            # --------------------------------------------------------
            # ------ заполняем массив квадратами амплитуд модальных скоростей ------------
            modal_vel_i = np.matmul(np.linalg.inv(eigenvectors_normalized), vel_i)
            modal_vel_i_transp = [modal_vel for sublist in modal_vel_i for modal_vel in sublist]
            # for jj in range(len(modal_vel_i_transp)):
            #     vel_modes[jj] = max(vel_modes[jj], modal_vel_i_transp[jj] ** 2)
            # --------------------------------------------------------

            # ------ заполняем массив полной энергии мод ------------
            full_en_mode = 1 / 2 * np.array(modal_vel_i_transp) ** 2 + 1 / 2 * eigenvalues * np.array(modal_dis_i_transp) ** 2
            # plt.figure(3)
            if len(time_lst) % step_plot == 0:
                df = pd.DataFrame(full_en_mode, columns=['origin'], index=range(1, len(modal_vel_i_transp) + 1))
                axs[1][1].set_title('Energy distribution over modes\nMax energy = {} %'.format(round(max(df.origin[:number_mode_plot]) / sum(df.origin[:number_mode_plot]) * 100)), fontsize=10)
                # axs[1][1].bar(df.index[:number_mode_plot], df.origin[:number_mode_plot])
                axs[1][1].bar(np.arange(1, number_mode_plot - 1), df.origin[:number_mode_plot].iloc[[1, 2, 3, 4, 6, 7, 8, 9]])
                # axs[1][1].axis([0, df.index[number_mode_plot], 0, 1e-3])
                # axs[1][1].axis([0, df.index[number_mode_plot] - 2, 0, 1e-3])
                # axs[1][1].set_xticks(np.arange(1, number_mode_plot + 1))
                axs[1][1].set_xticks(np.arange(1, number_mode_plot - 1))

                axs[2][1].set_title('Log distribution of energy over modes', fontsize=10)
                # axs[2][1].bar(df.index[:number_mode_plot], df.origin[:number_mode_plot], log=True)
                axs[2][1].bar(np.arange(1, number_mode_plot - 1), df.origin[:number_mode_plot].iloc[[1, 2, 3, 4, 6, 7, 8, 9]], log=True)
                # axs[2][1].set_xticks(np.arange(1, number_mode_plot + 1))
                axs[2][1].set_xticks(np.arange(1, number_mode_plot - 1))

                if len(time_lst) % (step_plot * 200) == 0:  # сохраняем график как картинку
                    # file_name = 'loc_{}_time_{}.pdf'.format(loc_bar, str('%.2f' % t))
                    file_name = 'time_{}.pdf'.format(str('%.2f' % t))
                    # path = './plots/' + file_name
                    path = './plots/location_{}/'.format(round(loc_bar, 1)) + file_name
                    plt.savefig(path, bbox_inches='tight')

                plt.pause(0.000001)
                axs[1][1].clear()
                axs[2][1].clear()

                axs[0][0].clear()
                axs[1][0].clear()
                axs[2][0].clear()

                axs[0][1].clear()
                # print(df.origin[:20])
            # --------------------------------------------------------
            # # -------- строим график распределения квадратов модальных скоростей по модам -------
            # plt.figure(4)
            # df = pd.DataFrame(vel_modes, columns=['origin'], index=range(1, len(vel_modes) + 1))
            # df.origin.plot.bar(rot=0, log=True)
            # # --------------------------------------------------------

            # if 10 * dis_i[point_bar, 0] <= -delta:
            if dis_i[point_bar, 0] <= (start_def_loc_bar / 10):
                dt = 1e-8  # если рядом с барьером
                # print('little step')
            else:
                dt = 2e-6

            # # каждые сколько-то шагов записываем значения амплитуд колебаний на рассматриваемый частотах в файл
            # if len(time_lst) % 10 == 0:
            #     file_name = 'write_ampl_modes_{}_nodes.txt'.format(MaxNode)
            #     with open(r'./initial_disp/' + file_name, 'w') as cur_file:
            #         cur_file.write(str(disp_modes))
            #
            #     file_name = 'write_ampl_vel_modes_{}_nodes.txt'.format(MaxNode)
            #     with open(r'./initial_disp/' + file_name, 'w') as cur_file:
            #         cur_file.write(str(vel_modes))


    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    # # loc_bar_list = np.arange(0.1, 1, 0.1)
    # loc_bar_list = np.arange(0.7, 1, 0.1)
    # for loc_bar in loc_bar_list:
    #     path = './plots/location_{}/'.format(round(loc_bar, 1))
    #     os.mkdir(path)
    #
    #     with open(path + 'readme.txt', 'w') as f:
    #         f.write('Close to the barrier 2e-7\nFar from the berrier 2e-6\nForce 1')
    #
    #     main_body_fun(loc_bar=loc_bar)
    #     plt.close()

    loc_bar = 0.1
    path = './plots/location_{}/'.format(round(loc_bar, 1))
    os.mkdir(path)
    with open(path + 'readme.txt', 'w') as f:
        f.write('Close to the barrier 1e-8\nFar from the berrier 2e-6\nForce 1\nkc = 1\ndelta = 0')
    main_body_fun(loc_bar=loc_bar)
