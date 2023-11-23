from class_beam_elements import Class_beam_elements
from create_global_matrix import *
import numpy as np
from matplotlib import pyplot as plt
from chart_script import *
import pandas as pd
import os
from timeit import default_timer as timer


# np.set_printoptions(precision=1)
def main_body_fun(loc_bar=0.9):
    timer_start = timer()
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
    dm = dl * ro
    # ---------определяем параметры временного шага----------
    # первая частота балки = 8 Гц, период = 0.125с
    # dt = 2e-6  # шаг по времени
    # t_end = 200  # исследуемый интервал времени
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
    global_force = create_global_force(global_force, f_ampl=0)  # создаем начальный вектор сил

    eigenvalues, eigenvectors_normalized = create_modal_matrix(global_stiffness, global_mass)  # создаем модальную матрицу для перехода в модальные координаты для расчета возбуждаемый мод

    # ksi_list = np.array([0.03] * (2 * MaxNode))
    ksi_list = np.array([0, 0.015, 0.015, 0.015, 0.015, 0] + [0.015] * 36)
    # ksi_list = np.array([0.0] * (2 * MaxNode))

    list_diag_damping_modal = 2 * ksi_list * (eigenvalues ** 0.5)
    global_damping_modal = np.diag(list_diag_damping_modal)
    # global_damping = 0 * global_stiffness + 0 * global_mass  # глобальная МД (матрица демпфирования). Нулевое демпфирование
    global_damping = np.dot(global_mass, np.dot(eigenvectors_normalized, np.dot(global_damping_modal, np.dot(eigenvectors_normalized.T, global_mass))))

    # global_damping = 0 * global_stiffness + 0 * global_mass  # глобальная МД (матрица демпфирования). Нулевое демпфирование

    # проверочный для МЖ (матрица жесткости) статический расчет
    # с помощью него зададим начальные координаты
    start_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    # ----------------------------------------------------------

    # print(start_def)
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
    # ---------------------------

    time_disp = [dis_i[point_bar, 0]]  # запоминаем з-ть коодинаты середины балки
    time_disp_end = [dis_i[-2, 0]]  # запоминаем з-ть коодинаты конца балки
    time_force = [global_force[point_bar, 0]]  # запоминаем з-ть VI силы
    time_lst = [0]  # массив времени

    step_plot = 1000  # каждый 200ый шаг выводим графики
    number_mode_plot = 10  # количество мод, которое выводим на графиках
    full_en_lst = [0]  # массив полной энергии стержня, кин + потен
    earthquake_en_lst = [0]  # массив энергии, переданной балке через силы инерции
    en_func = []  # лист функционала энергии
    time_en_func = []  # лист времени для массива функционала энергии
    file_num = 1  # порядковый номер текстового файла, в который записываем очередную порцию дданных
    is_plot = False

    fig, axs = plt.subplots(3, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    # fig.suptitle('ГРАФИКИ')
    axs[0][0].set_title('Beam shape')
    axs[1][0].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10)
    axs[2][0].set_title('VI force')
    axs[0][0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r', linewidth=1)
    scale = start_def[-2][0]  # Масштаб графика формы балки
    # axs[0][0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей
    axs[1][0].plot(time_lst, time_disp, 'g', linewidth=1)
    axs[1][0].plot(time_lst, time_disp_end, 'k', linewidth=1)
    axs[2][0].plot(time_lst, time_force, 'k', linewidth=1)
    plt.pause(7)
    axs[0][0].clear()
    axs[1][0].clear()
    axs[2][0].clear()

    # ------- для вычисления силы VI ----------------------
    k_c = 100 * 2 * E * (10e-3 ** 0.5) / 3 / (1 - nu ** 2)  # константа в формуле силы VI
    print('k_c = {}'.format(k_c))
    vel_i_before = vel_i[point_bar, 0]
    dis_i_before = dis_i[point_bar, 0]
    # -------------------------------------------------------------------------------------
    global_force = create_global_force(global_force, f_ampl=0)  # обнуляем силу на конце балки


    # ------- EARTHQUAKE ----------------

    # earthquake_time_step, earthquake_all_data = open_file_earthquake_data()  # записываем шаг акселерограммы и данные землетрясения
    # normal_fr = np.sqrt(2688.5 / 244.98707749566427)  # нормализуем землетрясение в Кобе, сужаем акселлелограмму, что бы несущая частота совпала с первой частотой колебания балки

    earthquake_time_step, earthquake_all_data, impulse_period = create_impulse_earthquake_data()  # создаем импульсное поле ускорений
    normal_fr = 1

    earthquake_time_step = earthquake_time_step / normal_fr  # нормализация через изменение шага по времени акселерограммы
    earthquake_time_lst = np.linspace(0, (len(earthquake_all_data) - 1) * earthquake_time_step, len(earthquake_all_data))  # нормализированный массив времени для акселерограммы

    # вырезаем из акселлелограммы только интересующий нас интервал
    # -------------------
    # def cut_list(lst, value):
    #     id_cut = 0
    #     for i in range(len(lst)):
    #         if lst[i] > value:
    #             id_cut = i
    #             break
    #     return id_cut
    #
    # # time_start = 7.38
    # time_start = 0  # начало нужного участка
    # time_end = 13  # конец
    # i_start = cut_list(earthquake_time_lst, time_start)
    # i_end = cut_list(earthquake_time_lst, time_end)

    # earthquake_all_data = earthquake_all_data[i_start:i_end]  # оставляем нужный интервал
    earthquake_all_data = earthquake_all_data[:]  # оставляем нужный интервал

    # -------------------
    # инвентируем массив данных акселерограммы, что бы читать с конца и постепенно отрезать с помощью pop
    earthquake_data_inverse = earthquake_all_data[::-1]
    accumulated_earthquake_time = 0
    # ---------- END EARTHQUAKE -------------------------

    # начинаем цикл по времени
    t = 0
    t_end = 0.4

    dt_lst = [2e-8, 1e-7, 1e-6]  # лист временных шагов, которые будем динамически менять
    # dt_lst = [1e-6] * 3  # лист временных шагов без барьера
    # Начинаем с самого большого шага. Если этим большим шагом зашли вовнутрь барьера, то откываемся на шаг цикла назад и меняем временной шаг на следующий в листе.
    # Так делаем до тех пор, пока шаг не станет самым маленьким из списка. Потом считаем на этом шаге, но как только балка выйдет из барьера, каждый
    # следующий шаг делаем на один больше из списка.

    # Для каждого шага по времени заполняем массивы MCK и MCK_inv
    MCK_lst = [(global_mass + gamma * dt_cur * global_damping + beta * dt_cur ** 2 * global_stiffness) for dt_cur in dt_lst]
    MCK_inv_lst = [(np.linalg.inv(MCK_cur)) for MCK_cur in MCK_lst]

    time_step_id = len(MCK_inv_lst) - 1  # индекс вревенного шага в листе, который используем в данный момент
    A_earthquake = 0  # работа инерционных сил землетрясения

    check_if, check_if_2 = True, True
    check_while = True

    try:
        while (t < t_end) and check_while:
        # while True:
            dt = dt_lst[time_step_id]
            t += dt

            # для свободных колебаний
            # global_force = create_global_force(global_force, f_ampl=0)

            global_force[point_bar, 0] = 0  # обнуляем силу взаимодействия с барьером

            # ------- EARTHQUAKE ----------------
            # Накапливаем время, пока не перепрыгнем через временной шаг акселерограммы. Когда перепрыгнули, берем следующее значение данных
            # акселерограммы путем отрезания конца листа данных. Из накопленного времени вычитаем временной шаг акселерограммы.
            accumulated_earthquake_time += dt
            flag_earthquake_time = True
            last_value = 0
            if accumulated_earthquake_time >= earthquake_time_step:
                flag_earthquake_time = False
                accumulated_earthquake_time = (accumulated_earthquake_time - earthquake_time_step)
                last_value = earthquake_data_inverse.pop()
                # print(last_value)

            global_force = earthquake_force(global_force, earthquake_data_inverse[-1], dm)  # инерционная сила при землетрясении
            # -----------------------------------

            VI_force = 0
            if -dis_i[point_bar, 0] - delta >= 0:
                # print('Действует сила')

                # гасим в случае, когда фигачим без барьера
                global_force, VI_force = create_VI_force(global_force, point_bar, delta, dis_i[point_bar, 0], vel_i[point_bar, 0],
                                               vel_i_before, k_c, restitution=0.7)
                # pass
            else:
                # vel_i_before = vel_i[point_bar, 0]
                # dis_i_before = dis_i[point_bar, 0]
                pass

            # Алгоритм НьюМарка
            vel_i1_pred = vel_i + (1 - gamma) * dt * acc_i
            dis_i1_pred = dis_i + dt * vel_i + (0.5 - beta) * dt ** 2 * acc_i

            acc_i1 = np.matmul(MCK_inv_lst[time_step_id],
                               global_force - np.matmul(global_damping, vel_i1_pred) - np.matmul(global_stiffness,
                                                                                                 dis_i1_pred))

            vel_i1 = vel_i1_pred + gamma * dt * acc_i1
            dis_i1 = dis_i1_pred + beta * dt ** 2 * acc_i1

            # Если зашли в барьер, откатываемся на шаг назад и меняем временной шаг на один меньше по листу
            # STEP BACK
            # if jump in barrier make a time step back and decrease time step
            if (dt > dt_lst[0]) and (dis_i1[point_bar, 0] < 0):
                # step back
                t -= dt
                if flag_earthquake_time:
                    accumulated_earthquake_time -= dt
                else:
                    accumulated_earthquake_time += earthquake_time_step
                    accumulated_earthquake_time -= dt
                    earthquake_data_inverse.append(last_value)

                time_step_id -= 1
                # print(t, 'step --> <<')

                continue

            # если вышли из барьра, увеличиваем временной шаг на один больший по листу
            if (dt < dt_lst[-1]) and (dis_i1[point_bar, 0] > 0):
                time_step_id += 1
                # print(t, 'step --> >>')


            # считаем работу инерционных сил землетрясения
            if last_value != 0:
                cur_acc = last_value
            else:
                cur_acc = earthquake_data_inverse[-1]

            dis_diff = np.array([dis_i1[i * 2, 0] - dis_i[i * 2, 0] for i in range(MaxNode)])
            A_earthquake += (-cur_acc * dm) * np.sum(dis_diff)
            # -----------------------------

            # # ------ распределение энергии предыдущего шага по модам -----------
            # modal_dis_i_before = np.matmul(np.linalg.inv(eigenvectors_normalized), dis_i)
            # modal_dis_i_transp_before = [modal_dis for sublist in modal_dis_i_before for modal_dis in sublist]
            # modal_vel_i_before = np.matmul(np.linalg.inv(eigenvectors_normalized), vel_i)
            # modal_vel_i_transp_before = [modal_vel for sublist in modal_vel_i_before for modal_vel in sublist]
            # full_en_mode_before = 1 / 2 * np.array(modal_vel_i_transp_before) ** 2 + 1 / 2 * eigenvalues * np.array(modal_dis_i_transp_before) ** 2
            # # --------------------------------------------------------


            dis_i = dis_i1.copy()
            vel_i = vel_i1.copy()
            acc_i = acc_i1.copy()

            time_disp.append(dis_i1[point_bar, 0])
            time_disp_end.append(dis_i1[-2, 0])
            # time_force.append(global_force[point_bar, 0])
            time_force.append(VI_force)
            time_lst.append(t)

            if (len(time_lst) % step_plot == 0) or (round(t, 8) == 0.06568748):
                is_plot = True
            else:
                is_plot = False

            if is_plot:
                axs[0][0].set_title('Beam shape')
                axs[1][0].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10, pad=0)
                axs[2][0].set_title('VI force')
                # fig.suptitle('Time = ' + str('%.2f' % t)
                #              + ' s = ' + str('%.2f' % (t * 1e3)) + ' ms = ' + str('%.2f' % (t * 1e6)) + ' µs')
                fig.suptitle('Real Time = ' + str('%.2f' % t)
                             + 's = ' + str('%.2f' % (t * 1e3)) + 'ms' + '  (Algorithm Time = ' + str('%.1f' % ((timer() - timer_start) / 60)) + ' min)')
                axs[0][0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r',
                            linewidth=1)  # Положение балки
                axs[0][0].plot([L * (point_bar / 2) / (MaxNode - 1)], [dis_i1[point_bar, 0]], 'go', markersize=4)  # Жирная точка середина балки
                axs[0][0].plot([L], [dis_i1[-2, 0]], 'ko', markersize=4)  # Жирная точка конца балки
                axs[0][0].plot([L * (point_bar / 2) / (MaxNode - 1)], [-delta], 'b^', markersize=7)  # Местоположение барьера
                # scale = max(abs(min(time_disp_end)), abs(max(time_disp_end)), delta * 2)  # Масштаб графика формы балки
                scale = start_def[-2][0]  # Масштаб графика формы балки
                # axs[0][0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей

                axs[1][0].plot(time_lst, time_disp, color='g', linewidth=1)  # временная з-ть середины балки
                axs[1][0].plot(time_lst, time_disp_end, color='k', linewidth=1)  # временная з-ть конца балки
                axs[1][0].plot(time_lst, [-delta] * len(time_lst), 'r--', linewidth=1)  # барьер

                axs[2][0].plot(time_lst, time_force, 'k', linewidth=1)  # временная з-ть силы VI

            # ----------- выводим график полной энергии --------
            dis_i_transp = [disp for sublist in dis_i for disp in sublist]
            vel_i_transp = [vel for sublist in vel_i for vel in sublist]
            full_cur_en = 1 / 2 * np.dot(vel_i_transp, np.dot(global_mass, vel_i))[0] + 1 / 2 * np.dot(dis_i_transp, np.dot(global_stiffness, dis_i))[0]
            full_en_lst.append(full_cur_en)
            earthquake_en_lst.append(A_earthquake)
            if is_plot:
                axs[0][1].set_title('Energy')
                axs[0][1].plot(time_lst, full_en_lst, color='k', linewidth=1, label='Beam')
                axs[0][1].plot(time_lst, earthquake_en_lst, color='r', linestyle='--', linewidth=1, label='Earthquake')
                # axs[0][1].legend()

            # ------ ЕСЛИ ИМПУЛЬСНОЕ ВОЗМУЩЕНИЕ ----------
            # ------ Запоминаем начальную энергию сразу после конца возмущения -----
            if check_if and (t > impulse_period):
                check_if = False
                en_pulse_0 = full_cur_en
            if check_if_2 and (not check_if) and (en_pulse_0 / full_cur_en >= np.exp(1)):
                check_if_2 = False
                print('Relaxation time = ' + str(t))
                print('Algorithm Time = ' + str('%.1f' % ((timer() - timer_start) / 60)) + ' min')
                check_while = False

            # --------------------------------------------------------

            # ------заполняем массив амплитудами перемещений рассматриваемых мод------------
            modal_dis_i = np.matmul(np.linalg.inv(eigenvectors_normalized), dis_i)
            modal_dis_i_transp = [modal_dis for sublist in modal_dis_i for modal_dis in sublist]
            # --------------------------------------------------------
            # ------ заполняем массив квадратами амплитуд модальных скоростей ------------
            modal_vel_i = np.matmul(np.linalg.inv(eigenvectors_normalized), vel_i)
            modal_vel_i_transp = [modal_vel for sublist in modal_vel_i for modal_vel in sublist]
            # --------------------------------------------------------

            # ------ заполняем массив полной энергии мод ------------
            full_en_mode = 1 / 2 * np.array(modal_vel_i_transp) ** 2 + 1 / 2 * eigenvalues * np.array(modal_dis_i_transp) ** 2


            energy_values = [full_en_mode[i] for i in [1, 2, 3, 4, 6, 7, 8, 9]]
            if np.sum(energy_values) != 0:
            # if True:
                omega_first = np.array([(eigenvalues[i]) ** 0.5 for i in [1, 2, 3, 4, 6, 7, 8, 9]])
                energy_density = np.array([(en_cur / np.sum(energy_values)) for en_cur in energy_values])
                en_func_cur = -np.sum(energy_density * np.log(energy_density))  # энтропия
                # en_func_cur = np.sum(energy_density * omega_first)  # придуманный функционал
                en_func.append(en_func_cur)
            else:
                en_func.append(0)
            time_en_func.append(t)

            if is_plot:
                df = pd.DataFrame(full_en_mode, columns=['origin'], index=range(1, len(modal_vel_i_transp) + 1))
                axs[1][1].set_title('Energy distribution over modes\nMax energy = {} %'.format(round(max(df.origin[:number_mode_plot]) / sum(df.origin[:number_mode_plot]) * 100)), fontsize=10, pad=0)
                # axs[1][1].bar(df.index[:number_mode_plot], df.origin[:number_mode_plot])
                axs[1][1].bar(np.arange(1, number_mode_plot - 1), df.origin[:number_mode_plot].iloc[[1, 2, 3, 4, 6, 7, 8, 9]])
                # axs[1][1].axis([0, df.index[number_mode_plot], 0, 1e-3])
                # axs[1][1].axis([0, df.index[number_mode_plot] - 2, 0, 1e-3])
                # axs[1][1].set_xticks(np.arange(1, number_mode_plot + 1))
                axs[1][1].set_xticks(np.arange(1, number_mode_plot - 1))

                # вместо графика логарифма энергии будем строить график функционала распределения энергии
                # axs[2][1].set_title('Log distribution of energy over modes', fontsize=10)
                # # axs[2][1].bar(df.index[:number_mode_plot], df.origin[:number_mode_plot], log=True)
                # axs[2][1].bar(np.arange(1, number_mode_plot - 1), df.origin[:number_mode_plot].iloc[[1, 2, 3, 4, 6, 7, 8, 9]], log=True)
                # # axs[2][1].set_xticks(np.arange(1, number_mode_plot + 1))
                # axs[2][1].set_xticks(np.arange(1, number_mode_plot - 1))
                # ----------------------------------------------------------------------------------
                # строим график функционала распределения энергии по модам балки
                axs[2][1].set_title('Energy functional', fontsize=10)
                axs[2][1].plot(time_en_func, en_func, color='b', linewidth=1)  # исходное

                # if t > 1e-3:  # временно выводим скорость изменения энергии по конкретной моде
                #     axs[2][1].plot(time_en_func[1100:], full_en_mode_vel_lst[1100:], color='b', linewidth=1)
                # else:
                #     axs[2][1].plot(time_en_func, full_en_mode_vel_lst, color='b', linewidth=1)

                # axs[2][1].plot(time_en_func, full_en_mode_abs_lst[1:], color='b', linewidth=1)
                axs[2][1].grid()
                # ----------------------------------------------------------------------------------

                if len(time_lst) % (step_plot * 500) == 0:  # сохраняем график как картинку
                    file_name = 'time_{}.pdf'.format(str('%.2f' % t))
                    path_cur = path + file_name
                    plt.savefig(path_cur, bbox_inches='tight')

                if round(t, 8) == 0.06568748:
                    plt.savefig('./plots/small_damp.pdf', bbox_inches='tight')

                plt.pause(0.000001)
                # plt.pause(3)
                axs[0][0].clear()
                axs[1][0].clear()
                axs[2][0].clear()
                axs[0][1].clear()
                axs[1][1].clear()
                axs[2][1].clear()

            # # для экономии времени записи листов в файл и экономии места в этих листах, будем каждые сколько то шагов сбрасывать значения в новый файл и обнулять листы
            # if len(time_lst) % (step_plot * 500) == 0:
            #     # time_disp_end, time_disp(berrier), time_lst, time_force, full_en_lst, earthquake_en_lst, en_func, time_en_func
            #
            #     with open(path + 'time_disp_end_{}.txt'.format(file_num), 'w') as cur_file:
            #         cur_file.write(str(time_disp_end))
            #     with open(path + 'time_disp_{}.txt'.format(file_num), 'w') as cur_file:
            #         cur_file.write(str(time_disp))
            #     with open(path + 'time_lst_{}.txt'.format(file_num), 'w') as cur_file:
            #         cur_file.write(str(time_lst))
            #     with open(path + 'time_force_{}.txt'.format(file_num), 'w') as cur_file:
            #         cur_file.write(str(time_force))
            #     with open(path + 'full_en_lst_{}.txt'.format(file_num), 'w') as cur_file:
            #         cur_file.write(str(full_en_lst))
            #     with open(path + 'earthquake_en_lst_{}.txt'.format(file_num), 'w') as cur_file:
            #         cur_file.write(str(earthquake_en_lst))
            #     with open(path + 'en_func_{}.txt'.format(file_num), 'w') as cur_file:
            #         cur_file.write(str(en_func))
            #     with open(path + 'time_en_func_{}.txt'.format(file_num), 'w') as cur_file:
            #         cur_file.write(str(time_en_func))
            #
            #     file_num += 1
            #
            #     time_disp_end, time_disp, time_lst, time_force, full_en_lst, earthquake_en_lst, en_func, time_en_func = [], [], [], [], [], [], [], []
            #     # ------------------------------------------------------------------------------

        # # когда наступило время релаксации сохраняем данные в файлы
        # with open(path + 'time_disp_end.txt', 'w') as cur_file:
        #     cur_file.write(str(time_disp_end))
        # with open(path + 'time_disp.txt', 'w') as cur_file:
        #     cur_file.write(str(time_disp))
        # with open(path + 'time_lst.txt', 'w') as cur_file:
        #     cur_file.write(str(time_lst))
        # with open(path + 'time_force.txt', 'w') as cur_file:
        #     cur_file.write(str(time_force))
        # with open(path + 'full_en_lst.txt', 'w') as cur_file:
        #     cur_file.write(str(full_en_lst))
        # with open(path + 'earthquake_en_lst.txt', 'w') as cur_file:
        #     cur_file.write(str(earthquake_en_lst))
        # with open(path + 'en_func.txt', 'w') as cur_file:
        #     cur_file.write(str(en_func))

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

    loc_bar = 0.9
    path = './plots/location_{}/'.format(round(loc_bar, 1))
    os.mkdir(path)
    # with open(path + 'readme.txt', 'w') as f:
    #     f.write('Close to the barrier 1e-8\nFar from the berrier 2e-6\nForce 1\nkc = 1\ndelta = 0')
    main_body_fun(loc_bar=loc_bar)

    # loc_bar_list = np.arange(0.9, 0.4, -0.1)
    # for loc_bar in loc_bar_list:
    #     path = './plots/location_{}/'.format(round(loc_bar, 1))
    #     os.mkdir(path)
    #
    #     main_body_fun(loc_bar=loc_bar)
    #     plt.close()