from class_string_elements import Class_string_elements
from create_global_matrix_string import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from timeit import default_timer as timer


# np.set_printoptions(precision=1)
def main_body_fun(loc_bar=0.9):
    timer_start = timer()
    # # ---------определяем параметры материала образца----------
    # a = 10e-3  # сторона квадратного сечения
    # S = a * a  # площадь сечения балки
    # I_inertia = a * a ** 3 / 12  # момент инерции сечения
    # ro = 7850 * S  # погонная плотность
    # E = 2e11  # модуль Юнга
    # nu = 0.3  # коэффициент Пуассона

    # ---------определяем параметры материала образца----------
    a = 1  # сторона квадратного сечения
    S = a * a  # площадь сечения балки
    ro = 1 * S  # погонная плотность
    E = 1  # модуль Юнга
    nu = 0.3  # коэффициент Пуассона

    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    # MaxNode = 20 + 1  # количество узлов
    # MaxNode = 190 + 1  # количество узлов
    # MaxNode = 290 + 1  # количество узлов
    # dl = L / (MaxNode - 1)
    # dm = dl * ro
    # ---------определяем параметры метода Ньюмарка----------
    gamma = 0.5
    beta = 0.5
    # -------------------------------------------------------
    full_en_lst = [0]  # массив полной энергии стержня, кин + потен

    def generate_beam_elements(L_full, q, x, loc_bar):
        # если барьер в правой части балки, то отзеркаливаем,
        # что бы строить сетку проще, потом перевернем лист
        if loc_bar != 0.5:
            loc_bar_min = min(loc_bar, 1 - loc_bar)
            L = round(2 * loc_bar_min, 1)
        else:
            L = L_full

        # Расчет длин элементов с обеих сторон от центра
        lengths = [x]
        total_length = x
        i = 0

        # Добавляем элементы вправо до половины балки
        while total_length < L / 2:
            next_length = lengths[i] * q
            if total_length + next_length > L / 2:
                break
            lengths.append(next_length)
            total_length += next_length
            i += 1

        # Добавляем элементы влево (зеркально)
        #     lengths = lengths[::-1] + lengths[1:]
        lengths = lengths[::-1] + lengths[:]
        total_length = sum(lengths)

        # Корректировка крайних элементов, если общая длина не равна L
        if total_length < L:
            extra = (L - total_length) / 2
            lengths[0] += extra
            lengths[-1] += extra

        # подправляем размеры крайних элементов
        num_left = 0
        while lengths[0] > lengths[1] * 1.0001:
            num_left += 1
            lengths[0] /= 2
            lengths.insert(0, lengths[0])
        if num_left > 0:
            lengths.pop()
            lengths += [lengths[0]] * (num_left * 2)

        # Словарь длин элементов
        el_barrier = len(lengths) // 2
        if loc_bar != 0.5:
            dist_lost = round(L_full - L, 1)
            size_el_lost = 0.005  # 1/200
            lengths += [size_el_lost] * round(dist_lost / size_el_lost)

        # если барьер в правой части, то переворачиваем лист длин элементов
        if loc_bar > 0.5:
            lengths = lengths[::-1]
            el_barrier = len(lengths) - el_barrier

        element_lengths = {i: length for i, length in enumerate(lengths)}



        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 5))
        start = 0
        for length in lengths:
            ax.add_patch(plt.Rectangle((start, 0), length, 1, facecolor="#D3D3D3", edgecolor="black"))
            start += length
        ax.set_xlim(0, L_full)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title(f"Amount of elements: {len(element_lengths)}\n Sum lenght: {sum(element_lengths.values())}\n El min: {el_barrier}")
        # plt.show()

        return element_lengths, el_barrier

    # Пример использования
    L, q, x = 1, 1.04, 0.001
    loc_bar = 0.5
    dict_elements, smallest_index = generate_beam_elements(L, q, x, loc_bar)
    print("Элементы и их длины:", dict_elements)
    print("Номер наименьшего элемента:", smallest_index)
    print("Amount of elements:", len(dict_elements))
    print("Sum lenght:", sum(dict_elements.values()))
    print("----------------------------------------------------------")
    print("Close the figure to continue running the programme.")
    print("----------------------------------------------------------")
    plt.show()

    MaxNode = len(dict_elements) + 1  # количество узлов
    point_bar = smallest_index  # номер эелемента в глобальном векторе сил, на который действует сила VI

    # ------------------------------ параметры барьера ----------------------------------
    # loc_bar = 0.9  # местоположение барьера вдоль оси балки (от 0 до 1)
    # point_bar = round((MaxNode - 1) * loc_bar)  # номер эелемента в глобальном векторе сил, на который действует сила VI
    # point_bar = round((MaxNode - 1) / 2)  # номер эелемента в глобальном векторе сил, на который действует сила VI
    # ---------------------------

    elements = [Class_string_elements(i, L, MaxNode, E, S, ro, point_bar, dict_elements) for i in
                range(MaxNode - 1)]  # создаем массив балочных элементов

    print(sum([elements[i].dl for i in range(MaxNode - 1)]))

    nodes_coor = []
    cur_coor = 0
    for i in range(MaxNode - 1):
        nodes_coor.append(cur_coor)
        cur_coor += elements[i].dl
        print(cur_coor)
    nodes_coor.append(L)
    nodes_coor = np.array(nodes_coor)

    print(nodes_coor)

    # формируем матрицы для дифура
    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ (матрица жесткости)
    global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ в МЖ
    print(global_stiffness)

    global_mass = build_global_mass_matrix(elements, MaxNode)  # собирает глобальную ММ (матрица масс)
    # print(global_mass)

    global_force = np.zeros((MaxNode, 1))  # создаем размер глобального вектора сил
    global_force = create_global_force(global_force, f_ampl=0)  # создаем начальный вектор сил

    eigenvalues, eigenvectors_normalized = create_modal_matrix(global_stiffness, global_mass)  # создаем модальную матрицу для перехода в модальные координаты для расчета возбуждаемый мод
    # print((ro / E / I_inertia * np.array(eigenvalues[:10])) ** (1/4))  # вывод корней аналитического уравнения

    # global_damping = 0 * global_stiffness + 0 * global_mass  # глобальная МД (матрица демпфирования). Нулевое демпфирование
    ksi_list = np.array([2.0] * (MaxNode))
    list_diag_damping_modal = 2 * ksi_list * (eigenvalues ** 0.5)
    global_damping_modal = np.diag(list_diag_damping_modal)
    global_damping = np.dot(global_mass, np.dot(eigenvectors_normalized, np.dot(global_damping_modal,
                                                                                np.dot(eigenvectors_normalized.T,
                                                                                       global_mass))))

    # проверочный для МЖ (матрица жесткости) статический расчет
    # с помощью него зададим начальные координаты
    start_def = np.matmul(np.linalg.inv(global_stiffness), global_force)
    # ----------------------------------------------------------

    # print(start_def)
    print(np.concatenate(start_def).ravel().tolist())

    # Начало метода Ньюмарка
    # dis_i = np.zeros((MaxNode, 1))  # начальный вектор координат
    dis_i = -np.sin(np.pi * nodes_coor / L) - np.sin(3 * np.pi * nodes_coor / L)
    dis_i = dis_i.reshape(-1, 1)
    # dis_i = start_def.copy()  # начальный вектор координат

    # vel_i = np.zeros((MaxNode, 1))  # начальный вектор скоростей
    # vel_i = -1e0 * np.sin(np.pi * np.linspace(0, 1, MaxNode))
    # vel_i = vel_i.reshape(-1, 1)

    # vel_i = -np.sin(np.pi * nodes_coor / L)
    vel_i = np.zeros_like(nodes_coor)
    vel_i = vel_i.reshape(-1, 1)

    # считаем начальный вектор ускорений
    acc_i = np.matmul(np.linalg.inv(global_mass),
                      (global_force - np.matmul(global_damping, vel_i) - np.matmul(global_stiffness, dis_i)))

    # plt.plot(np.linspace(0, L, MaxNode), np.matmul(global_damping, vel_i))
    # plt.show()



    time_disp = [dis_i[point_bar, 0]]  # запоминаем з-ть коодинаты середины балки
    time_disp_end = [dis_i[-1, 0]]  # запоминаем з-ть коодинаты конца балки
    time_force = [global_force[point_bar, 0]]  # запоминаем з-ть VI силы
    time_lst = [0]  # массив времени

    step_plot = 400  # каждый 200ый шаг выводим графики
    number_mode_plot = 10  # количество мод, которое выводим на графиках
    en_func = []  # лист функционала энергии
    en_func_2 = []  # лист второго функционала энергии
    time_en_func = []  # лист времени для массива функционала энергии


    fig, axs = plt.subplots(3, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    # fig.suptitle('ГРАФИКИ')
    axs[0][0].set_title('Beam shape')
    axs[1][0].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10)
    axs[2][0].set_title('VI force')
    axs[0][0].plot(nodes_coor, [dis_i[i, 0] for i in range(MaxNode)], 'r', linewidth=1)
    scale = start_def[-1][0]  # Масштаб графика формы балки
    # axs[0][0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей
    axs[1][0].plot(time_lst, time_disp, 'g', linewidth=1)
    axs[1][0].plot(time_lst, time_disp_end, 'k', linewidth=1)
    axs[2][0].plot(time_lst, time_force, 'k', linewidth=1)

    axs[0][1].set_title('Initial velocity')
    axs[0][1].grid()
    axs[0][1].plot(nodes_coor, [vel_i[i, 0] for i in range(MaxNode)], 'r', linewidth=1)

    plt.pause(2)
    axs[0][0].clear()
    axs[1][0].clear()
    axs[2][0].clear()
    axs[0][1].clear()


    # ------- для вычисления силы VI ----------------------
    R_barrier = 10e-3
    k_c = 1e5 * 2 * E * (R_barrier ** 0.5) / 3 / (1 - nu ** 2)  # константа в формуле силы VI
    print('k_c = {}'.format(k_c))
    # -------------------------------------------------------------------------------------
    global_force = create_global_force(global_force, f_ampl=0)  # обнуляем силу на конце балки


    # начинаем цикл по времени
    t = 0
    # t_end = 0.15
    t_end = 0.7

    # dt_lst = [2e-8, 1e-7, 1e-6]  # лист временных шагов, которые будем динамически менять
    dt_lst = [2e-6] * 3  # лист временных шагов без барьера
    # Начинаем с самого большого шага. Если этим большим шагом зашли вовнутрь барьера, то откываемся на шаг цикла назад и меняем временной шаг на следующий в листе.
    # Так делаем до тех пор, пока шаг не станет самым маленьким из списка. Потом считаем на этом шаге, но как только балка выйдет из барьера, каждый
    # следующий шаг делаем на один больше из списка.

    # Для каждого шага по времени заполняем массивы MCK и MCK_inv
    MCK_lst = [(global_mass + gamma * dt_cur * global_damping + beta * dt_cur ** 2 * global_stiffness) for dt_cur in dt_lst]

    MCK_inv_lst = [(np.linalg.inv(MCK_cur)) for MCK_cur in MCK_lst]
    print(MCK_inv_lst)

    time_step_id = len(MCK_inv_lst) - 1  # индекс вревенного шага в листе, который используем в данный момент

    try:
        while (t < t_end):
            dt = dt_lst[time_step_id]
            t += dt

            # if len(time_lst) < 100:
            #     print(len(time_lst))
            #     print('load')
            #     # для свободных колебаний
            #     global_force = create_global_force(global_force, f_ampl=1)

            global_force[point_bar, 0] = 0  # обнуляем силу взаимодействия с барьером

            VI_force = 0
            if dis_i[point_bar, 0] <= 0:
                # print('Действует сила')

                # гасим в случае, когда фигачим без барьера
                global_force, VI_force = create_VI_force(point_bar, global_force, dis_i[point_bar, 0], k_c)
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

            # print(MCK_inv_lst[time_step_id])
            # print(np.matmul(global_stiffness, dis_i1_pred))
            # print(np.matmul(MCK_inv_lst[time_step_id], np.matmul(global_stiffness, dis_i1_pred)))

            vel_i1 = vel_i1_pred + gamma * dt * acc_i1
            dis_i1 = dis_i1_pred + beta * dt ** 2 * acc_i1

            dis_i = dis_i1.copy()
            vel_i = vel_i1.copy()
            acc_i = acc_i1.copy()



            time_disp.append(dis_i1[point_bar, 0])
            time_disp_end.append(dis_i1[-1, 0])
            time_force.append(global_force[point_bar, 0])
            # time_force.append(VI_force)
            time_lst.append(t)
            time_en_func.append(t)  # переместили сюда, что бы строить графики напряжений, которые шли до этой команды


            if (len(time_lst) % step_plot == 0):  # or (round(t, 8) == 0.14201218):
                is_plot = True
            else:
                is_plot = False

            if is_plot:
                axs[0][0].set_title('Beam shape')
                axs[1][0].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=7, pad=0)
                axs[2][0].set_title('VI force, max ' + str('%.2f' % max(time_force)))
                axs[0][0].grid()
                axs[1][0].grid()
                axs[2][0].grid()
                # fig.suptitle('Time = ' + str('%.2f' % t)
                #              + ' s = ' + str('%.2f' % (t * 1e3)) + ' ms = ' + str('%.2f' % (t * 1e6)) + ' µs')
                fig.suptitle('Real Time = ' + str('%.2f' % t)
                             + 's = ' + str('%.2f' % (t * 1e3)) + 'ms' + '  (Algorithm Time = ' + str('%.1f' % ((timer() - timer_start) / 60)) + ' min)')
                axs[0][0].plot(nodes_coor, [dis_i[i, 0] for i in range(MaxNode)], 'r',
                            linewidth=1)  # Положение балки
                # axs[0][0].plot([L * (point_bar) / (MaxNode - 1)], [dis_i1[point_bar, 0]], 'go', markersize=4)  # Жирная точка середина балки
                axs[0][0].plot([L], [dis_i1[-1, 0]], 'ko', markersize=4)  # Жирная точка конца балки
                # axs[0][0].plot([L * point_bar / (MaxNode - 1)], [0], 'b^', markersize=7)  # Местоположение барьера
                axs[0][0].plot([L / 2], [0], 'b^', markersize=7)  # Местоположение барьера
                # scale = max(abs(min(time_disp_end)), abs(max(time_disp_end)), delta * 2)  # Масштаб графика формы балки
                scale = start_def[-1][0]  # Масштаб графика формы балки
                # axs[0][0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей

                axs[1][0].plot(time_lst, time_disp, color='g', linewidth=1)  # временная з-ть середины балки
                axs[1][0].plot(time_lst, time_disp_end, color='k', linewidth=1)  # временная з-ть конца балки

                axs[2][0].plot(time_lst, time_force, 'k', linewidth=1)  # временная з-ть силы VI

            # ----------------- ENERGY -------------------
            dis_i_transp = [disp for sublist in dis_i for disp in sublist]
            vel_i_transp = [vel for sublist in vel_i for vel in sublist]
            full_cur_en = 1 / 2 * np.dot(vel_i_transp, np.dot(global_mass, vel_i))[0] + 1 / 2 * \
                          np.dot(dis_i_transp, np.dot(global_stiffness, dis_i))[0]
            full_en_lst.append(full_cur_en)
            if is_plot:
                axs[1][1].set_title('Energy')
                axs[1][1].grid(True)
                axs[1][1].plot(time_lst, full_en_lst, color='k', linewidth=1, label='Beam')
                # axs[0][1].legend()
            # --------------------------------------------


                plt.pause(0.01)
                # plt.pause(10)
                axs[0][0].clear()
                axs[1][0].clear()
                axs[2][0].clear()



            # ------заполняем массив амплитудами перемещений рассматриваемых мод------------
            modal_dis_i = np.matmul(np.linalg.inv(eigenvectors_normalized), dis_i)
            modal_dis_i_transp = [modal_dis for sublist in modal_dis_i for modal_dis in sublist]
            # --------------------------------------------------------
            # ------ заполняем массив квадратами амплитуд модальных скоростей ------------
            modal_vel_i = np.matmul(np.linalg.inv(eigenvectors_normalized), vel_i)
            modal_vel_i_transp = [modal_vel for sublist in modal_vel_i for modal_vel in sublist]
            # --------------------------------------------------------

        # writing results
        # with open(r'./plots/string_break_contact/nodes_lst_k_1e4_bar_07__2.txt', 'w') as cur_file:
        #     cur_file.write(str(nodes_coor))
        # with open(r'./plots/string_break_contact/dis_lst_k_1e4_bar_07__2.txt', 'w') as cur_file:
        #     cur_file.write(str([dis_i[i, 0] for i in range(MaxNode)]))
        #     with open(r'./plots/string_break_contact/vel_lst_k_1e4_bar_07__2.txt', 'w') as cur_file:
        #         cur_file.write(str([vel_i[i, 0] for i in range(MaxNode)]))



                # ------------------------------------------------------------------------------
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    loc_bar = 0.5
    main_body_fun(loc_bar=loc_bar)
