import numpy as np
from sympy import *
import pylab as plt
import time
import pandas as pd


# динамика балки без барьера
# def beam_no_VI_vibrations(y_vals, v_vals, gamma_, just_no_VI=False):
def beam_no_VI_vibrations(alpha_lst_init, beta_lst_init, gamma_, just_no_VI=False):
    global t_global, time_is_up, t_global_lst, energy_data_global, tau

    t_local = 0
    if_we_are_ready_for_new_VI = False


    in_barrier = False

    t_step_id = 0
    dt_next = t_step_lst[t_step_id]

    # Инициализация массивов решения
    u_init = np.zeros(Nx)
    for n0, beta_i in enumerate(beta_lst_init, start=1):
        u_init += beta_i * np.sin(np.pi * n0 * x_vals)
    # Граничные условия
    u_init[0] = 0
    u_init[-1] = 0

    v_init = np.zeros(Nx)
    for m0, alpha_i in enumerate(alpha_lst_init, start=1):
        v_init += (-(np.pi * m0)**2 * gamma_ / 2 * beta_lst_init[m0-1] + omega_m_damping(m0) * alpha_i) * np.sin(np.pi * m0 * x_vals)
    # Граничные условия
    v_init[0] = 0
    v_init[-1] = 0

    energy_data = []
    flag = True

    print('NO barrier')
    is_first_no_VI = True

    first_step = True


    # while ((u[point_barrier] >= 0) or (first_step)) and (not time_is_up):  # ???????????????????????
    # while not True:

    pass_for_next_while = False
    u_next = u_init.copy()
    while (u_next[point_barrier] >= 0) or just_no_VI or not if_we_are_ready_for_new_VI or pass_for_next_while:
        pass_for_next_while = False

        # if (t_global >= t_end) and not time_is_up:
        if (len(energy_data_global) > 0) and (energy_data_global[-1] <= energy_data_global[0] / np.exp(1)) and not time_is_up:
            time_is_up = True
            tau = t_global
            print('----------------------------------')
            print(f'tau = {t_global}')
            print(f'Number VI = {number_VI}')
            print('----------------------------------')
            graph(u_next, v_next, energy_over_modes_lst[:num_modes_plot_energy], case_plot=False, case_time_delay=False, case_time_delay_2=True)
            with open(r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\26_01_2025_string_multyVI\time_energy_test.txt', 'w') as cur_file:
                cur_file.write(str(t_global_lst))
            with open(r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\26_01_2025_string_multyVI\energy_test.txt', 'w') as cur_file:
                cur_file.write(str(energy_data_global))


        first_step = False
        t_global += dt_next
        t_global_lst.append(t_global)
        t_local += dt_next




        # Основной вычислительный цикл
        u_next = np.zeros(Nx)
        for m0, alpha_i in enumerate(alpha_lst_init, start=1):
            u_next += np.exp(-(np.pi * m0) ** 2 * gamma_ / 2 * t_local) * (
                        alpha_i * np.sin(omega_m_damping(m0) * t_local) + beta_lst_init[m0 - 1] * np.cos(
                    omega_m_damping(m0) * t_local)) * np.sin(np.pi * m0 * x_vals)

        # print(f'2 = {dt}')

        # Граничные условия
        u_next[0] = 0
        u_next[-1] = 0

        if not just_no_VI and if_we_are_ready_for_new_VI:
            # если зашли в барьер, откатываем шаг назад и уменьшаем шаг на меньший из листа шагов
            if (t_step_id > 0) and (u_next[point_barrier] < 0):
                pass_for_next_while = True
                in_barrier = True
                print('In barrier')
                t_global -= dt_next
                t_global_lst.pop()
                t_local -= dt_next

                t_step_id -= 1
                print(f't_step_id = {t_step_id}')
                dt_next = t_step_lst[t_step_id]

                continue

        # print(f'PREV = {dt_prev}')
        # print(f'NEXT = {dt_next}')
        v_next = np.zeros(Nx)
        v_next_t_lst = []
        u_next_t_lst = []
        for m0, alpha_i in enumerate(alpha_lst_init, start=1):
            v_next_t = (np.exp(-(np.pi * m0) ** 2 * gamma_ / 2 * t_local) * (-(np.pi * m0) ** 2 * gamma_ / 2 * (
                    alpha_i * np.sin(omega_m_damping(m0) * t_local) + beta_lst_init[m0 - 1] * np.cos(
                omega_m_damping(m0) * t_local)) + omega_m_damping(m0) * (alpha_i * np.cos(
                omega_m_damping(m0) * t_local) - beta_lst_init[m0 - 1] * np.sin(
                omega_m_damping(m0) * t_local))))
            v_next_t_lst.append(v_next_t)
            v_next += v_next_t * np.sin(np.pi * m0 * x_vals)


            u_next_t = np.exp(-(np.pi * m0) ** 2 * gamma_ / 2 * t_local) * (
                    alpha_i * np.sin(omega_m_damping(m0) * t_local) + beta_lst_init[m0 - 1] * np.cos(
                omega_m_damping(m0) * t_local))
            u_next_t_lst.append(u_next_t)


        # cur_energy = total_energy(u_next, v_cur)
        kinetic_energy = 0.25 * sum([v_next_t_lst[ii]**2 for ii in range(len(v_next_t_lst))])
        potential_energy = 0.25 * sum([(np.pi * (ii+1) * u_next_t_lst[ii]) ** 2 for ii in range(len(u_next_t_lst))])
        cur_energy = kinetic_energy + potential_energy
        # print(cur_energy)
        energy_data_global.append(cur_energy)
        # # print(cur_energy)
        # energy_over_modes_lst = energy_over_modes(u_next, v_cur, case_damping=True)
        energy_over_modes_lst = 0.25 * (np.array([v_next_t_lst[ii] ** 2 for ii in range(len(v_next_t_lst))]) + np.array(
            [(np.pi * (ii + 1) * u_next_t_lst[ii]) ** 2 for ii in range(len(u_next_t_lst))]))
        graph(u_next, v_next, energy_over_modes_lst[:num_modes_plot_energy], case_plot=True)
        # print(cur_energy)


        # ------------------------------------------------------

        # если выходим из барьера увеличиваем шаги
        # print((u_next[point_barrier]))
        if (not in_barrier) and (t_step_id < len(t_step_lst)-1):
            t_step_id += 1
            print('Out of barrier')
            print(f't_step_id = {t_step_id}')

        dt_next = t_step_lst[t_step_id]

        # print(f'1 = {u[point_barrier]}')

        # если выруливаем из случайного минуса после ВИ, то мы открыты к новому удару
        if u_next[point_barrier] > 1e-4:
            if_we_are_ready_for_new_VI = True


    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ С БАРЬЕРОМ
    beta_lst_end = np.zeros(num_modes)
    for m0, alpha_i in enumerate(alpha_lst_init, start=1):
        beta_lst_end[m0-1] = np.exp(-(np.pi * m0) ** 2 * gamma_ / 2 * t_local) * (
                alpha_i * np.sin(omega_m_damping(m0) * t_local) + beta_lst_init[m0 - 1] * np.cos(
            omega_m_damping(m0) * t_local))

    alpha_lst_end = np.zeros(num_modes)
    for m0, alpha_i in enumerate(alpha_lst_init, start=1):
        alpha_lst_end[m0-1] = (np.exp(-(np.pi * m0) ** 2 * gamma_ / 2 * t_local) * (-(np.pi * m0) ** 2 * gamma_ / 2 * (
                alpha_i * np.sin(omega_m_damping(m0) * t_local) + beta_lst_init[m0 - 1] * np.cos(
            omega_m_damping(m0) * t_local)) + omega_m_damping(m0) * (alpha_i * np.cos(
            omega_m_damping(m0) * t_local) - beta_lst_init[m0 - 1] * np.sin(
            omega_m_damping(m0) * t_local))))


    print('Before VI')
    graph(u_next, v_next, energy_over_modes_lst[:num_modes_plot_energy], case_plot=False, case_time_delay=False)

    return alpha_lst_end, beta_lst_end


def modal_coefficients(x, y, num_modes):
    coefficients = []
    for k in range(1, num_modes + 1):
        phi_k = np.sin(k * np.pi * x)
        a_k = 2 * np.trapz(y * phi_k, x=x)
        coefficients.append(a_k)

    return np.array(coefficients)

def disp_vel_to_alpha_beta_lst(y_vals, v_vals, num_modes):
    # Функция для вычисления коэффициентов модального разложения
    modal_coords = modal_coefficients(x_vals, y_vals, num_modes)
    modal_velocities = modal_coefficients(x_vals, v_vals, num_modes)

    # plt.figure()
    # plt.plot(x_vals, sum([np.array(modal_coords[i] * np.sin((i+1) * np.pi * x_vals)) for i in range(len(modal_coords))]))
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(x_vals,
    #          sum([np.array(modal_velocities[i] * np.sin((i + 1) * np.pi * x_vals)) for i in range(len(modal_coords))]))
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(x_vals, y_vals)
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(x_vals, v_vals)
    # plt.grid()
    #
    # plt.show()


    return modal_coords, modal_velocities


def F_tr_t_alpha(t_lst, m0, alpha_m0, a):
    return -alpha_m0 * m0 * np.cos(pi * m0 * t_lst) * np.sin(pi * m0 * a)


def F_tr_t_beta(t_lst, n0, beta_n0, a):
    return beta_n0 * n0 * np.sin(pi * n0 * t_lst) * np.sin(pi * n0 * a)


def VI_force(alpha_coef_lst, beta_coef_lst, t_lst):
    #     alpha_coef_lst = np.array([1, 1])
    #     beta_coef_lst = np.array([0, 1])
    #     a = 0.5
    #     t_lst = np.linspace(0, 0.51, 10000)
    VI_force_sum = 0
    for m0, alpha_m0 in enumerate(alpha_coef_lst, start=1):
        if alpha_m0 != 0:
            VI_force_sum += F_tr_t_alpha(t_lst, m0, alpha_m0, a_bar)
    for n0, beta_n0 in enumerate(beta_coef_lst, start=1):
        if beta_n0 != 0:
            VI_force_sum += F_tr_t_beta(t_lst, n0, beta_n0, a_bar)

    VI_force_sum *= 2 * pi

    index_detachment = np.argmax(VI_force_sum < 0)
    t = t_lst[index_detachment]
    # print(f'Time of detachment = {t}')
    # print(f'Force max = {VI_force_sum[0]}')

    return t


def VI_force_accuracy(alpha_coef_lst, beta_coef_lst):
    t_lst_big = np.linspace(0.0001, 0.51, 100)
    t_about = VI_force(alpha_coef_lst, beta_coef_lst, t_lst_big)
    print(f't_about = {t_about}')

    t_lst_right = np.linspace(0.0, t_about * 1.2, 10000)
    t_f_zero = VI_force(alpha_coef_lst, beta_coef_lst, t_lst_right)
    print(f't_f_zero = {t_f_zero}')

    return t_f_zero


# аналитическая сила ВИ для произвольных НУ
# -----------------------------------------

# Динамика балки с барьером
def beam_with_VI_vibrations(alpha_coef_lst, beta_coef_lst):
    # plt.figure()
    # plt.plot(x_vals, sum([np.array(beta_coef_lst[i] * np.sin((i+1) * np.pi * x_vals)) for i in range(len(beta_coef_lst))]))
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(x_vals,
    #          sum([np.array(alpha_coef_lst[i] * np.pi * (i + 1) * np.sin((i + 1) * np.pi * x_vals)) for i in range(len(alpha_coef_lst))]))
    # plt.grid()
    #
    #
    #
    # plt.show()

    detachment_time = VI_force_accuracy(alpha_coef_lst, beta_coef_lst)
    print(f'detachment_time = {detachment_time} !')

    print('with barrier')

    is_first_no_VI = True
    global t_global, t_global_lst, energy_data_global, number_VI, time_start_VI, time_end_VI, energy_during_VI
    number_VI += 1
    time_start_VI.append(t_global)
    time_end_VI.append(t_global + detachment_time)

    # это для рисовски вспомогательных линий на графике границ времени ВИ, для положения линии по оси игрик
    if len(energy_data_global) > 0:
        energy_during_VI.append(energy_data_global[-1])
    else:
        y1_init = np.zeros_like(x_vals)
        v1_init = np.zeros_like(x_vals)

        for n0, beta_n0 in enumerate(beta_coef_lst, start=1):
            if beta_n0 != 0:
                y1_init += beta_n0 * np.sin(pi * n0 * x_vals)

        for m0, alpha_m0 in enumerate(alpha_coef_lst, start=1):
            if alpha_m0 != 0:
                v1_init += alpha_m0 * np.sin(pi * m0 * x_vals) * np.pi * m0

        energy_during_VI.append(total_energy(y1_init, v1_init))


    first_step = True
    # while ((Q_t < 0) and (not time_is_up)): # or first_step:  # ???????????????????????
    # while True:
        # first_step = False



    # --------------------------------------------------------------


    # plt.figure()
    # plt.plot(x_vals, y1_init)
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(x_vals, v1_init)
    # plt.grid()
    #
    # plt.show()

    t = detachment_time

    t_lst = np.linspace(0, detachment_time, 4)
    t_lst_step = t_lst[1]
    for t in t_lst[1:]:

        t_global += t_lst_step
        t_global_lst.append(t_global)


        # Compute initial dis and vel
        # Compute y1, v1 (without VI influence)
        y1 = np.zeros_like(x_vals)
        v1 = np.zeros_like(x_vals)
        y1_init = np.zeros_like(x_vals)
        v1_init = np.zeros_like(x_vals)
        for n0, beta_n0 in enumerate(beta_coef_lst, start=1):
            if beta_n0 != 0:
                y1 += beta_n0 * np.sin(pi * n0 * x_vals) * np.cos(pi * n0 * t)
                y1_init += beta_n0 * np.sin(pi * n0 * x_vals)

                v1 += -beta_n0 * np.sin(pi * n0 * x_vals) * np.pi * n0 * np.sin(pi * n0 * t)

        for m0, alpha_m0 in enumerate(alpha_coef_lst, start=1):
            if alpha_m0 != 0:
                y1 += alpha_m0 * np.sin(pi * m0 * x_vals) * np.sin(pi * m0 * t)

                v1 += alpha_m0 * np.sin(pi * m0 * x_vals) * np.pi * m0 * np.cos(pi * m0 * t)
                v1_init += alpha_m0 * np.sin(pi * m0 * x_vals) * np.pi * m0




        # Compute y2, v2
        y2 = np.zeros_like(x_vals)
        v2 = np.zeros_like(x_vals)

        for m0, alpha_m0 in enumerate(alpha_coef_lst, start=1):
            if alpha_m0 != 0:
                sin_pi_m0_a = np.sin(pi * m0 * a_bar)
                sin_pi_m0_x = np.sin(pi * m0 * x_vals)
                sin_pi_m0_t = np.sin(pi * m0 * t)
                cos_pi_m0_t = np.cos(pi * m0 * t)

                y_term1_alpha = 0
                v_term1_alpha = 0

                y_term1_alpha = - (pi * t) / (2 * m0) * sin_pi_m0_t * sin_pi_m0_a * sin_pi_m0_x
                v_term1_alpha = sin_pi_m0_a * sin_pi_m0_x * (sin_pi_m0_t / m0 + pi * t * cos_pi_m0_t)

                # Sum over k, excluding k = m0
                sum_k_y_alpha = np.zeros_like(x_vals)
                sum_k_v_alpha = np.zeros_like(x_vals)

                for k in range(1, N_max + 1):
                    sin_pi_k_a = np.sin(pi * k * a_bar)
                    sin_pi_k_x = np.sin(pi * k * x_vals)
                    sin_pi_k_t = np.sin(pi * k * t)
                    cos_pi_k_t = np.cos(pi * k * t)

                    if k != m0:
                        numerator = sin_pi_k_a * sin_pi_k_x * (cos_pi_k_t - cos_pi_m0_t)
                        denominator = k ** 2 - m0 ** 2
                        sum_k_y_alpha += numerator / denominator

                        sum_k_v_alpha += sin_pi_k_a * sin_pi_k_x * (
                                1 / (m0 + k) * (sin_pi_m0_t + sin_pi_k_t) + 1 / (m0 - k) * (
                                sin_pi_m0_t - sin_pi_k_t))

                y2 += (4 / pi) * alpha_m0 * m0 * sin_pi_m0_a * (y_term1_alpha + sum_k_y_alpha)
                v2 += -2 * alpha_m0 * m0 * sin_pi_m0_a * (v_term1_alpha + sum_k_v_alpha)

        for n0, beta_n0 in enumerate(beta_coef_lst, start=1):
            if beta_n0 != 0:
                sin_pi_n0_a = np.sin(pi * n0 * a_bar)
                sin_pi_n0_x = np.sin(pi * n0 * x_vals)
                sin_pi_n0_t = np.sin(pi * n0 * t)
                cos_pi_n0_t = np.cos(pi * n0 * t)

                y_term1_beta = 0
                v_term1_beta = 0

                y_term1_beta = -(pi * n0 * t * cos_pi_n0_t - sin_pi_n0_t) / (2 * n0**2) * sin_pi_n0_a * sin_pi_n0_x
                v_term1_beta = sin_pi_n0_a * sin_pi_n0_x * pi * t * sin_pi_n0_t / (-2 * n0)

                # Sum over k, excluding k = m0
                sum_k_y_beta = np.zeros_like(x_vals)
                sum_k_v_beta = np.zeros_like(x_vals)

                for k in range(1, N_max + 1):
                    sin_pi_k_a = np.sin(pi * k * a_bar)
                    sin_pi_k_x = np.sin(pi * k * x_vals)
                    sin_pi_k_t = np.sin(pi * k * t)
                    cos_pi_k_t = np.cos(pi * k * t)

                    if k != n0:
                        sum_k_y_beta += 1 / k * sin_pi_k_a * sin_pi_k_x / (n0 ** 2 - k ** 2) * (
                                    n0 * sin_pi_k_t - k * np.sin(pi * n0 * t))
                        sum_k_v_beta += sin_pi_k_a * sin_pi_k_x / (n0 ** 2 - k ** 2) * (
                                    np.cos(pi * n0 * t) - np.cos(pi * k * t))

                y2 += (4 / pi) * beta_n0 * n0 * np.sin(pi * n0 * a_bar) * (y_term1_beta + sum_k_y_beta)
                v2 += -4 * beta_n0 * n0**2 * np.sin(pi * n0 * a_bar) * (v_term1_beta + sum_k_v_beta)

        # Total y(x,t)
        y_vals = y1 + y2
        v_vals = v1 + v2
        # y_vals = y2
        # v_vals = v2

        # вручную сдвигаем поле перемещений, что бы в точке барьера перемещение было бы нулевым
        y_vals[0], y_vals[-1] = 0, 0
        if y_vals[point_barrier] < 0:
            y_vals -= y_vals[point_barrier]

        cur_energy = total_energy(y_vals, v_vals)
        print(f'cur_energy={cur_energy}')
        energy_data_global.append(cur_energy)
        energy_over_modes_lst = energy_over_modes(y_vals, v_vals, case_damping=False)
        graph(y_vals, v_vals, energy_over_modes_lst, case_plot=False, if_contact=True, case_time_delay=False)

    # plt.figure()
    # plt.plot(x_vals, y1_init)
    # plt.grid()
    # plt.figure()
    # plt.plot(x_vals, v1_init)
    # plt.grid()



    # plt.figure()
    # plt.plot(x_vals, y_vals)
    # plt.grid()
    # plt.figure()
    # plt.plot(x_vals, v_vals)
    # plt.grid()
    #
    # plt.show()

    # plt.figure()
    # plt.plot(x_vals, y_vals)
    # plt.title('disp')
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(x_vals, v_vals)
    # plt.title('vel')
    # plt.grid()
    # --------------------------------------------------------------

    # # если вышли из барьера, откатываем шаг назад и уменьшаем шаг на меньший из листа шагов
    # if (t_step_id > 0) and (Q_t >= 0):
    #     t_loc -= t_step
    #     t_global -= t_step
    #     t_global_lst.pop()
    #     Q_t = Q_t_previous
    #     t_loc_lst.pop()
    #     y_list = y_list_previous.copy()
    #
    #     t_step_id -= 1
    #
    #     continue
    #
    # if (t_step_id < len(t_step_lst)-1) and (Q_t < 0):
    #     t_step_id += 1
    # # --------------------------------------------------------------
    #
    # if first_step:
    #     # print('VI, kin = ')
    #     # print(kinetic_energy_id_sum)
    #     # print('VI, pot = ')
    #     # print(potential_energy_id_sum)
    #     # pass
    #     print('log pause')
    #
    # first_step = False
    #
    # if Q_t < 0:
    #     graph(y_list, vel_list)

    # # ----------------------------------------------------
    # # # последний шаг откатываем, поскольку Q>0 мы не хотим
    # # # if t_loc > 2e-8:
    # # y_list = y_list_previous
    # t_loc -= t_step
    # t_global -= t_step
    # t_global_lst.pop()
    # Q_list_time.pop()
    # y_barrier_global.pop()
    # y_end_global.pop()
    # global_en_lst.pop()
    # kinetic_energy_id_sum_lst.pop()
    # potential_energy_id_sum_lst.pop()
    # t_loc_lst.pop()
    # # # ----------------------------------------------------


    # if len(t_loc_lst) == 0:
    #     return disp_start, vel_start, t_loc_lst

    # y_list[point_barrier] = 0
    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ БЕЗ БАРЬЕРА
    # beam_no_VI_vibrations(y_list, vel_list)


    # print(y_vals[point_barrier])

    graph(y_vals, v_vals, energy_over_modes_lst, case_plot=False, case_time_delay=False, if_contact=True)

    return y_vals, v_vals


# Задачем начальные условия
def initial_conditions():
    # disp_start = 0 * np.sin(3 * np.pi * x_vals)
    # vel_start = -np.sin(np.pi * x_vals) - np.sin(5 * np.pi * x_vals)
    alpha_coef_lst = np.zeros(num_modes)
    alpha_coef_lst[1-1] = 1
    alpha_coef_lst[3 - 1] = -1
    # alpha_coef_lst[5 - 1] = 2
    # alpha_coef_lst[7 - 1] = -1
    # alpha_coef_lst[9 - 1] = -1
    # alpha_coef_lst[11 - 1] = -1
    # alpha_coef_lst[13 - 1] = 1
    # alpha_coef_lst[15 - 1] = -1

    # np.random.seed(467)  # Фиксируем генератор случайных чисел
    # alpha_coef_lst[:6] = np.random.uniform(-1, 1, 6)

    alpha_coef_lst = alpha_coef_lst

    beta_coef_lst = np.zeros(num_modes)

    # ---------- рисуем начальные условия ------
    alpha_coef_lst_plot = alpha_coef_lst / np.arange(1, num_modes + 1) / np.pi

    y1_init = np.zeros_like(x_vals)
    v1_init = np.zeros_like(x_vals)
    for n0, beta_n0 in enumerate(beta_coef_lst, start=1):
        if beta_n0 != 0:
            y1_init += beta_n0 * np.sin(pi * n0 * x_vals)

    for m0, alpha_m0 in enumerate(alpha_coef_lst_plot, start=1):
        if alpha_m0 != 0:
            v1_init += alpha_m0 * np.sin(pi * m0 * x_vals) * np.pi * m0


    cur_energy = total_energy(y1_init, v1_init)
    print(f'First value cur_energy={cur_energy}')
    energy_data_global.append(cur_energy)
    energy_over_modes_lst = energy_over_modes(y1_init, v1_init, case_damping=False)
    graph(y1_init, v1_init, energy_over_modes_lst, case_plot=False, if_contact=True, case_time_delay=False)

    # -----------------

    return alpha_coef_lst, beta_coef_lst


def graph(y_list, vel_list, energy_over_modes_lst, case_plot, case_time_delay=True, if_contact=False, case_time_delay_2=False):
    global def_graph_first_time

    if def_graph_first_time:  # для отрисовки начальных условий
        t_global_lst.append(0)
    # pass
    # if len(t_global_lst) % 200 == 0:
    # if round(t_global * 1e5) % 10 == 0:

    if not case_plot or def_graph_first_time:
        statement = True
    else:
        statement = len(t_global_lst) % 20 == 0

    if statement:
        # print(total_energy)
        # print(sum(total_energy))

        fig.suptitle(
            'Time=' + str('%.2f' % t_global) + 's=' + str('%.2f' % (t_global * 1e3)) + 'ms' + ', loc_bar = ' + str(
                a_bar) + '\n' + 'Contact ' + str(if_contact) + '. Number of VI = ' + str(number_VI) + ', tau = ' + str(round(tau, 3)))

        axs[0][0].set_title('Beam shape')
        axs[0][0].plot(x_vals, y_list, 'k', linewidth=1)
        axs[0][0].grid()
        # axs[0][0].set_ylim(-max_disp_start*1.0, max_disp_start*1.0)


        axs[0][1].set_title('Beam vel')
        # axs[2][1].plot(np.linspace(0, l2, point), vel_list, 'k', linewidth=1)
        axs[0][1].plot(x_vals, vel_list, 'g', linewidth=1)
        axs[0][1].grid()

        axs[1][0].set_title('Energy')
        # axs[2][1].plot(np.linspace(0, l2, point), vel_list, 'k', linewidth=1)
        axs[1][0].plot(t_global_lst, energy_data_global, 'g', linewidth=1)
        axs[1][0].set_ylim(0, energy_data_global[0] * 1.2)
        axs[1][0].grid()
        # вспомогательные линии начала и конца ВИ
        if case_plot or case_time_delay_2:
            for ii in range(len(time_start_VI)):
                axs[1][0].plot([time_start_VI[ii], time_start_VI[ii]],
                               [energy_during_VI[ii] - 0.1, energy_during_VI[ii] + 0.1], 'k--', linewidth=1)
                axs[1][0].plot([time_end_VI[ii], time_end_VI[ii]],
                               [energy_during_VI[ii] - 0.1, energy_during_VI[ii] + 0.1], 'k--', linewidth=1)

        df = pd.DataFrame(energy_over_modes_lst, columns=['origin'], index=range(1, num_modes_plot_energy + 1))
        axs[1][1].set_title('Energy distribution over modes\nMax energy = {} %'.format(
            round(max(df.origin[:]) / sum(df.origin[:]) * 100)), fontsize=7, pad=0)
        axs[1][1].bar(np.arange(1, num_modes_plot_energy + 1), df.origin[:])
        axs[1][1].set_xticks(np.arange(1, num_modes_plot_energy))

        if def_graph_first_time:
            plt.pause(1)
        if case_time_delay:
            plt.pause(0.01)
        else:
            plt.pause(1)
        if case_time_delay_2:
            plt.pause(12)
        axs[0][0].clear()
        axs[1][0].clear()
        axs[0][1].clear()
        axs[1][1].clear()

    if def_graph_first_time:
        energy_data_global.pop()
        t_global_lst.pop()
    def_graph_first_time = False  # строим начальное распределение перемещений и скорости



# -----------------------------------------------
pi = np.pi
Nx = 2000 + 1    # Number of x points
x_vals = np.linspace(0, 1, Nx)
a_bar = 0.5
point_barrier = round((Nx - 1) * a_bar)
N_max = 2000  # number terms in sum
num_modes = 150
gamma_ = 0.001

L = 1
# Параметры задачи
T = 3.0  # Время моделирования
c = 1.0  # Скорость волны
dx = L / (Nx - 1)
# ------------------------------------------------
t_global = 0.0
t_global_lst = []
energy_data_global = []
number_VI = 0
time_start_VI = []
time_end_VI = []
energy_during_VI = []
tau = 0
def_graph_first_time = True
# -----------------------------------------------
num_modes_plot_energy = 20
omega_m_damping = lambda m: (np.pi * m) * np.sqrt(1 - ((np.pi * m) ** 2 * gamma_ ** 2) / 4)
omega_m = lambda m: (np.pi * m)
omega_m_damping_lst = np.array([omega_m_damping(i) for i in range(1, num_modes_plot_energy + 1)])
omega_m_lst = np.array([omega_m(i) for i in range(1, num_modes_plot_energy + 1)])



# Функция для расчёта энергии
def total_energy(u, v):
    kinetic = np.sum(v ** 2) / 2
    potential = np.sum((np.diff(u) / dx) ** 2) / 2
    return (kinetic + potential) * dx


# Функция для расчёта распределения энергии
def energy_over_modes(y_vals, v_vals, case_damping):
    if case_damping:
        case_omega_n_lst = omega_m_damping_lst
    else:
        case_omega_n_lst = omega_m_lst
    modal_coords, modal_velocities = disp_vel_to_alpha_beta_lst(y_vals, v_vals, num_modes_plot_energy)
    energy_over_modes_lst = 0.5 * (modal_velocities**2 + case_omega_n_lst**2 * modal_coords**2)

    return energy_over_modes_lst





# t_step_lst = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]  # лист временных шагов, которые будем динамически менять


# относительное различие между соседними шагами t_step_lst не превышало заданного значения
def create_time_steps(dt_smallest, dt_biggest, relative_difference_max):
    # Проверяем, чтобы входные параметры были валидны
    if dt_smallest <= 0 or dt_biggest <= 0 or dt_smallest >= dt_biggest:
        raise ValueError("dt_smallest должен быть > 0, dt_biggest должен быть > 0, и dt_smallest < dt_biggest.")
    if relative_difference_max <= 0:
        raise ValueError("relative_difference_max должен быть больше 0.")

    # Логарифмическое увеличение временных шагов
    steps = [dt_smallest]
    while steps[-1] < dt_biggest:
        next_step = steps[-1] * (1 + relative_difference_max)
        if next_step > dt_biggest:  # Ограничиваем последний шаг значением dt_biggest
            steps.append(dt_biggest)
            break
        steps.append(next_step)

    return np.array(steps)


# Пример использования
dt_smallest = 1e-15
dt_biggest = 8e-4
relative_difference_max = 0.3  # Относительная разница не больше 20%

t_step_lst = create_time_steps(dt_smallest, dt_biggest, relative_difference_max)
print(t_step_lst)


t_step_id = 0
t_step = t_step_lst[t_step_id]

t_end = 4.00
time_is_up = False  # флаг на то, закончилось ли время
# ---------------------------------------------

print('Начинаем фигачить')

fig, axs = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)  # создаем саб плот из 2 графиков
# plt.subplots_adjust(wspace=0.4, hspace=0.7)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.pause(0.1)

# Задаем начальную деформацию и скорость балки
[alpha_coef_lst_start, beta_coef_lst_start] = initial_conditions()
print('Задали НУ')

print('Запускаем динамику')



# while True:
# while not time_is_up:
# ---------------------------------------------------------
# Switch to case without VI
beta_coef_lst, alpha_coef_lst_draft = beta_coef_lst_start.copy(), alpha_coef_lst_start.copy()
alpha_coef_lst = ((alpha_coef_lst_draft + (np.pi * np.arange(1, num_modes + 1)) ** 2 * gamma_ / 2 * beta_coef_lst) /
                  np.array([omega_m_damping(ii+1) for ii in range(num_modes)]))

print('Time = ', str(t_global))
# Вначале запускаем динамику балки без барьера
# [disp_start, vel_start] = beam_no_VI_vibrations(disp_start, vel_start, gamma_, just_no_VI=False)
[alpha_lst_init, beta_lst_init] = beam_no_VI_vibrations(alpha_coef_lst, beta_coef_lst, gamma_, just_no_VI=False)
print(alpha_lst_init)
print(beta_lst_init)
# ---------------------------------------------------------
first_loop = False
while True:
    # ---------------------------------------------------------
    # Firstly we are modelling VI
    if first_loop:
        beta_coef_lst, alpha_coef_lst_draft = beta_coef_lst_start.copy(), alpha_coef_lst_start.copy()
        first_loop = False
    else:
        beta_coef_lst, alpha_coef_lst_draft = beta_lst_init.copy(), alpha_lst_init.copy()

    alpha_coef_lst = alpha_coef_lst_draft / np.arange(1, num_modes + 1) / np.pi
    print(alpha_coef_lst)
    print(beta_coef_lst)

    # alpha_coef_lst = np.array([-1])
    # beta_coef_lst = np.array([0])

    print('Time = ', str(t_global))
    [disp_start, vel_start] = beam_with_VI_vibrations(alpha_coef_lst, beta_coef_lst)
    # print(t_loc_lst)
    print(f'log = {disp_start[point_barrier]}')
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Switch to case without VI
    beta_coef_lst, alpha_coef_lst_draft = disp_vel_to_alpha_beta_lst(disp_start, vel_start, num_modes)
    alpha_coef_lst = ((alpha_coef_lst_draft + (np.pi * np.arange(1, num_modes + 1)) ** 2 * gamma_ / 2 * beta_coef_lst) /
                      np.array([omega_m_damping(ii+1) for ii in range(num_modes)]))

    print('Time = ', str(t_global))
    # Вначале запускаем динамику балки без барьера
    # [disp_start, vel_start] = beam_no_VI_vibrations(disp_start, vel_start, gamma_, just_no_VI=False)
    [alpha_lst_init, beta_lst_init] = beam_no_VI_vibrations(alpha_coef_lst, beta_coef_lst, gamma_, just_no_VI=False)
    print(alpha_lst_init)
    print(beta_lst_init)
    # ---------------------------------------------------------

    # # ---------------------------------------------------------
    # #  we are modelling VI
    # beta_coef_lst, alpha_coef_lst_draft = beta_lst_init.copy(), alpha_lst_init.copy()
    # alpha_coef_lst = alpha_coef_lst_draft / np.arange(1, num_modes + 1) / np.pi
    # # print(alpha_coef_lst)
    # # print(max(abs(alpha_coef_lst)))
    # # print(beta_coef_lst)
    # # print(max(abs(beta_coef_lst)))
    #
    # # alpha_coef_lst = np.array([-1])
    # # beta_coef_lst = np.array([0])
    #
    # print('Time = ', str(t_global))
    # [disp_start, vel_start] = beam_with_VI_vibrations(alpha_coef_lst, beta_coef_lst)
    # # print(t_loc_lst)
    #
    # print(disp_start[point_barrier-5:point_barrier+5])
    # print(vel_start[point_barrier-5:point_barrier+5])
    # # ---------------------------------------------------------
    #
    #
    # # ---------------------------------------------------------
    # print('Time = ', str(t_global))
    # # Switch to case without VI
    # beta_coef_lst, alpha_coef_lst_draft = disp_vel_to_alpha_beta_lst(disp_start, vel_start, num_modes)
    # alpha_coef_lst = ((alpha_coef_lst_draft + (np.pi * np.arange(1, num_modes + 1)) ** 2 * gamma_ / 2 * beta_coef_lst) /
    #                   np.array([omega_m_damping(ii + 1) for ii in range(num_modes)]))
    #
    # print('Time = ', str(t_global))
    # # Вначале запускаем динамику балки без барьера
    # # [disp_start, vel_start] = beam_no_VI_vibrations(disp_start, vel_start, gamma_, just_no_VI=False)
    # [alpha_lst_init, beta_lst_init] = beam_no_VI_vibrations(alpha_coef_lst, beta_coef_lst, gamma_, just_no_VI=True)
    # # ---------------------------------------------------------


