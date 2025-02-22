import numpy as np
from sympy import *
import pylab as plt
import time
import pandas as pd


# динамика балки без барьера
# def beam_no_VI_vibrations(y_vals, v_vals, gamma_, just_no_VI=False):
def beam_no_VI_vibrations(alpha_lst_init, beta_lst_init, gamma_, just_no_VI=False):
    global t_global, time_is_up, t_global_lst, energy_data_global, tau, first_time_tau

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
        v_init += (-(np.pi * m0)**2 * gamma_ / 2 * beta_lst_init[m0-1] + omega_m_damping(m0, gamma_) * alpha_i) * np.sin(np.pi * m0 * x_vals)
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
        if (len(energy_data_global) > 0) and (energy_data_global[-1] <= energy_data_global[0] / np.exp(1)) and first_time_tau:
            time_is_up = True
            first_time_tau = False
            tau = t_global
            print('----------------------------------')
            print(f'tau = {t_global}')
            print(f'Number VI = {number_VI}')
            print('----------------------------------')
            graph(u_next, v_next, energy_over_modes_lst[:num_modes_plot_energy], case_plot=False, case_time_delay=False, case_time_delay_2=True)
            # with open(r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\26_01_2025_string_multyVI\time_energy_test_2.txt', 'w') as cur_file:
            #     cur_file.write(str(t_global_lst))
            # with open(r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\26_01_2025_string_multyVI\energy_test_2.txt', 'w') as cur_file:
            #     cur_file.write(str(energy_data_global))

            return u_next, v_next


        first_step = False
        # t_global_prev = t_global

        t_global += dt_next
        t_global_lst.append(t_global)
        t_local += dt_next

        # print(f't_global_step = {t_global - t_global_prev}')
        # print(f'dt_next = {dt_next}')


        # Основной вычислительный цикл
        u_next = np.zeros(Nx)
        for m0, alpha_i in enumerate(alpha_lst_init, start=1):
            u_next += np.exp(-(np.pi * m0) ** 2 * gamma_ / 2 * t_local) * (
                        alpha_i * np.sin(omega_m_damping(m0, gamma_) * t_local) + beta_lst_init[m0 - 1] * np.cos(
                    omega_m_damping(m0, gamma_) * t_local)) * np.sin(np.pi * m0 * x_vals)

        # print(f'u_next[point_barrier] = {u_next[point_barrier]}')
        # print(f't_step_id = {t_step_id}')

        # print(f'2 = {dt}')

        # Граничные условия
        u_next[0] = 0
        u_next[-1] = 0

        # print(f'not just_no_VI = {not just_no_VI}')
        # print(f'if_we_are_ready_for_new_VI = {if_we_are_ready_for_new_VI}')
        # print(f'(t_step_id > 0) = {(t_step_id > 0)}')
        # print(f'(u_next[point_barrier] < 0) = {(u_next[point_barrier] < 0)}')

        if not just_no_VI and if_we_are_ready_for_new_VI:
            # если зашли в барьер, откатываем шаг назад и уменьшаем шаг на меньший из листа шагов
            if (t_step_id > 0) and (u_next[point_barrier] < 0):
                pass_for_next_while = True
                in_barrier = True
                # print('In barrier. No VI. Trying in')
                t_global -= dt_next
                t_global_lst.pop()
                t_local -= dt_next

                t_step_id -= 1
                # print(f't_step_id = {t_step_id}')
                dt_next = t_step_lst[t_step_id]

                continue

        # print(f'PREV = {dt_prev}')
        # print(f'NEXT = {dt_next}')
        v_next = np.zeros(Nx)
        v_next_t_lst = []
        u_next_t_lst = []
        for m0, alpha_i in enumerate(alpha_lst_init, start=1):
            v_next_t = (np.exp(-(np.pi * m0) ** 2 * gamma_ / 2 * t_local) * (-(np.pi * m0) ** 2 * gamma_ / 2 * (
                    alpha_i * np.sin(omega_m_damping(m0, gamma_) * t_local) + beta_lst_init[m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_) * t_local)) + omega_m_damping(m0, gamma_) * (alpha_i * np.cos(
                omega_m_damping(m0, gamma_) * t_local) - beta_lst_init[m0 - 1] * np.sin(
                omega_m_damping(m0, gamma_) * t_local))))
            v_next_t_lst.append(v_next_t)
            v_next += v_next_t * np.sin(np.pi * m0 * x_vals)


            u_next_t = np.exp(-(np.pi * m0) ** 2 * gamma_ / 2 * t_local) * (
                    alpha_i * np.sin(omega_m_damping(m0, gamma_) * t_local) + beta_lst_init[m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_) * t_local))
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
            # print('No VI. Increase steps')
            # print(f't_step_id = {t_step_id}')

        dt_next = t_step_lst[t_step_id]

        # print(f'1 = {u[point_barrier]}')

        # если выруливаем из случайного минуса после ВИ, то мы открыты к новому удару
        if u_next[point_barrier] > 1e-4:
            if_we_are_ready_for_new_VI = True


    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ С БАРЬЕРОМ
    print('Before VI')
    graph(u_next, v_next, energy_over_modes_lst[:num_modes_plot_energy], case_plot=False, case_time_delay=False)

    return u_next, v_next


def modal_coefficients(x, y, num_modes, string_length):
    coefficients = []
    for k in range(1, num_modes + 1):
        phi_k = np.sin(k * np.pi / string_length * x)
        a_k = (2 / string_length) * np.trapz(y * phi_k, x=x)
        coefficients.append(a_k)

    return np.array(coefficients)

def disp_vel_to_alpha_beta_lst(y_vals, v_vals, num_modes, string_length=1.0, left_string=True):
    # Функция для вычисления коэффициентов модального разложения
    if round(string_length, 3) == 1:
        modal_coords = modal_coefficients(x_vals, y_vals, num_modes, string_length)
        modal_velocities = modal_coefficients(x_vals, v_vals, num_modes, string_length)
    elif left_string:
        modal_coords = modal_coefficients(x_vals_L, y_vals, num_modes, string_length)
        modal_velocities = modal_coefficients(x_vals_L, v_vals, num_modes, string_length)
    else:
        modal_coords = modal_coefficients(x_vals_R, y_vals, num_modes, string_length)
        modal_velocities = modal_coefficients(x_vals_R, v_vals, num_modes, string_length)

    return modal_coords, modal_velocities




# аналитическая сила ВИ для произвольных НУ
# -----------------------------------------

# Динамика балки с барьером
def beam_with_VI_vibrations(alpha_coef_lst_L, alpha_coef_lst_R, beta_coef_lst_L, beta_coef_lst_R, gamma_, just_VI=False):

    print('with barrier')
    global t_global, time_is_up, t_global_lst, energy_data_global, tau, first_time_tau

    # нам надо получить модальные коэффициенты для двух новых струн


    t_local = 0
    if_we_are_ready_for_new_VI = False

    in_barrier = False

    t_step_id = 0
    dt_next = t_step_lst[t_step_id]

    # Инициализация массивов решения
    diff_u_init_L, diff_u_init_R = 0, 0
    for n0, beta_i in enumerate(beta_coef_lst_L, start=1):
        diff_u_init_L += beta_i * (np.pi * n0 / a_bar) * np.cos(np.pi * n0)
        diff_u_init_R += beta_coef_lst_R[n0-1] * (np.pi * n0 / (1-a_bar))

    VI_force_init = -diff_u_init_L + diff_u_init_R
    print(f'VI_force_init = {VI_force_init}')

    # v_init_L, v_init_R = np.zeros(Nx), np.zeros(Nx)
    # for m0, alpha_i in enumerate(alpha_coef_lst_L, start=1):
    #     v_init_L += (-(np.pi * m0 / a_bar) ** 2 * gamma_ / 2 * beta_coef_lst_L[m0 - 1] +
    #                omega_m_damping(m0, string_length=a_bar) * alpha_i) * np.sin(np.pi * m0 / a_bar * x_vals)
    #     v_init_R += (-(np.pi * m0 / (1-a_bar)) ** 2 * gamma_ / 2 * beta_coef_lst_R[m0 - 1] +
    #                  omega_m_damping(m0, string_length=(1-a_bar)) * alpha_coef_lst_R[m0-1]) * np.sin(np.pi * m0 / (1-a_bar) * x_vals)
    # # Граничные условия
    # v_init_L[0], v_init_R[0], v_init_L[-1], v_init_R[-1] = 0, 0, 0, 0

    energy_data = []
    flag = True

    is_first_no_VI = True

    first_steps = True

    VI_force = VI_force_init
    pass_for_next_while = False
    while (VI_force <= 0) or pass_for_next_while or first_steps or just_VI:
        pass_for_next_while = False
    # while True:

        # if (t_global >= t_end) and not time_is_up:
        if (len(energy_data_global) > 0) and (energy_data_global[-1] <= energy_data_global[0] / np.exp(1)) and first_time_tau:
            time_is_up = True
            first_time_tau = False
            tau = t_global
            print('----------------------------------')
            print(f'tau = {t_global}')
            print(f'Number VI = {number_VI}')
            print('----------------------------------')
            graph(np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0), energy_over_modes_lst[:num_modes_plot_energy], case_plot=False, case_time_delay=False, case_time_delay_2=True, if_contact=True)

            # with open(r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\26_01_2025_string_multyVI\time_energy_test_2.txt', 'w') as cur_file:
            #     cur_file.write(str(t_global_lst))
            # with open(r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\26_01_2025_string_multyVI\energy_test_2.txt', 'w') as cur_file:
            #     cur_file.write(str(energy_data_global))

            return np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0)

        # first_steps = False
        t_global += dt_next
        t_global_lst.append(t_global)
        t_local += dt_next

        # Основной вычислительный цикл
        diff_u_next_L, diff_u_next_R = 0, 0
        for m0, alpha_i in enumerate(alpha_coef_lst_L, start=1):
            diff_u_next_L += np.exp(-(np.pi * m0 / a_bar) ** 2 * gamma_ / 2 * t_local) * (
                    alpha_i * np.sin(omega_m_damping(m0, gamma_, string_length=a_bar) * t_local) + beta_coef_lst_L[m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_, string_length=a_bar) * t_local)) * (np.pi * m0 / a_bar) * np.cos(np.pi * m0)

            diff_u_next_R += np.exp(-(np.pi * m0 / (1-a_bar)) ** 2 * gamma_ / 2 * t_local) * (
                    alpha_coef_lst_R[m0-1] * np.sin(omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * t_local) + beta_coef_lst_R[
                m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * t_local)) * (np.pi * m0 / (1-a_bar))

        VI_force = -diff_u_next_L + diff_u_next_R
        if VI_force <= 0:
            first_steps = False
        # print(f'VI_force = {VI_force}')

        # если WE ARE GOING OUT FROM барьер, откатываем шаг назад и уменьшаем шаг на меньший из листа шагов
        if not just_VI and not first_steps:
            if (t_step_id > 0) and (VI_force > 0):
                pass_for_next_while = True
                in_barrier = True
                # print('In barrier. VI. Trying out')
                t_global -= dt_next
                t_global_lst.pop()
                t_local -= dt_next

                t_step_id -= 1
                # print(f'log t_step_id = {t_step_id}')
                dt_next = t_step_lst[t_step_id]

                continue

        # print(f'PREV = {dt_prev}')
        # print(f'NEXT = {dt_next}')
        u_next_L, u_next_R = np.zeros(Nx_L), np.zeros(Nx_R)
        v_next_L, v_next_R = np.zeros(Nx_L), np.zeros(Nx_R)
        v_next_t_lst_L, v_next_t_lst_R = [], []
        u_next_t_lst_L, u_next_t_lst_R = [], []
        for m0, alpha_i in enumerate(alpha_coef_lst_L, start=1):
            u_next_t_L = np.exp(-(np.pi * m0 / a_bar) ** 2 * gamma_ / 2 * t_local) * (
                    alpha_i * np.sin(omega_m_damping(m0, gamma_, string_length=a_bar) * t_local) + beta_coef_lst_L[m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_, string_length=a_bar) * t_local))

            u_next_t_R = np.exp(-(np.pi * m0 / (1-a_bar)) ** 2 * gamma_ / 2 * t_local) * (
                    alpha_coef_lst_R[m0 - 1] * np.sin(omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * t_local) + beta_coef_lst_R[m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * t_local))

            u_next_t_lst_L.append(u_next_t_L)
            u_next_t_lst_R.append(u_next_t_R)

            u_next_L += u_next_t_L * np.sin(np.pi * m0 / a_bar * x_vals_L)
            u_next_R += u_next_t_R * np.sin(np.pi * m0 / (1-a_bar) * x_vals_R)

            v_next_t_L = (np.exp(-(np.pi * m0 / a_bar) ** 2 * gamma_ / 2 * t_local) * (-(np.pi * m0 / a_bar) ** 2 * gamma_ / 2 * (
                    alpha_i * np.sin(omega_m_damping(m0, gamma_, string_length=a_bar) * t_local) + beta_coef_lst_L[m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_, string_length=a_bar) * t_local)) + omega_m_damping(m0, gamma_, string_length=a_bar) * (alpha_i * np.cos(
                omega_m_damping(m0, gamma_, string_length=a_bar) * t_local) - beta_coef_lst_L[m0 - 1] * np.sin(
                omega_m_damping(m0, gamma_, string_length=a_bar) * t_local))))

            v_next_t_R = (np.exp(-(np.pi * m0 / (1-a_bar)) ** 2 * gamma_ / 2 * t_local) * (-(np.pi * m0 / (1-a_bar)) ** 2 * gamma_ / 2 * (
                    alpha_coef_lst_R[m0 - 1] * np.sin(omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * t_local) + beta_coef_lst_R[m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * t_local)) + omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * (alpha_coef_lst_R[m0 - 1] * np.cos(
                omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * t_local) - beta_coef_lst_R[m0 - 1] * np.sin(
                omega_m_damping(m0, gamma_, string_length=(1-a_bar)) * t_local))))

            v_next_t_lst_L.append(v_next_t_L)
            v_next_t_lst_R.append(v_next_t_R)

            v_next_L += v_next_t_L * np.sin(np.pi * m0 / a_bar * x_vals_L)
            v_next_R += v_next_t_R * np.sin(np.pi * m0 / (1-a_bar) * x_vals_R)



        # cur_energy = total_energy(u_next, v_cur)
        kinetic_energy_L = a_bar * 0.25 * sum([v_next_t_lst_L[ii] ** 2 for ii in range(len(v_next_t_lst_L))])
        kinetic_energy_R = (1 - a_bar) * 0.25 * sum([v_next_t_lst_R[ii] ** 2 for ii in range(len(v_next_t_lst_R))])
        potential_energy_L = a_bar * 0.25 * sum([(np.pi * (ii + 1) / a_bar * u_next_t_lst_L[ii]) ** 2 for ii in range(len(u_next_t_lst_L))])
        potential_energy_R = (1 - a_bar) * 0.25 * sum([(np.pi * (ii + 1) / (1 - a_bar) * u_next_t_lst_R[ii]) ** 2 for ii in range(len(u_next_t_lst_R))])
        cur_energy_L = kinetic_energy_L + potential_energy_L
        cur_energy_R = kinetic_energy_R + potential_energy_R
        cur_energy_sum = cur_energy_L + cur_energy_R
        # print(cur_energy_sum)
        energy_data_global.append(cur_energy_sum)
        # # print(cur_energy)
        # energy_over_modes_lst = energy_over_modes(u_next, v_cur, case_damping=True)
        energy_over_modes_lst = 0.25 * (np.array([v_next_t_lst_L[ii] ** 2 for ii in range(len(v_next_t_lst_L))]) +
                                        np.array([v_next_t_lst_R[ii] ** 2 for ii in range(len(v_next_t_lst_R))]) +
                                        np.array([(np.pi * (ii + 1) / a_bar * u_next_t_lst_L[ii]) ** 2 for ii in range(len(u_next_t_lst_L))]) +
                                        np.array([(np.pi * (ii + 1) / (1 - a_bar) * u_next_t_lst_R[ii]) ** 2 for ii in range(len(u_next_t_lst_R))]))

        graph(np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0), energy_over_modes_lst[:num_modes_plot_energy], case_plot=True, if_contact=True)
        # print(cur_energy)

        # ------------------------------------------------------

        # если выходим из барьера увеличиваем шаги
        # print((u_next[point_barrier]))
        if (not in_barrier) and (t_step_id < len(t_step_lst) - 1):
            t_step_id += 1
            # print('VI. Increase steps')
            # print(f't_step_id = {t_step_id}')

        dt_next = t_step_lst[t_step_id]

        # print(f'1 = {u[point_barrier]}')

    print('Before VI')
    graph(np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0), energy_over_modes_lst[:num_modes_plot_energy], case_plot=False, case_time_delay=False, if_contact=True)

    return np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0)




# Задачем начальные условия
def initial_conditions():
    beta_coef_lst = np.zeros(num_modes)

    # disp_start = 0 * np.sin(3 * np.pi * x_vals)
    # vel_start = -np.sin(np.pi * x_vals) - np.sin(5 * np.pi * x_vals)
    alpha_coef_lst = np.zeros(num_modes)
    alpha_coef_lst[1-1] = 1  #8
    alpha_coef_lst[2 - 1] = -1 #5
    # alpha_coef_lst[3 - 1] = -2
    # alpha_coef_lst[5 - 1] = 2
    # alpha_coef_lst[7 - 1] = -1
    # alpha_coef_lst[9 - 1] = -1
    # alpha_coef_lst[11 - 1] = -1
    # alpha_coef_lst[13 - 1] = 1
    # alpha_coef_lst[15 - 1] = -1

    # np.random.seed(165)  # Фиксируем генератор случайных чисел
    # alpha_coef_lst[:6] = np.random.uniform(-1, 1, 6)
    print('alpha_coef_lst')
    print(alpha_coef_lst[:20])

    # --------------------------------------------------------------------
    # проверяем, надо ли менять знак начального поля скорости
    v1_init = 0
    alpha_coef_lst_right = ((alpha_coef_lst + (np.pi * np.arange(1, num_modes + 1)) ** 2 * gamma_ / 2 * beta_coef_lst) /
                          np.array([omega_m_damping(ii + 1, gamma_) for ii in range(num_modes)]))
    for m0, alpha_i in enumerate(alpha_coef_lst_right, start=1):
        v1_init += (-(np.pi * m0)**2 * gamma_ / 2 * beta_coef_lst[m0-1] + omega_m_damping(m0, gamma_) * alpha_i) * np.sin(np.pi * m0 * a_bar)
    if v1_init < 0:
        alpha_coef_lst = -alpha_coef_lst
        alpha_coef_lst_right = -alpha_coef_lst_right
    # ----------------------------------------------------------------------

    # ---------- рисуем начальные условия ------
    y1_init = np.zeros_like(x_vals)
    v1_init = np.zeros_like(x_vals)
    for n0, beta_i in enumerate(beta_coef_lst, start=1):
        y1_init += beta_i * np.sin(np.pi * n0 * x_vals)

    for m0, alpha_i in enumerate(alpha_coef_lst_right, start=1):
        v1_init += (-(np.pi * m0)**2 * gamma_ / 2 * beta_coef_lst[m0-1] + omega_m_damping(m0, gamma_) * alpha_i) * np.sin(np.pi * m0 * x_vals)


    cur_energy = total_energy(y1_init, v1_init)
    print(f'First value cur_energy={cur_energy}')
    energy_data_global.append(cur_energy)
    energy_over_modes_lst = energy_over_modes(y1_init, v1_init, case_damping=False)
    graph(y1_init, v1_init, energy_over_modes_lst, case_plot=False, if_contact=False, case_time_delay=False)

    # -----------------

    return alpha_coef_lst, beta_coef_lst


def graph(y_list, vel_list, energy_over_modes_lst, case_plot, case_time_delay=False, if_contact=False, case_time_delay_2=False):
    global def_graph_first_time

    if def_graph_first_time:  # для отрисовки начальных условий
        t_global_lst.append(0)
    # pass
    # if len(t_global_lst) % 200 == 0:
    # if round(t_global * 1e5) % 10 == 0:

    if not case_plot or def_graph_first_time:
        statement = True
    else:
        statement = len(t_global_lst) % 200 == 0

    if statement:
        # print(total_energy)
        # print(sum(total_energy))

        fig.suptitle(
            'Time=' + str('%.2f' % t_global) + 's=' + str('%.2f' % (t_global * 1e3)) + 'ms' + ', loc_bar = ' + str(
                a_bar) + ', num modes = ' + str(num_modes) + '\n' + 'Contact ' + str(if_contact) + '. Number of VI = ' + str(number_VI) + ', tau = ' + str(round(tau, 3)))

        axs[0][0].set_title('Beam shape')
        axs[0][0].plot(x_vals, y_list, 'k', linewidth=1)
        axs[0][0].grid()
        # axs[0][0].set_ylim(-max_disp_start*1.0, max_disp_start*1.0)


        axs[0][1].set_title('Beam vel')
        # axs[2][1].plot(np.linspace(0, l2, point), vel_list, 'k', linewidth=1)
        axs[0][1].plot(x_vals, vel_list, 'g', linewidth=1)
        if def_graph_first_time:
            axs[0][1].plot([a_bar, a_bar], [min(vel_list), max(vel_list)], 'k--', linewidth=1)
        axs[0][1].grid()

        axs[1][0].set_title('Energy')
        # axs[2][1].plot(np.linspace(0, l2, point), vel_list, 'k', linewidth=1)
        axs[1][0].plot(t_global_lst, energy_data_global, 'g', linewidth=1)
        axs[1][0].set_ylim(0, energy_data_global[0] * 1.2)
        axs[1][0].grid()
        # вспомогательные линии начала и конца ВИ
        for ii in range(len(contact_borders)):
            axs[1][0].plot([contact_borders[ii], contact_borders[ii]],
                           [0, energy_data_global[0]], 'k--', linewidth=1)

        df = pd.DataFrame(energy_over_modes_lst, columns=['origin'], index=range(1, num_modes_plot_energy + 1))
        axs[1][1].set_title('Energy distribution over modes\nMax energy = {} %'.format(
            round(max(df.origin[:]) / sum(df.origin[:]) * 100)), fontsize=7, pad=0)
        axs[1][1].bar(np.arange(1, num_modes_plot_energy + 1), df.origin[:])
        axs[1][1].set_xticks(np.arange(1, num_modes_plot_energy))

        if def_graph_first_time:
            plt.pause(2)
        if case_time_delay:
            plt.pause(0.01)
        else:
            plt.pause(0.01)
        if case_time_delay_2:
            plt.pause(0.01)
        axs[0][0].clear()
        axs[1][0].clear()
        axs[0][1].clear()
        axs[1][1].clear()

    if def_graph_first_time:
        energy_data_global.pop()
        t_global_lst.pop()
    def_graph_first_time = False  # строим начальное распределение перемещений и скорости







# Функция для расчёта энергии
def total_energy(u, v):
    kinetic = np.sum(v ** 2) / 2
    potential = np.sum((np.diff(u) / dx) ** 2) / 2
    return (kinetic + potential) * dx


# Функция для расчёта распределения энергии
def energy_over_modes(y_vals, v_vals, case_damping):
    if case_damping:
        case_omega_n_lst = omega_m_damping_lst.copy()
    else:
        case_omega_n_lst = omega_m_lst.copy()
    modal_coords, modal_velocities = disp_vel_to_alpha_beta_lst(y_vals, v_vals, num_modes_plot_energy)
    energy_over_modes_lst = 0.5 * (modal_velocities**2 + case_omega_n_lst**2 * modal_coords**2)

    return energy_over_modes_lst



# -----------------------------------------------
pi = np.pi
Nx = 2000 + 1  # Number of x points
x_vals = np.linspace(0, 1, Nx)
# -----------------------------------------------
L = 1
# Параметры задачи
T = 3.0  # Время моделирования
c = 1.0  # Скорость волны
dx = L / (Nx - 1)
# ------------------------------------------------
tau_lst = []
N_max = 2000  # number terms in sum
# num_modes = 100
gamma_ = 0.001
gamma_noVI = 0.001
gamma_VI = 0.001
# ----------------------------------------------------------------------------------
num_modes_plot_energy = 20
omega_m_damping = lambda m, gamma_cur, string_length=1: (np.pi * m / string_length) * np.sqrt(
    1 - ((np.pi * m / string_length) ** 2 * gamma_cur ** 2) / 4)
omega_m = lambda m, string_length=1: (np.pi * m / string_length)
omega_m_damping_lst = np.array([omega_m_damping(i, gamma_noVI) for i in range(1, num_modes_plot_energy + 1)])
omega_m_lst = np.array([omega_m(i) for i in range(1, num_modes_plot_energy + 1)])
# -----------------------------------------------------------------------------------------------


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
dt_biggest = 1e-3
relative_difference_max = 0.3  # Относительная разница не больше 20%

t_step_lst = create_time_steps(dt_smallest, dt_biggest, relative_difference_max)
print(t_step_lst)



# ---------------------------------------------





# while True:
# while not time_is_up:
# # ---------------------------------------------------------
# # Switch to case without VI
# beta_coef_lst, alpha_coef_lst_draft = beta_coef_lst_start.copy(), alpha_coef_lst_start.copy()
# alpha_coef_lst = ((alpha_coef_lst_draft + (np.pi * np.arange(1, num_modes + 1)) ** 2 * gamma_ / 2 * beta_coef_lst) /
#                   np.array([omega_m_damping(ii+1) for ii in range(num_modes)]))
#
# print('Time = ', str(t_global))
# # Вначале запускаем динамику балки без барьера
# # [disp_start, vel_start] = beam_no_VI_vibrations(disp_start, vel_start, gamma_, just_no_VI=False)
# [alpha_lst_init, beta_lst_init] = beam_no_VI_vibrations(alpha_coef_lst, beta_coef_lst, gamma_, just_no_VI=False)
# print(alpha_lst_init)
# print(beta_lst_init)
# # ---------------------------------------------------------

a_bar_lst = np.arange(0.1, 0.901, 0.02).tolist()
start_time = time.time()
for i_id, a_bar_cur in enumerate(a_bar_lst):
    # -----------------------------------------------

    a_bar = round(a_bar_cur, 4)  # -----
    point_barrier = round((Nx - 1) * a_bar)  # -----
    Nx_L = point_barrier + 1
    Nx_R = Nx - point_barrier
    x_vals_L = np.linspace(0, a_bar, Nx_L)  # -----
    x_vals_R = np.linspace(0, 1 - a_bar, Nx_R)  # -------



    # проверяем, удовлетворяет ли количество мод заданному коэффициенту демпфирования
    # ---------------------------------------
    def finding_max_number_mode(gamma, string_length, num_modes_wish):
        omega_sqrt = lambda n: 1 - (np.pi * n / string_length * gamma) ** 2 / 4
        if omega_sqrt(num_modes_wish) > 0:
            return num_modes_wish
        else:
            for n0 in range(1, num_modes_wish + 5):
                if omega_sqrt(n0) < 0:
                    #             print(f'max number of the mode = {n}')
                    # raise ValueError(f'Максимально возможный номер моды = {n0 - 1}')
                    return n0 - 1


    num_modes_draft = 300

    print('For whole string')
    num_modes_draft = finding_max_number_mode(gamma_, 1.0, num_modes_draft)
    print('For left part')
    num_modes_draft = finding_max_number_mode(gamma_, a_bar, num_modes_draft)
    print('For right part')
    num_modes_draft = finding_max_number_mode(gamma_, (1 - a_bar), num_modes_draft)

    num_modes = num_modes_draft
    # ---------------------------------


    # ------------------------------------------------
    t_global = 0.0
    t_global_lst = []
    energy_data_global = []
    number_VI = 0
    time_start_VI = []
    time_end_VI = []
    energy_during_VI = []
    tau = 0

    first_time_tau = True
    def_graph_first_time = True
    contact_borders = []  # будем записывать значения начала и конца контактов для визуализации на графике энергии
    # -----------------------------------------------

    # -----------------------------------------------------------------------------------------------



    just_no_VI = False
    just_VI = False
    first_loop = True
    num_loop = 0

    t_step_id = 0
    t_step = t_step_lst[t_step_id]
    t_end = 4.00
    time_is_up = False  # флаг на то, закончилось ли время

    print('Начинаем фигачить')

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)  # создаем саб плот из 2 графиков
    # plt.subplots_adjust(wspace=0.4, hspace=0.7)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.pause(0.1)

    # Задаем начальную деформацию и скорость балки
    [alpha_coef_lst_start, beta_coef_lst_start] = initial_conditions()
    print('Задали НУ')

    print('Запускаем динамику')


    while True:
        num_loop += 1
        # ---------------------------------------------------------
        # Switch to case without VI
        if first_loop:
            beta_coef_lst, alpha_coef_lst_draft = beta_coef_lst_start.copy(), alpha_coef_lst_start.copy()
            first_loop = False
        else:
            # beta_coef_lst, alpha_coef_lst_draft = beta_lst_init.copy(), alpha_lst_init.copy()
            beta_coef_lst, alpha_coef_lst_draft = disp_vel_to_alpha_beta_lst(disp_start, vel_start, num_modes)

        # beta_coef_lst, alpha_coef_lst_draft = disp_vel_to_alpha_beta_lst(disp_start, vel_start, num_modes)
        alpha_coef_lst = ((alpha_coef_lst_draft + (np.pi * np.arange(1, num_modes + 1)) ** 2 * gamma_noVI / 2 * beta_coef_lst) /
                          np.array([omega_m_damping(ii + 1, gamma_noVI) for ii in range(num_modes)]))

        print('Time = ', str(t_global))

        # if num_loop == 20:
        #     just_no_VI = True
        # else:
        #     just_no_VI = False
        [u_sum, v_sum] = beam_no_VI_vibrations(alpha_coef_lst, beta_coef_lst, gamma_noVI, just_no_VI=just_no_VI)
        # [u_sum, v_sum] = beam_no_VI_vibrations(alpha_coef_lst, beta_coef_lst, gamma_noVI, just_no_VI=True)

        if tau != 0:
            tau_lst.append(tau)
            plt.close(fig)
            break
        # ---------------------------------------------------------

        contact_borders.append(t_global)

        # ---------------------------------------------------------
        # u_sum, v_sum = np.zeros(Nx), np.zeros(Nx)
        # for m0, alpha_i in enumerate(alpha_lst_init, start=1):
        #     u_sum += beta_lst_init[m0-1] * np.sin(np.pi * m0 * x_vals)
        #     v_sum += (-(np.pi * m0)**2 * gamma_ / 2 * beta_lst_init[m0-1] + omega_m_damping(m0) * alpha_i) * np.sin(np.pi * m0 * x_vals)

        number_VI += 1

        u_sum_L, u_sum_R = u_sum[:point_barrier + 1], u_sum[point_barrier:]
        v_sum_L, v_sum_R = v_sum[:point_barrier + 1], v_sum[point_barrier:]

        beta_coef_lst_L, alpha_coef_lst_draft_L = disp_vel_to_alpha_beta_lst(u_sum_L, v_sum_L, num_modes, string_length=a_bar, left_string=True)
        beta_coef_lst_R, alpha_coef_lst_draft_R = disp_vel_to_alpha_beta_lst(u_sum_R, v_sum_R, num_modes, string_length=(1.0 - a_bar), left_string=False)

        alpha_coef_lst_L = ((alpha_coef_lst_draft_L + (
                    np.pi * np.arange(1, num_modes + 1) / a_bar) ** 2 * gamma_VI / 2 * beta_coef_lst_L) / np.array(
            [omega_m_damping(ii + 1, gamma_VI, string_length=a_bar) for ii in range(num_modes)]))

        alpha_coef_lst_R = ((alpha_coef_lst_draft_R + (
                np.pi * np.arange(1, num_modes + 1) / (1-a_bar)) ** 2 * gamma_VI / 2 * beta_coef_lst_R) / np.array(
            [omega_m_damping(ii + 1, gamma_VI, string_length=(1-a_bar)) for ii in range(num_modes)]))

        u1_init_L, u1_init_R = np.zeros(Nx_L), np.zeros(Nx_R)
        for i in range(1, len(alpha_coef_lst_L) + 1):
            u1_init_L += beta_coef_lst_L[i - 1] * np.sin(np.pi * i / a_bar * x_vals_L)
            u1_init_R += beta_coef_lst_R[i - 1] * np.sin(np.pi * i / (1-a_bar) * x_vals_R)
        # plt.figure()
        # plt.plot(x_vals, np.concatenate((u1_init_L, u1_init_R[1:]), axis=0))
        # plt.grid()
        # plt.show()



        print('Time = ', str(t_global))
        # if num_loop == 20:
        #     just_VI = True
        # else:
        #     just_VI = False
        [disp_start, vel_start] = beam_with_VI_vibrations(alpha_coef_lst_L, alpha_coef_lst_R, beta_coef_lst_L, beta_coef_lst_R, gamma_VI, just_VI=just_VI)

        if tau != 0:
            tau_lst.append(tau)
            plt.close(fig)
            break

        # print(t_loc_lst)
        print(f'log = {disp_start[point_barrier]}')
        # ---------------------------------------------------------


        contact_borders.append(t_global)

end_time = time.time()
print(f"Execution time: {(end_time - start_time) / 60:.6f} min")
print(f'tau lst')
print(tau_lst)

with open(r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\26_01_2025_string_multyVI\log_tau.txt', 'w') as cur_file:
    cur_file.write(str(tau_lst))

plt.figure()
plt.plot(a_bar_lst, tau_lst)
plt.grid()
plt.show()




