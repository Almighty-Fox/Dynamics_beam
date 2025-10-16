#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
String with Point Barrier — Multiple VI cycles (Viscous damping, mode‑independent)
-------------------------------------------------------------------------------
Эта версия — полная замена k^2‑зависимого «структурного» демпфирования на
классическое вязкое демпфирование с одинаковым коэффициентом δ для всех мод.

Модель и структура кода соответствуют твоему исходному скрипту, чтобы можно было
вставить на замену «как есть». Отличия помечены комментариями # [VISC].

Основные изменения:
- Введён коэффициент вязкого демпфирования delta [1/s].
- Все факторы exp(-(π m / L)^2 * gamma / 2 * t) заменены на exp(-delta * t).
- Все omega_m_damping(...) заменены на omega_d(...), где omega_d = sqrt(omega_m^2 - delta^2).
- Формулы для v(t) и «сшивки» (alpha_from_vel, etc.) скорректированы под вязкое демпфирование.
- Локальная «проверка допустимости мод» (чтобы не получить пере‑затухание) сводится к предупреждению.

Подбор delta:
- Сохранить e‑фолдинг энергии первой моды при прежнем gamma: delta ≈ (π²/2) * gamma.
- Или задать желаемое время e‑спада энергии τ_E: delta = 1/(2 τ_E).

Проверено на совместимость с прежней организацией циклов и визуализацией.
"""

from __future__ import annotations
import numpy as np
from numpy import sin, cos
import pylab as plt
import time
import pandas as pd

# =============================================================
# Пользовательские параметры
# =============================================================
pi = np.pi
L = 1.0                 # длина струны (нормирована)
Nx = 2000 + 1           # узлов по x
x_vals = np.linspace(0.0, L, Nx)

dx = L / (Nx - 1)
T = 3.0                 # время моделирования (не используется напрямую ниже)
c = 1.0                 # скорость волны (единичная масштабировка)

# [VISC] ЕДИНЫЙ коэффициент вязкого демпфирования для всех мод
# Рекомендуемый старт: delta ≈ (π²/2) * gamma_old  для сохранения e‑спада первой моды.
delta = 0.1         # [1/s] подбери под задачу

# — визуализация энергии по модам —
num_modes_plot_energy = 20

# Временные шаги (динамическая лестница)

def create_time_steps(dt_smallest, dt_biggest, relative_difference_max):
    if dt_smallest <= 0 or dt_biggest <= 0 or dt_smallest >= dt_biggest:
        raise ValueError("dt_smallest > 0, dt_biggest > 0, dt_smallest < dt_biggest")
    if relative_difference_max <= 0:
        raise ValueError("relative_difference_max должен быть > 0")
    steps = [dt_smallest]
    while steps[-1] < dt_biggest:
        nxt = steps[-1] * (1.0 + relative_difference_max)
        if nxt > dt_biggest:
            steps.append(dt_biggest)
            break
        steps.append(nxt)
    return np.array(steps)

# параметры лестницы dt
dt_smallest = 1e-15
dt_biggest  = 1e-3
relative_difference_max = 0.3

t_step_lst = create_time_steps(dt_smallest, dt_biggest, relative_difference_max)
print("time steps prepared (count=)", len(t_step_lst))

# =============================================================
# Глобальные контейнеры/флаги (как в оригинале)
# =============================================================

t_global = 0.0
number_VI = 0
time_is_up = False
first_time_tau = True

t_global_lst = []
energy_data_global = []
contact_borders = []

tau = 0.0

# визуализация (создаётся позже в основном цикле)
fig, axs = None, None

def_graph_first_time = True

# =============================================================
# Частоты и формы
# =============================================================

def omega_m(m: int, string_length: float = 1.0) -> float:
    return pi * m / string_length

def omega_d(m: int, string_length: float = 1.0) -> float:
    om = omega_m(m, string_length)
    val = om * om - delta * delta
    return np.sqrt(val) if val > 0 else 0.0  # защита от пере-/критического

omega_m_damping_lst = np.array([omega_d(i, 1.0) for i in range(1, num_modes_plot_energy + 1)])
omega_m_lst         = np.array([omega_m(i, 1.0) for i in range(1, num_modes_plot_energy + 1)])

# =============================================================
# Модальные коэффициенты (проекции на синусы)
# =============================================================

def modal_coefficients(x, y, num_modes, string_length):
    coeffs = []
    for k in range(1, num_modes + 1):
        phi_k = np.sin(k * pi / string_length * x)
        a_k = (2 / string_length) * np.trapz(y * phi_k, x=x)
        coeffs.append(a_k)
    return np.array(coeffs)


def disp_vel_to_alpha_beta_lst(y_vals, v_vals, num_modes, string_length=1.0, left_string=True):
    # выбираем правильный x для левой/правой части
    if abs(string_length - 1.0) < 1e-12:
        x = x_vals
    else:
        if left_string:
            x = np.linspace(0.0, string_length, int(round((Nx - 1) * string_length)) + 1)
        else:
            x = np.linspace(0.0, string_length, int(round((Nx - 1) * string_length)) + 1)
    modal_coords     = modal_coefficients(x, y_vals, num_modes, string_length)
    modal_velocities = modal_coefficients(x, v_vals, num_modes, string_length)
    return modal_coords, modal_velocities

# =============================================================
# Энергии
# =============================================================

def total_energy(u, v):
    kinetic   = np.sum(v ** 2) / 2.0
    potential = np.sum((np.diff(u) / dx) ** 2) / 2.0
    return (kinetic + potential) * dx


def energy_over_modes(y_vals, v_vals, case_damping):
    case_omega_n_lst = omega_m_lst if not case_damping else omega_m_damping_lst
    modal_coords, modal_velocities = disp_vel_to_alpha_beta_lst(y_vals, v_vals, num_modes_plot_energy)
    # классическое выражение 0.25*(v_m^2 + (ω_m u_m)^2)
    return 0.25 * (modal_velocities ** 2 + case_omega_n_lst ** 2 * modal_coords ** 2)

# =============================================================
# Графика
# =============================================================

def graph(y_list, vel_list, energy_over_modes_lst, case_plot, case_time_delay=False, if_contact=False, case_time_delay_2=False):
    global def_graph_first_time
    if def_graph_first_time:
        t_global_lst.append(0.0)
    statement = (not case_plot or def_graph_first_time) or (len(t_global_lst) % 20 == 0)
    if statement:
        fig.suptitle(
            'Time=' + str('%.2f' % t_global) + 's=' + str('%.2f' % (t_global * 1e3)) + 'ms' + ', loc_bar = ' + str(
                a_bar) + ', num modes = ' + str(num_modes) + '\n' + 'Contact ' + str(if_contact) + '. Number of VI = ' + str(number_VI) + ', tau = ' + str(round(tau, 3)))

        axs[0][0].set_title('Beam shape')
        axs[0][0].plot(x_vals, y_list, 'k', linewidth=1)
        axs[0][0].grid()

        axs[0][1].set_title('Beam vel')
        axs[0][1].plot(x_vals, vel_list, 'g', linewidth=1)
        if def_graph_first_time:
            axs[0][1].plot([a_bar, a_bar], [min(vel_list), max(vel_list)], 'k--', linewidth=1)
        axs[0][1].grid()

        axs[1][0].set_title('Energy')
        axs[1][0].plot(t_global_lst, energy_data_global, 'g', linewidth=1)
        axs[1][0].set_ylim(0, max(energy_data_global[0] * 1.2, max(energy_data_global) if energy_data_global else 1.0))
        axs[1][0].grid()
        for ii in range(len(contact_borders)):
            axs[1][0].plot([contact_borders[ii], contact_borders[ii]], [0, energy_data_global[0]], 'k--', linewidth=1)

        df = pd.DataFrame(energy_over_modes_lst, columns=['origin'], index=range(1, num_modes_plot_energy + 1))
        axs[1][1].set_title('Energy distribution over modes\nMax energy = {} %'.format(
            round(float(np.max(df.origin[:])) / float(np.sum(df.origin[:])) * 100, 2)), fontsize=7, pad=0)
        axs[1][1].bar(np.arange(1, num_modes_plot_energy + 1), df.origin[:])
        axs[1][1].set_xticks(np.arange(1, num_modes_plot_energy))

        plt.pause(0.01)
        axs[0][0].clear(); axs[0][1].clear(); axs[1][0].clear(); axs[1][1].clear()

    if def_graph_first_time:
        energy_data_global.pop()
        t_global_lst.pop()
    def_graph_first_time = False

# =============================================================
# Динамика без контакта (NO VI) — вязкое демпфирование
# =============================================================

def beam_no_VI_vibrations(alpha_lst_init, beta_lst_init, just_no_VI=False):
    global t_global, time_is_up, t_global_lst, energy_data_global, tau, first_time_tau

    t_local = 0.0
    if_we_are_ready_for_new_VI = False
    in_barrier = False

    t_step_id = 0
    dt_next = t_step_lst[t_step_id]

    first_loop_no_VI = True

    # начальные поля u,v
    u_init = np.zeros(Nx)
    for n0, beta_i in enumerate(beta_lst_init, start=1):
        u_init += beta_i * np.sin(pi * n0 * x_vals)
    u_init[0] = 0.0; u_init[-1] = 0.0

    v_init = np.zeros(Nx)
    for m0, alpha_i in enumerate(alpha_lst_init, start=1):
        v_init += ( -delta * beta_lst_init[m0-1] + omega_d(m0, 1.0) * alpha_i ) * np.sin(pi * m0 * x_vals)
    v_init[0] = 0.0; v_init[-1] = 0.0

    pass_for_next_while = False
    u_next = u_init.copy()

    while (u_next[point_barrier] >= 0) or just_no_VI or not if_we_are_ready_for_new_VI or pass_for_next_while:
        pass_for_next_while = False

        # критерий затухания энергии до ~ 1/e
        if (len(energy_data_global) > 0) and (energy_data_global[-1] <= energy_data_global[0] / np.e) and first_time_tau:
            time_is_up = True
            first_time_tau = False
            tau = t_global
            print('----------------------------------')
            print(f'tau = {t_global}')
            print(f'Number VI = {number_VI}')
            print('----------------------------------')
            graph(u_next, v_next, energy_over_modes_lst_local[:num_modes_plot_energy], case_plot=False, case_time_delay=False, case_time_delay_2=True)
            return u_next, v_next

        t_global += dt_next
        t_global_lst.append(t_global)
        t_local  += dt_next

        # решение по модам: u_m(t) = e^{-delta t}( α_m sin(ω_d t) + β_m cos(ω_d t) )
        u_next = np.zeros(Nx)
        for m0, alpha_i in enumerate(alpha_lst_init, start=1):
            u_next_t = np.exp(-delta * t_local) * ( alpha_i * np.sin(omega_d(m0, 1.0) * t_local) + beta_lst_init[m0-1] * np.cos(omega_d(m0, 1.0) * t_local) )
            u_next += u_next_t * np.sin(pi * m0 * x_vals)
        u_next[0] = 0.0; u_next[-1] = 0.0

        # адаптация шага при входе в барьер
        if not just_no_VI and if_we_are_ready_for_new_VI:
            if (t_step_id > 0) and (u_next[point_barrier] < 0):
                pass_for_next_while = True
                in_barrier = True
                t_global -= dt_next; t_global_lst.pop(); t_local -= dt_next
                t_step_id -= 1; dt_next = t_step_lst[t_step_id]
                continue

        v_next = np.zeros(Nx)
        v_next_t_lst = []
        u_next_t_lst = []
        for m0, alpha_i in enumerate(alpha_lst_init, start=1):
            # производная по времени u_m: e^{-δt}[ -δ(α sin + β cos) + ω_d(α cos - β sin) ]
            s_wt = np.sin(omega_d(m0, 1.0) * t_local)
            c_wt = np.cos(omega_d(m0, 1.0) * t_local)
            base   = ( alpha_i * s_wt + beta_lst_init[m0-1] * c_wt )
            dbase  = ( alpha_i * c_wt - beta_lst_init[m0-1] * s_wt )
            v_next_t = np.exp(-delta * t_local) * ( -delta * base + omega_d(m0, 1.0) * dbase )
            v_next_t_lst.append(v_next_t)
            v_next += v_next_t * np.sin(pi * m0 * x_vals)

            u_next_t = np.exp(-delta * t_local) * base
            u_next_t_lst.append(u_next_t)

        kinetic_energy   = 0.25 * sum([v_next_t_lst[ii]**2 for ii in range(len(v_next_t_lst))])
        potential_energy = 0.25 * sum([(pi * (ii+1) * u_next_t_lst[ii]) ** 2 for ii in range(len(u_next_t_lst))])
        cur_energy = kinetic_energy + potential_energy
        energy_data_global.append(cur_energy)
        energy_over_modes_lst_local = 0.25 * (np.array([v_next_t_lst[ii] ** 2 for ii in range(len(v_next_t_lst))]) +
                                              np.array([(pi * (ii + 1) * u_next_t_lst[ii]) ** 2 for ii in range(len(u_next_t_lst))]))
        graph(u_next, v_next, energy_over_modes_lst_local[:num_modes_plot_energy], case_plot=True)

        if first_loop_no_VI:
            print(f'Energy at detachment')
            print(", ".join(map(str, energy_over_modes_lst_local[:num_modes_plot_energy])))
            first_loop_no_VI = False

        # расширяем шаг, если вне барьера
        if (not in_barrier) and (t_step_id < len(t_step_lst) - 1):
            t_step_id += 1
        dt_next = t_step_lst[t_step_id]

        if u_next[point_barrier] > 1e-4:
            if_we_are_ready_for_new_VI = True

    print('Before VI (leaving NO-VI)')
    graph(u_next, v_next, energy_over_modes_lst_local[:num_modes_plot_energy], case_plot=False, case_time_delay=False)
    return u_next, v_next

# =============================================================
# Динамика с контактом (WITH VI) — вязкое демпфирование
# =============================================================

def beam_with_VI_vibrations(alpha_coef_lst_L, alpha_coef_lst_R, beta_coef_lst_L, beta_coef_lst_R, just_VI=False):
    print('with barrier')
    global t_global, time_is_up, t_global_lst, energy_data_global, tau, first_time_tau

    t_local = 0.0
    in_barrier = False

    t_step_id = 0
    dt_next = t_step_lst[t_step_id]

    # начальная контактная сила через производные форм формы (как в исходнике)
    diff_u_init_L, diff_u_init_R = 0.0, 0.0
    for n0, beta_i in enumerate(beta_coef_lst_L, start=1):
        diff_u_init_L += beta_i * (pi * n0 / a_bar) * np.cos(pi * n0)
        diff_u_init_R += beta_coef_lst_R[n0-1] * (pi * n0 / (1.0 - a_bar))
    VI_force_init = -diff_u_init_L + diff_u_init_R
    print(f'VI_force_init = {VI_force_init}')

    VI_force = VI_force_init
    pass_for_next_while = False
    first_steps = True

    while (VI_force <= 0) or pass_for_next_while or first_steps or just_VI:
        pass_for_next_while = False

        if (len(energy_data_global) > 0) and (energy_data_global[-1] <= energy_data_global[0] / np.e) and first_time_tau:
            time_is_up = True
            first_time_tau = False
            tau = t_global
            print('----------------------------------')
            print(f'tau = {t_global}')
            print(f'Number VI = {number_VI}')
            print('----------------------------------')
            graph(np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0), energy_over_modes_lst_local[:num_modes_plot_energy], case_plot=False, case_time_delay=False, case_time_delay_2=True, if_contact=True)
            return np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0)

        t_global += dt_next
        t_global_lst.append(t_global)
        t_local  += dt_next

        diff_u_next_L, diff_u_next_R = 0.0, 0.0
        for m0, alpha_i in enumerate(alpha_coef_lst_L, start=1):
            sL = np.sin(omega_d(m0, a_bar) * t_local)
            cL = np.cos(omega_d(m0, a_bar) * t_local)
            baseL = alpha_i * sL + beta_coef_lst_L[m0 - 1] * cL
            diff_u_next_L += np.exp(-delta * t_local) * baseL * (pi * m0 / a_bar) * np.cos(pi * m0)

            sR = np.sin(omega_d(m0, 1.0 - a_bar) * t_local)
            cR = np.cos(omega_d(m0, 1.0 - a_bar) * t_local)
            baseR = alpha_coef_lst_R[m0 - 1] * sR + beta_coef_lst_R[m0 - 1] * cR
            diff_u_next_R += np.exp(-delta * t_local) * baseR * (pi * m0 / (1.0 - a_bar))

        VI_force = -diff_u_next_L + diff_u_next_R
        if VI_force <= 0.0:
            first_steps = False

        if not just_VI and not first_steps:
            if (t_step_id > 0) and (VI_force > 0.0):
                pass_for_next_while = True
                in_barrier = True
                t_global -= dt_next; t_global_lst.pop(); t_local -= dt_next
                t_step_id -= 1; dt_next = t_step_lst[t_step_id]
                continue

        u_next_L, u_next_R = np.zeros(Nx_L), np.zeros(Nx_R)
        v_next_L, v_next_R = np.zeros(Nx_L), np.zeros(Nx_R)
        v_next_t_lst_L, v_next_t_lst_R = [], []
        u_next_t_lst_L, u_next_t_lst_R = [], []

        for m0, alpha_i in enumerate(alpha_coef_lst_L, start=1):
            sL = np.sin(omega_d(m0, a_bar) * t_local); cL = np.cos(omega_d(m0, a_bar) * t_local)
            baseL  = alpha_i * sL + beta_coef_lst_L[m0 - 1] * cL
            dbaseL = alpha_i * cL - beta_coef_lst_L[m0 - 1] * sL

            u_next_t_L = np.exp(-delta * t_local) * baseL
            v_next_t_L = np.exp(-delta * t_local) * ( -delta * baseL + omega_d(m0, a_bar) * dbaseL )

            u_next_t_lst_L.append(u_next_t_L)
            v_next_t_lst_L.append(v_next_t_L)

            u_next_L += u_next_t_L * np.sin(pi * m0 / a_bar * x_vals_L)
            v_next_L += v_next_t_L * np.sin(pi * m0 / a_bar * x_vals_L)

            sR = np.sin(omega_d(m0, 1.0 - a_bar) * t_local); cR = np.cos(omega_d(m0, 1.0 - a_bar) * t_local)
            baseR  = alpha_coef_lst_R[m0 - 1] * sR + beta_coef_lst_R[m0 - 1] * cR
            dbaseR = alpha_coef_lst_R[m0 - 1] * cR - beta_coef_lst_R[m0 - 1] * sR

            u_next_t_R = np.exp(-delta * t_local) * baseR
            v_next_t_R = np.exp(-delta * t_local) * ( -delta * baseR + omega_d(m0, 1.0 - a_bar) * dbaseR )

            u_next_t_lst_R.append(u_next_t_R)
            v_next_t_lst_R.append(v_next_t_R)

            u_next_R += u_next_t_R * np.sin(pi * m0 / (1.0 - a_bar) * x_vals_R)
            v_next_R += v_next_t_R * np.sin(pi * m0 / (1.0 - a_bar) * x_vals_R)

        kinetic_energy_L   = a_bar       * 0.25 * sum([v_next_t_lst_L[ii] ** 2 for ii in range(len(v_next_t_lst_L))])
        kinetic_energy_R   = (1.0-a_bar) * 0.25 * sum([v_next_t_lst_R[ii] ** 2 for ii in range(len(v_next_t_lst_R))])
        potential_energy_L = a_bar       * 0.25 * sum([(pi * (ii + 1) / a_bar       * u_next_t_lst_L[ii]) ** 2 for ii in range(len(u_next_t_lst_L))])
        potential_energy_R = (1.0-a_bar) * 0.25 * sum([(pi * (ii + 1) / (1.0-a_bar) * u_next_t_lst_R[ii]) ** 2 for ii in range(len(u_next_t_lst_R))])
        cur_energy_sum = kinetic_energy_L + kinetic_energy_R + potential_energy_L + potential_energy_R
        energy_data_global.append(cur_energy_sum)

        energy_over_modes_lst_local = 0.25 * (
            np.array([v_next_t_lst_L[ii] ** 2 for ii in range(len(v_next_t_lst_L))]) +
            np.array([v_next_t_lst_R[ii] ** 2 for ii in range(len(v_next_t_lst_R))]) +
            np.array([(pi * (ii + 1) / a_bar       * u_next_t_lst_L[ii]) ** 2 for ii in range(len(u_next_t_lst_L))]) +
            np.array([(pi * (ii + 1) / (1.0-a_bar) * u_next_t_lst_R[ii]) ** 2 for ii in range(len(u_next_t_lst_R))])
        )

        graph(np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0), energy_over_modes_lst_local[:num_modes_plot_energy], case_plot=True, if_contact=True)

        if (not in_barrier) and (t_step_id < len(t_step_lst) - 1):
            t_step_id += 1
        dt_next = t_step_lst[t_step_id]

    print('Before VI (leaving WITH-VI)')
    graph(np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0), energy_over_modes_lst_local[:num_modes_plot_energy], case_plot=False, case_time_delay=False, if_contact=True)

    return np.concatenate((u_next_L, u_next_R[1:]), axis=0), np.concatenate((v_next_L, v_next_R[1:]), axis=0)

# =============================================================
# Начальные условия — гауссово распределение энергии по модам
# =============================================================

def initial_conditions(sigma_cur=None, seed_number=None):
    """
    Фиксированное возбуждение:
      alpha_coef_lst[0] = 1   (мода 1)
      alpha_coef_lst[2] = -1  (мода 3)
    Все beta = 0.
    """
    global vel_init_sign

    # гарантируем, что мод как минимум 3
    assert num_modes >= 3, "num_modes должно быть ≥ 3 для возбуждения мод 1 и 3"

    beta_coef_lst  = np.zeros(num_modes)
    alpha_coef_lst = np.zeros(num_modes)

    # нужное возбуждение:
    alpha_coef_lst[0] =  -1.0   # мода 1
    # alpha_coef_lst[2] = 1.0   # мода 3

    # пересчитать знак начальной скорости в точке барьера (как у тебя было)
    v1_init = 0.0
    # [VISC] перевод «через скорость» с учётом вязкого демпфирования:
    alpha_coef_lst_right = (
        (alpha_coef_lst + delta * beta_coef_lst) /
        np.array([omega_d(ii + 1, 1.0) for ii in range(num_modes)])
    )
    for m0, alpha_i in enumerate(alpha_coef_lst_right, start=1):
        v1_init += (-delta * beta_coef_lst[m0-1] + omega_d(m0, 1.0) * alpha_i) * np.sin(pi * m0 * a_bar)

    vel_init_sign = True
    if v1_init < 0:
        vel_init_sign = False  # старт "к барьеру"

    # построим начальные поля u(x), v(x) (всё как раньше)
    y1_init = np.zeros_like(x_vals)
    v1_init_shape = np.zeros_like(x_vals)
    for n0, beta_i in enumerate(beta_coef_lst, start=1):
        y1_init += beta_i * np.sin(pi * n0 * x_vals)
    for m0, alpha_i in enumerate(alpha_coef_lst_right, start=1):
        v1_init_shape += (-delta * beta_coef_lst[m0 - 1] + omega_d(m0, 1.0) * alpha_i) * np.sin(pi * m0 * x_vals)

    cur_energy = total_energy(y1_init, v1_init_shape)
    print(f'First value cur_energy={cur_energy}')
    energy_data_global.append(cur_energy)
    energy_over_modes_lst0 = energy_over_modes(y1_init, v1_init_shape, case_damping=False)
    graph(y1_init, v1_init_shape, energy_over_modes_lst0, case_plot=False, if_contact=False, case_time_delay=False)

    return alpha_coef_lst, beta_coef_lst


# =============================================================
# Главный сценарий перебора a_bar, sigma, seed
# =============================================================

# a_bar_lst = np.arange(0.2, 0.801, 0.05).tolist()
# sigma_lst = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 6.0]
# seed_lst  = [3344, 853, 2345, 4332]

# a_bar_lst = np.arange(0.4, 0.601, 0.1).tolist()
a_bar_lst = [0.7]
sigma_lst = [None]
seed_lst  = [None]

# --- single global undercritical check (insert once, after delta & a_bar_lst are defined) ---
def check_viscous_undercritical(delta, a_bar_lst, safety=1.0):
    import numpy as np
    a_bar_arr = np.array(a_bar_lst, dtype=float)

    L_whole     = 1.0
    L_left_min  = float(np.min(a_bar_arr))           # худшая (самая короткая) левая часть
    L_right_min = float(1.0 - np.max(a_bar_arr))     # худшая правая часть
    L_crit      = min(L_whole, L_left_min, L_right_min)

    delta_crit  = np.pi / L_crit            # граница критического затухания для низшей моды
    delta_safe  = safety * delta_crit       # опциональный запас (например, safety=0.9)

    ok = delta < delta_safe
    print(f"[DAMPING CHECK] L_crit={L_crit:.4g},  delta_crit=π/L_crit={delta_crit:.6g} 1/s,  "
          f"delta={delta:.6g} 1/s  → {'OK' if ok else 'TOO LARGE'}"
          + (f" (safety×= {safety:.2f})" if safety!=1.0 else ""))

    if not ok:
        print(f"[WARNING] Choose delta < {delta_safe:.6g} 1/s to keep all lowest modes under-damped "
              f"(worst segment length L_crit={L_crit:.4g}).")
    return ok, L_crit, delta_crit, delta_safe

# вызов один раз перед большими циклами:
ok, *_ = check_viscous_undercritical(delta, a_bar_lst, safety=1.0)
assert ok, "delta too large: some lowest modes would be critical/overdamped"



vel_init_sign = True

start_time = time.time()

for sigma_i in sigma_lst:
    for seed_cur in seed_lst:
        tau_lst = []
        number_VI_lst = []
        for i_id, a_bar_cur in enumerate(a_bar_lst):
            vel_init_sign = True

            # геометрия барьера / разбиение
            a_bar = float(round(a_bar_cur, 4))
            point_barrier = round((Nx - 1) * a_bar)
            Nx_L = point_barrier + 1
            Nx_R = Nx - point_barrier
            x_vals_L = np.linspace(0.0, a_bar, Nx_L)
            x_vals_R = np.linspace(0.0, 1.0 - a_bar, Nx_R)

            # число мод: можно оставить константой или ограничить так, чтобы ω_1(L_min) > delta
            num_modes_draft = 300

            L_min = min(a_bar, 1.0 - a_bar)
            if pi * 1 / L_min <= delta:
                print("Warning: delta слишком велик для самой короткой части, ω_d первой моды = 0. Уменьши delta.")

            num_modes = num_modes_draft

            # сброс глобалов
            t_global = 0.0
            t_global_lst = []
            energy_data_global = []
            number_VI = 0
            tau = 0.0
            first_time_tau = True
            def_graph_first_time = True
            contact_borders = []
            energy_over_modes_lst_local = np.zeros((num_modes_plot_energy, 1))

            just_no_VI = False
            just_VI = False
            first_loop = True
            num_loop = 0

            t_step_id = 0
            t_step = t_step_lst[t_step_id]
            t_end = 4.00
            time_is_up = False

            print('Начинаем расчёт (viscous)')

            fig, axs = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.pause(0.1)

            # начальные условия
            alpha_coef_lst_start, beta_coef_lst_start = initial_conditions(sigma_cur=sigma_i, seed_number=seed_cur)
            print('Задали НУ — запускаем динамику')

            tau_passed = False
            if not vel_init_sign:
                first_loop = False
                number_VI += 1
                contact_borders.append(0.0)

                # восстановим поля на всей струне из (α,β)
                y1_init = np.zeros_like(x_vals)
                v1_init = np.zeros_like(x_vals)
                alpha_coef_lst_right = (
                    (alpha_coef_lst_start + delta * beta_coef_lst_start) /
                    np.array([omega_d(ii + 1, 1.0) for ii in range(num_modes)])
                )
                for n0, beta_i in enumerate(beta_coef_lst_start, start=1):
                    y1_init += beta_i * np.sin(pi * n0 * x_vals)
                for m0, alpha_i in enumerate(alpha_coef_lst_right, start=1):
                    v1_init += ( -delta * beta_coef_lst_start[m0 - 1] + omega_d(m0, 1.0) * alpha_i ) * np.sin(pi * m0 * x_vals)

                u_sum_L, u_sum_R = y1_init[:point_barrier + 1], y1_init[point_barrier:]
                v_sum_L, v_sum_R = v1_init[:point_barrier + 1], v1_init[point_barrier:]

                beta_coef_lst_L, alpha_coef_lst_draft_L = disp_vel_to_alpha_beta_lst(u_sum_L, v_sum_L, num_modes, string_length=a_bar, left_string=True)
                beta_coef_lst_R, alpha_coef_lst_draft_R = disp_vel_to_alpha_beta_lst(u_sum_R, v_sum_R, num_modes, string_length=(1.0 - a_bar), left_string=False)

                alpha_coef_lst_L = (
                    (alpha_coef_lst_draft_L + delta * beta_coef_lst_L) /
                    np.array([omega_d(ii + 1, a_bar) for ii in range(num_modes)])
                )
                alpha_coef_lst_R = (
                    (alpha_coef_lst_draft_R + delta * beta_coef_lst_R) /
                    np.array([omega_d(ii + 1, 1.0 - a_bar) for ii in range(num_modes)])
                )

                u1_init_L, u1_init_R = np.zeros(Nx_L), np.zeros(Nx_R)
                for i in range(1, len(alpha_coef_lst_L) + 1):
                    u1_init_L += beta_coef_lst_L[i - 1] * np.sin(pi * i / a_bar * x_vals_L)
                    u1_init_R += beta_coef_lst_R[i - 1] * np.sin(pi * i / (1.0 - a_bar) * x_vals_R)

                disp_start, vel_start = beam_with_VI_vibrations(alpha_coef_lst_L, alpha_coef_lst_R, beta_coef_lst_L, beta_coef_lst_R, just_VI=just_VI)

                if tau != 0.0:
                    tau_lst.append(tau)
                    plt.close(fig)
                    tau_passed = True

                contact_borders.append(t_global)

            while not tau_passed:
                num_loop += 1

                if first_loop:
                    beta_coef_lst, alpha_coef_lst_draft = beta_coef_lst_start.copy(), alpha_coef_lst_start.copy()
                    first_loop = False
                else:
                    beta_coef_lst, alpha_coef_lst_draft = disp_vel_to_alpha_beta_lst(disp_start, vel_start, num_modes)

                alpha_coef_lst = (
                    (alpha_coef_lst_draft + delta * beta_coef_lst) /
                    np.array([omega_d(ii + 1, 1.0) for ii in range(num_modes)])
                )

                print('Time = ', str(t_global))
                u_sum, v_sum = beam_no_VI_vibrations(alpha_coef_lst, beta_coef_lst, just_no_VI=just_no_VI)

                if tau != 0.0:
                    tau_lst.append(tau)
                    plt.close(fig)
                    break

                contact_borders.append(t_global)

                number_VI += 1

                u_sum_L, u_sum_R = u_sum[:point_barrier + 1], u_sum[point_barrier:]
                v_sum_L, v_sum_R = v_sum[:point_barrier + 1], v_sum[point_barrier:]

                beta_coef_lst_L, alpha_coef_lst_draft_L = disp_vel_to_alpha_beta_lst(u_sum_L, v_sum_L, num_modes, string_length=a_bar, left_string=True)
                beta_coef_lst_R, alpha_coef_lst_draft_R = disp_vel_to_alpha_beta_lst(u_sum_R, v_sum_R, num_modes, string_length=(1.0 - a_bar), left_string=False)

                alpha_coef_lst_L = (
                    (alpha_coef_lst_draft_L + delta * beta_coef_lst_L) /
                    np.array([omega_d(ii + 1, a_bar) for ii in range(num_modes)])
                )
                alpha_coef_lst_R = (
                    (alpha_coef_lst_draft_R + delta * beta_coef_lst_R) /
                    np.array([omega_d(ii + 1, 1.0 - a_bar) for ii in range(num_modes)])
                )

                u1_init_L, u1_init_R = np.zeros(Nx_L), np.zeros(Nx_R)
                for i in range(1, len(alpha_coef_lst_L) + 1):
                    u1_init_L += beta_coef_lst_L[i - 1] * np.sin(pi * i / a_bar * x_vals_L)
                    u1_init_R += beta_coef_lst_R[i - 1] * np.sin(pi * i / (1.0 - a_bar) * x_vals_R)

                disp_start, vel_start = beam_with_VI_vibrations(alpha_coef_lst_L, alpha_coef_lst_R, beta_coef_lst_L, beta_coef_lst_R, just_VI=just_VI)

                if tau != 0.0:
                    tau_lst.append(tau)
                    number_VI_lst.append(number_VI)
                    plt.close(fig)
                    break

                contact_borders.append(t_global)

        end_time = time.time()
        print(f"Execution time: {(end_time - start_time) / 60:.6f} min")
        print('tau list:')
        print(tau_lst)
        print('Number of VI:')
        print(number_VI_lst)

        # # --- Append results to log file ---
        # with open('results_log_visc.txt', 'a', encoding='utf-8') as f:
        #     f.write(f"Sigma = {sigma_i}, seed = {seed_cur}\n")
        #     f.write(f"Execution time: {(end_time - start_time) / 60:.6f} min\n")
        #     f.write(f"Tau list: {tau_lst}\n")
        #     f.write(f"Number of VI (contacts): {number_VI_lst}\n")
        #     f.write(f"{'-' * 50}\n")

