#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snap-through energy vs α for a two-mass arch.

α = |E1-E2| / (E1+E2),           0 ≤ α ≤ 1,
E1, E2 – начальные кинетические энергии масс.

Алгоритм
--------
1. Находим верхний и нижний устойчивые эквилибрии (уже есть в исходном коде).
2. Для заданного α формируем скорости
       v1² = K (1+α),   v2² = K (1-α),   K = нач. кинетическая энергия.
   Знак берём отрицательным (оба «вниз»), чтобы двигаться к нижней яме.
3. Интегрируем до T_final (тот же solve_ivp).
4. Условие успеха: к T_final система
       • почти покоится (|v| < VEL_TOL)   и
       • ближе к нижнему эквилибрию, чем к верхнему.
5. Подбираем K: doubling-search, затем binary-search (n_iter=10 по умолчанию).
6. Строим график E*(α)=V_up+K*(α).
"""
import numpy as np
from numpy import sqrt, hypot
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# ──────────────────────────────────────────────────────────────
# 0. Ваш исходный блок параметров и функций (Geometry, EoM, …)
#    *** без изменений, только убрали визуализации ***
# ──────────────────────────────────────────────────────────────
a1 = 0.5
a2 = 0.5
a3 = 0.5
l01 = 1.5
l02 = 0.5
l03 = 1.5
k1 = 3.0e7
k2 = 3.0e7
k3 = 3.0e7
k_theta = 2.0e3
m = 1.0

RTOL = 1e-7
ATOL = 1e-10
max_step_solver = 1e-5
T_final = 0.015
VEL_TOL = 1e-2  # «покоится»


# ---------- геометрия --------------------------------------------------
def geometry(y1, y2):
    dL1 = sqrt(a1 ** 2 + y1 ** 2) - l01
    dL2 = sqrt(a2 ** 2 + (y2 - y1) ** 2) - l02
    dL3 = sqrt(a3 ** 2 + y2 ** 2) - l03
    th1 = np.arctan2(y1, a1)
    th2 = np.arctan2(y2, a3)
    # производные
    dL1_dy1 = y1 / (sqrt(a1 ** 2 + y1 ** 2))
    dL2_dy1 = (y1 - y2) / (sqrt(a2 ** 2 + (y2 - y1) ** 2))
    dth1_dy1 = a1 / (a1 ** 2 + y1 ** 2)
    dL3_dy2 = y2 / (sqrt(a3 ** 2 + y2 ** 2))
    dL2_dy2 = (y2 - y1) / (sqrt(a2 ** 2 + (y2 - y1) ** 2))
    dth2_dy2 = a3 / (a3 ** 2 + y2 ** 2)
    return (dL1, dL2, dL3, th1, th2,
            dL1_dy1, dL2_dy1, dth1_dy1,
            dL3_dy2, dL2_dy2, dth2_dy2)


# ---------- потенциальная энергия --------------------------------------
def V(y1, y2):
    dL1, dL2, dL3, th1, th2, *_ = geometry(y1, y2)
    return 0.5 * (k1 * dL1 ** 2 + k2 * dL2 ** 2 + k3 * dL3 ** 2 + k_theta * (th1 ** 2 + th2 ** 2))


# ---------- статика (эквилибриумы) -------------------------------------
def static_residuals(Y):
    y1, y2 = Y
    (dL1, dL2, dL3, th1, th2,
     dL1_dy1, dL2_dy1, dth1_dy1,
     dL3_dy2, dL2_dy2, dth2_dy2) = geometry(y1, y2)
    Q1 = k1 * dL1 * dL1_dy1 + k2 * dL2 * dL2_dy1 + k_theta * th1 * dth1_dy1
    Q2 = k3 * dL3 * dL3_dy2 + k2 * dL2 * dL2_dy2 + k_theta * th2 * dth2_dy2
    return Q1, Q2


y_eq_up = fsolve(static_residuals, (1.1, 1.1))
y_eq_down = fsolve(static_residuals, (-1.1, -1.1))
V_up = V(*y_eq_up)
V_down = V(*y_eq_down)
V_origin = V(0, 0)

# print(V(0, 0))


# ---------- уравнения движения -----------------------------------------
c = 10.0  # демпфирование
def eom(t, y):
    y1, y2, v1, v2 = y
    (dL1, dL2, dL3, th1, th2,
     dL1_dy1, dL2_dy1, dth1_dy1,
     dL3_dy2, dL2_dy2, dth2_dy2) = geometry(y1, y2)
    # обобщённые силы
    Q1 = (k1 * dL1 * dL1_dy1 + k2 * dL2 * dL2_dy1 + k_theta * th1 * dth1_dy1)
    Q2 = (k3 * dL3 * dL3_dy2 + k2 * dL2 * dL2_dy2 + k_theta * th2 * dth2_dy2)
    a1_dd = (-Q1 - c * v1) / m
    a2_dd = (-Q2 - c * v2) / m
    return v1, v2, a1_dd, a2_dd


# ---------- событие: обе координаты пересекли ноль вниз ------------
def both_negative_event(t, y):
    # y = [y1, y2, v1, v2]
    # если все четыре величины < 0, то max из них станет < 0
    return max(y[0], y[1], y[2], y[3])


both_negative_event.terminal = True
both_negative_event.direction = -1

# ──────────────────────────────────────────────────────────────
# 1. Один прогон динамики – «достаточно ли энергии?»
# ──────────────────────────────────────────────────────────────
def snapped_through(K, alpha):
    """True, если при начальной кинетической энергии K и данном α
    система в конце интегрирования осела в нижнем устойчивом положении."""
    # начальные скорости (оба «вниз»)
    v1 = -sqrt(K * (1 + alpha))
    v2 = -sqrt(K * (1 - alpha))
    y0 = np.array([*y_eq_up, v1, v2])  # старт из верхней ямы
    sol = solve_ivp(eom, (0, T_final), y0, events=both_negative_event,
                    max_step=max_step_solver,
                    rtol=RTOL, atol=ATOL, dense_output=False)

    # 3) После интегрирования можно проверить, было ли событие:
    # если событие произошло:
    if sol.t_events[0].size > 0:
        print(sol.t_events[0].size)
        print(sol.t_events[0][0])
        y1_fin, y2_fin, v1_fin, v2_fin = sol.y_events[0][0]
        return True
    else:
        # дотолкали до T_final
        print('дотолкали')
        y1_fin, y2_fin, v1_fin, v2_fin = sol.y[:, -1]
        return False

    # y1_fin, y2_fin, v1_fin, v2_fin = sol.y[:, -1]
    # «покоится»?
    # if hypot(v1_fin, v2_fin) > VEL_TOL:
    #     print(f"⚠️  (v1={v1_fin: .1e}, v2={v2_fin: .1e}) "
    #           f"к T_FINAL скорость ещё значительная")

    # ближе к нижней яме?
    dist_up = hypot(y1_fin - y_eq_up[0], y2_fin - y_eq_up[1])
    dist_down = hypot(y1_fin - y_eq_down[0], y2_fin - y_eq_down[1])
    return dist_down < dist_up


# ──────────────────────────────────────────────────────────────
# 2. Поиск K*(α) : doubling + binary search
# ──────────────────────────────────────────────────────────────
def find_K_star(alpha, k_iter, k0_guess):
    """Возвращает минимальную кинетическую энергию K*, достаточную
    для перщёлкивания при данном α."""
    K_low = 2.48e7
    K_high = k0_guess
    # экспоненциальное увеличение, пока не сработало
    while not snapped_through(K_high, alpha):
        K_low = K_high
        K_high *= 2.0
        if K_high > 1e9:
            raise RuntimeError("Не удалось найти верхнюю границу K.")


    # линейная разбивка СНИЗУ ВВЕРХ
    n_sub = 100
    subdiv = np.linspace(K_low, K_high, n_sub + 1)

    for ii in range(1, len(subdiv)):
        if snapped_through(subdiv[ii], alpha):  # нашли первую «безопасную» амплитуду
            K_low, K_high = subdiv[ii - 1], subdiv[ii]
            break
    # если цикл закончился без break, интервал остаётся [a_low, a_high]



    # двоичный поиск
    for _ in range(k_iter):
        K_mid = 0.5 * (K_low + K_high)
        if snapped_through(K_mid, alpha):
            K_high = K_mid
        else:
            K_low = K_mid

        # print(abs(K_high - K_low))

    final_precision = abs(K_high - K_low)
    print(f'precision ≈ {final_precision:.3e}')

    return K_high  # прицельная точность ~ (K_high-K_low)


# ──────────────────────────────────────────────────────────────
# 2*. Скан по K без двоичного поиска: просто проходим subdiv
# ──────────────────────────────────────────────────────────────

def find_K_bracket(alpha, k0_guess=3.05e7, K_low_init=2.0e7, K_max_cap=1e9):
    """
    Подбираем грубую «рамку» [K_low, K_high], где K_high уже щёлкает.
    Если начальный K_low не щёлкает, растягиваем вверх (doubling).
    Возвращаем (K_low, K_high).
    """
    # K_low = 2.44e7
    # K_high = 3.05e7
    K_low = 2.45e7
    K_high = 3.04e7

    # гарантируем, что нижняя граница не щёлкает
    if snapped_through(K_low, alpha):
        # если вдруг уже щёлкнуло на нижней границе — опустим ещё ниже
        while snapped_through(K_low, alpha) and K_low > 1.0:
            K_low *= 0.5

    # наращиваем верхнюю границу пока не щёлкнет
    while not snapped_through(K_high, alpha):
        K_low = K_high
        K_high *= 2.0
        if K_high > K_max_cap:
            raise RuntimeError("Не удалось найти верхнюю границу K (превышен cap).")

    return K_low, K_high

# ──────────────────────────────────────────────────────────────
# 3. Основной цикл по α и построение графика
# ──────────────────────────────────────────────────────────────
alphas = np.linspace(0.0, 1.0, 101)   # можно уплотнить/разрядить сетку
n_sub = 200                           # точек K на каждую α

alpha_list = []
K_list = []
flag_list = []   # True (зелёный) если snapped_through, иначе False (красный)

print("Scanning grid (α, K) …")
for a in alphas:
    print(f"α = {a:.3f} → bracket for K …")
    # подбираем рабочий диапазон K для текущего α
    K_low, K_high = find_K_bracket(a, k0_guess=3.05e7, K_low_init=2.0e7)

    # создаём равномерную линейную сетку по K в найденной рамке
    subdiv = np.linspace(K_low, K_high, n_sub + 1)

    # проходим сетку без двоичного поиска
    for K in subdiv:
        ok = snapped_through(K, a)
        alpha_list.append(a)
        K_list.append(K)
        flag_list.append(ok)

alpha_arr = np.array(alpha_list)
K_arr = np.array(K_list)
flag_arr = np.array(flag_list)
# ──────────────────────────────────────────────────────────────
# 4. Визуализация
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(7.5, 5.2))
# Цвет: зелёный = успех, красный = нет
colors = np.where(flag_arr, 'tab:green', 'tab:red')
plt.scatter(alpha_arr, K_arr, c=colors, s=12, alpha=0.8, edgecolors='none')

plt.plot([0, 1], [V_origin, V_origin], '--k', lw=1)

plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$K$ (initial kinetic energy)', fontsize=16)
# plt.title('Snap-through map: green = snapped, red = not snapped')
plt.title(r'Green = snapped, red = not snapped, c = {}'.format(c))
plt.grid(True, alpha=0.3)
plt.show()
