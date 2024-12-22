# import numpy as np
# import matplotlib.pyplot as plt
# from sympy import symbols, integrate, pi, sin
# import bisect
#
# # 1. Параметры задачи
# L = 1  # длина струны
# c = 1  # скорость распространения волны
# dx = 0.001 # шаг по пространству
# step_t = 0.001  # шаг по времени
# T = 3  # максимальное время
#
# # 2. Сетка по пространству и времени
# x_lst = np.arange(0, L, dx)  # сетка по x
# n_points = len(x_lst)
#
# # 3. Начальные условия
# # Начальная форма струны f(x)
# x = symbols('x')
# # f = sin(pi * x / L)  # f(x) = sin(pi * x / L) на [0, L]
# f = 0 * x
#
# # Начальное поле скоростей g(x)
# x = symbols('x')
# # g_sym_1 = sin(pi * x / L)
# # g_sym_2 = sin(pi * x / L)
# # g_sym_3 = sin(pi * x / L)
# # g_sym_4 = sin(pi * x / L)
# g_sym_1, g_sym_2, g_sym_3, g_sym_4 = 0, -1, 1, 0
#
# loc_bar = 0.7
# change_fun_lst = [0, loc_bar - 0.5, loc_bar, 1.5 - loc_bar, 1]
#
# f_sym_lst = [g_sym_1, g_sym_2, g_sym_3, g_sym_4]
# f_sym_lst_integ = [integrate(g_sym_1, x), integrate(g_sym_2, x), integrate(g_sym_3, x), integrate(g_sym_4, x)]
# f_sym_lst_integ_lim = [integrate(g_sym_1, (x, change_fun_lst[0], change_fun_lst[1])), integrate(g_sym_2, (x, change_fun_lst[1], change_fun_lst[2])),
#                        integrate(g_sym_3, (x, change_fun_lst[2], change_fun_lst[3])), integrate(g_sym_4, (x, change_fun_lst[3], change_fun_lst[4]))]
#
# def integ_0_start_end_L(start_x, end_x):
#     # возвращаем 0 если пределы интеграла совпадают
#     if start_x == end_x:
#         return 0
#     # если пределы интеграла в пределах длины струны
#     if 0 <= start_x < end_x <= L:
#         index_start_x = bisect.bisect_left(change_fun_lst, start_x) - 1
#         index_end_x = bisect.bisect_left(change_fun_lst, end_x) - 1
#         if index_start_x == index_end_x:
#             return f_sym_lst_integ[index_start_x].subs(x, end_x) - f_sym_lst_integ[index_start_x].subs(x, start_x)
#
#         integ_part_1 = f_sym_lst_integ[index_start_x].subs(x, change_fun_lst[index_start_x + 1]) - f_sym_lst_integ[index_start_x].subs(x, start_x)
#         integ_part_2 = f_sym_lst_integ[index_end_x].subs(x, end_x) - f_sym_lst_integ[index_end_x].subs(x, change_fun_lst[index_end_x])
#         if index_end_x - index_start_x == 1:
#             return integ_part_1 + integ_part_2
#         integ_part_3 = 0
#         for ii in range(index_start_x + 1, index_end_x):
#             integ_part_3 += f_sym_lst_integ_lim[ii]
#         return integ_part_1 + integ_part_2 + integ_part_3
#
#
# def integ_L_start_end_2L(start_x, end_x):
#     return -integ_0_start_end_L(2 * L - end_x, 2 * L - start_x)
#
#
# def integ_L_end_start_2L(start_x, end_x):
#     integ_part_1 = integ_L_start_end_2L(start_x, 2 * L)
#     integ_part_2 = sum(f_sym_lst_integ_lim)
#     integ_part_3 = integ_L_start_end_2L(L, end_x)
#
#     return integ_part_1 + integ_part_2 + integ_part_3
#
#
# def integ_0_end_start_L(start_x, end_x):
#     integ_part_1 = integ_0_start_end_L(start_x, L)
#     integ_part_2 = -sum(f_sym_lst_integ_lim)
#     integ_part_3 = integ_0_start_end_L(0, end_x)
#
#     return integ_part_1 + integ_part_2 + integ_part_3
#
#
# def integ_0_start_L_end_2L(start_x, end_x):
#     integ_part_1 = integ_0_start_end_L(start_x, L)
#     integ_part_2 = integ_L_start_end_2L(L, end_x)
#
#     return integ_part_1 + integ_part_2
#
#
# def integ_0_end_L_start_2L(start_x, end_x):
#     integ_part_1 = integ_L_start_end_2L(start_x, 2 * L)
#     integ_part_2 = integ_0_start_end_L(0, end_x)
#
#     return integ_part_1 + integ_part_2
#
#
# def r_mod_2L(r, L):
#     r_mod = r % (2 * L)  # остаток от деления на 2L
#     return r_mod
#
#
#
#
# t_cur = 0
# u_lst = np.zeros_like(x_lst)
#
# while t_cur < T:
#     t_cur += step_t
#     for ii, x_cur in enumerate(x_lst):
#         r2 = r_mod_2L(x_cur + c * t_cur, L)
#         r1 = r_mod_2L(x_cur - c * t_cur, L)
#         if r1 <= L:
#             f1 = f.subs(x, r1)
#         else:
#             f1 = -f.subs(x, 2 * L - r1)
#
#         if r2 <= L:
#             f2 = f.subs(x, r2)
#         else:
#             f2 = -f.subs(x, 2 * L - r2)
#
#         if r1 == r2:
#             integ_vel = 0
#         elif 0 <= r1 < r2 <= L:
#             integ_vel = integ_0_start_end_L(r1, r2)
#         elif 0 <= r2 < r1 <= L:
#             integ_vel = integ_0_end_start_L(r1, r2)
#         elif L <= r1 < r2 <= 2 * L:
#             integ_vel = integ_L_start_end_2L(r1, r2)
#         elif L <= r2 < r1 <= 2 * L:
#             integ_vel = integ_L_end_start_2L(r1, r2)
#         elif 0 <= r1 <= L <= r2 <= 2 * L:
#             integ_vel = integ_0_start_L_end_2L(r1, r2)
#         elif 0 <= r2 <= L <= r1 <= 2 * L:
#             integ_vel = integ_0_end_L_start_2L(r1, r2)
#         else:
#             raise SystemExit(f"r1 = {r1}, r2 = {r2}")
#
#
#         solution_f = 0.5 * (f1 + f2) + 0.5 / c * integ_vel
#         u_lst[ii] = solution_f
#
#     print(u_lst[10])
#     plt.plot(x_lst, u_lst)
#     plt.grid()
#     plt.pause(0.001)
#     plt.clf()
#
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, integrate
import bisect
from matplotlib.animation import FuncAnimation

# 1. Параметры задачи
L = 1    # длина струны
c = 1    # скорость распространения волны
dx = 0.001  # шаг по пространству
step_t = 0.001  # шаг по времени
T = 3    # максимальное время
Nt = int(T / step_t)

# 2. Сетка по пространству и времени
x_lst = np.arange(0, L, dx)  # сетка по x

# 3. Начальные условия
x = symbols('x')
f = 0 * x  # f(x) = 0

# Начальное поле скоростей g(x) кусочно-задано
g_sym_1, g_sym_2, g_sym_3, g_sym_4 = 0, -1, 1, 0

loc_bar = 0.7
change_fun_lst = [0, loc_bar - 0.5, loc_bar, 1.5 - loc_bar, 1]

f_sym_lst = [g_sym_1, g_sym_2, g_sym_3, g_sym_4]
f_sym_lst_integ = [integrate(g, x) for g in f_sym_lst]
f_sym_lst_integ_lim = [
    integrate(g_sym_1, (x, change_fun_lst[0], change_fun_lst[1])),
    integrate(g_sym_2, (x, change_fun_lst[1], change_fun_lst[2])),
    integrate(g_sym_3, (x, change_fun_lst[2], change_fun_lst[3])),
    integrate(g_sym_4, (x, change_fun_lst[3], change_fun_lst[4]))
]

def r_mod_2L(r, L):
    r_mod = r % (2 * L)  # остаток от деления на 2L
    return r_mod

def integ_0_start_end_L(start_x, end_x):
    if start_x == end_x:
        return 0
    if 0 <= start_x < end_x <= L:
        index_start_x = bisect.bisect_left(change_fun_lst, start_x) - 1
        index_end_x = bisect.bisect_left(change_fun_lst, end_x) - 1
        if index_start_x == index_end_x:
            return f_sym_lst_integ[index_start_x].subs(x, end_x) - f_sym_lst_integ[index_start_x].subs(x, start_x)
        integ_part_1 = f_sym_lst_integ[index_start_x].subs(x, change_fun_lst[index_start_x + 1]) - f_sym_lst_integ[index_start_x].subs(x, start_x)
        integ_part_2 = f_sym_lst_integ[index_end_x].subs(x, end_x) - f_sym_lst_integ[index_end_x].subs(x, change_fun_lst[index_end_x])
        if index_end_x - index_start_x == 1:
            return integ_part_1 + integ_part_2
        integ_part_3 = 0
        for ii in range(index_start_x + 1, index_end_x):
            integ_part_3 += f_sym_lst_integ_lim[ii]
        return integ_part_1 + integ_part_2 + integ_part_3

def integ_L_start_end_2L(start_x, end_x):
    return -integ_0_start_end_L(2 * L - end_x, 2 * L - start_x)

def integ_L_end_start_2L(start_x, end_x):
    integ_part_1 = integ_L_start_end_2L(start_x, 2 * L)
    integ_part_2 = sum(f_sym_lst_integ_lim)
    integ_part_3 = integ_L_start_end_2L(L, end_x)
    return integ_part_1 + integ_part_2 + integ_part_3

def integ_0_end_start_L(start_x, end_x):
    integ_part_1 = integ_0_start_end_L(start_x, L)
    integ_part_2 = -sum(f_sym_lst_integ_lim)
    integ_part_3 = integ_0_start_end_L(0, end_x)
    return integ_part_1 + integ_part_2 + integ_part_3

def integ_0_start_L_end_2L(start_x, end_x):
    integ_part_1 = integ_0_start_end_L(start_x, L)
    integ_part_2 = integ_L_start_end_2L(L, end_x)
    return integ_part_1 + integ_part_2

def integ_0_end_L_start_2L(start_x, end_x):
    integ_part_1 = integ_L_start_end_2L(start_x, 2 * L)
    integ_part_2 = integ_0_start_end_L(0, end_x)
    return integ_part_1 + integ_part_2

# Функция для получения значения g(r)
def g_value(r):
    r_mod_val = r_mod_2L(r, L)
    if r_mod_val <= L:
        if 0 <= r_mod_val < change_fun_lst[1]:
            return g_sym_1
        elif change_fun_lst[1] <= r_mod_val < change_fun_lst[2]:
            return g_sym_2
        elif change_fun_lst[2] <= r_mod_val < change_fun_lst[3]:
            return g_sym_3
        else:
            return g_sym_4
    else:
        r_mod_val = 2 * L - r_mod_val
        if 0 <= r_mod_val < change_fun_lst[1]:
            return -g_sym_1
        elif change_fun_lst[1] <= r_mod_val < change_fun_lst[2]:
            return -g_sym_2
        elif change_fun_lst[2] <= r_mod_val < change_fun_lst[3]:
            return -g_sym_3
        else:
            return -g_sym_4

# Создаем фигуру и оси
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
ax1.set_title("Displacement (u)")
ax2.set_title("Velocity (u_t)")
plt.tight_layout()

line1, = ax1.plot([], [], lw=2)
line2, = ax2.plot([], [], lw=2)

ax1.grid(True)
ax2.grid(True)

def update(frame):
    t_cur = frame * step_t
    u_t_lst = np.zeros_like(x_lst)
    u_lst = np.zeros_like(x_lst)

    for ii, x_cur in enumerate(x_lst):
        r2 = r_mod_2L(x_cur + c * t_cur, L)
        r1 = r_mod_2L(x_cur - c * t_cur, L)

        # f=0, поэтому f1 и f2 равны нулю
        f1 = 0
        f2 = 0

        # Интеграл по g
        if r1 == r2:
            integ_vel = 0
        elif 0 <= r1 < r2 <= L:
            integ_vel = integ_0_start_end_L(r1, r2)
        elif 0 <= r2 < r1 <= L:
            integ_vel = integ_0_end_start_L(r1, r2)
        elif L <= r1 < r2 <= 2 * L:
            integ_vel = integ_L_start_end_2L(r1, r2)
        elif L <= r2 < r1 <= 2 * L:
            integ_vel = integ_L_end_start_2L(r1, r2)
        elif 0 <= r1 <= L <= r2 <= 2 * L:
            integ_vel = integ_0_start_L_end_2L(r1, r2)
        elif 0 <= r2 <= L <= r1 <= 2 * L:
            integ_vel = integ_0_end_L_start_2L(r1, r2)
        else:
            raise SystemExit(f"r1 = {r1}, r2 = {r2}")

        solution_f = 0.5 * (f1 + f2) + 0.5 / c * integ_vel
        u_lst[ii] = solution_f

        # Расчет скорости:
        # u_t(x,t) = (g(x+ct) + g(x-ct))/2
        u_t_lst[ii] = 0.5 * (g_value(x_cur + c * t_cur) + g_value(x_cur - c * t_cur))

    # Обновляем данные линий
    line1.set_data(x_lst, u_lst)
    line2.set_data(x_lst, u_t_lst)
    ax1.set_xlim(0, L)
    ax2.set_xlim(0, L)
    # Установим вертикальные границы по displacement и velocity при желании (при необходимости)
    ax1.set_ylim(-0.3, 0.3)
    ax2.set_ylim(-1.5, 1.5)

    return line1, line2

ani = FuncAnimation(fig, update, frames=Nt, blit=True, interval=0.2)
plt.show()






