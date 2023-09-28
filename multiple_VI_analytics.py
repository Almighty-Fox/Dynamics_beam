import numpy as np
from sympy import *
from scipy import integrate
import pylab as plt
import time
import pandas as pd


# определяем частоты, формы и коэф (D = form ** 2) для балки без барьера
def beam_without_barrier():
    al_l = np.arange(0, 12, 0.00001)
    fun1 = np.sinh(al_l) * np.cos(al_l) + 1
    roots1 = []
    omega1 = []
    for i in range(len(fun1) - 1):
        if fun1[i] * fun1[i+1] < 0:
            roots1_i = (al_l[i] + al_l[i+1]) / 2
            roots1.append(roots1_i)
            omega1.append((roots1_i / l2) ** 2 * np.power((E * J_inertia / ro / S), (1 / 2)))

    D1 = []
    forms1 = []
    for root in roots1:
        alpha_i = root / l2

        u_i = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                 np.cosh(alpha_i * z) - np.cos(alpha_i * z)))
        form1_i = np.array([u_i(ii) for ii in np.linspace(0, l2, point)])
        forms1.append(form1_i)

        u_i_kv = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                    np.cosh(alpha_i * z) - np.cos(alpha_i * z))) ** 2
        D1_i = integrate.quad(u_i_kv, 0, l2)[0]
        D1.append(D1_i)

    return roots1, D1, forms1, omega1


# динамика балки без барьера
def beam_no_VI_vibrations(roots1, D1, forms1, omega1, disp_start, vel_start):
    global t_global
    integ_u = np.zeros(len(roots1), dtype=float)
    integ_u_dif = np.zeros(len(roots1), dtype=float)
    for id in range(len(roots1)):

        form1_i = forms1[id]

        integ_u_body = disp_start * form1_i
        integ_u_dif_body = vel_start * form1_i

        for ii in range(1, len(integ_u_body)):
            integ_u[id] += (integ_u_body[ii] + integ_u_body[ii - 1]) / 2
            integ_u_dif[id] += (integ_u_dif_body[ii] + integ_u_dif_body[ii - 1]) / 2
        integ_u[id] *= dl
        integ_u_dif[id] *= dl

    t_loc = 0
    y_list = disp_start
    while y_list[l1] > 0:  # ???????????????????????
        t_loc += t_step
        t_global += t_step
        y_list = np.zeros(point, dtype=float)
        for id in range(len(roots1)):
            w_i = omega1[id]

            y_i = (forms1[id] / D1[id]) * (np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])

            y_list += y_i  # суммарная функция колебания

    vel_list = np.zeros(point, dtype=float)
    for id in range(len(roots1)):
        w_i = omega1[id]
        vel_i = (forms1[id] / D1[id]) * (-w_i * np.sin(w_i * t_loc) * integ_u[id] + w_i * np.cos(w_i * t_loc) / w_i * integ_u_dif[id])
        vel_list += vel_i  # суммарная функция колебания

    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ С БАРЬЕРОМ
    beam_with_VI_vibrations()
    # return y_list, vel_list


# определяем частоты, формы и коэф (D = form ** 2) для балки с барьером
def beam_with_barrier():
    al_l = np.arange(0, 12, 0.00001)
    fun1 = np.sinh(al_l) * np.cos(al_l) + 1
    roots1 = []
    omega1 = []
    for i in range(len(fun1) - 1):
        if fun1[i] * fun1[i+1] < 0:
            roots1_i = (al_l[i] + al_l[i+1]) / 2
            roots1.append(roots1_i)
            omega1.append((roots1_i / l2) ** 2 * np.power((E * J_inertia / ro / S), (1 / 2)))

    D1 = []
    forms1 = []
    for root in roots1:
        alpha_i = root / l2

        u_i = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                 np.cosh(alpha_i * z) - np.cos(alpha_i * z)))
        form1_i = np.array([u_i(ii) for ii in np.linspace(0, l2, point)])
        forms1.append(form1_i)

        u_i_kv = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                    np.cosh(alpha_i * z) - np.cos(alpha_i * z))) ** 2
        D1_i = integrate.quad(u_i_kv, 0, l2)[0]
        D1.append(D1_i)

    return roots1, D1, forms1, omega1


# Динамика балки с барьером
def beam_with_VI_vibrations():
    pass


# Задачем начальные условия
def initial_conditions():
    # return disp, vel
    pass


# -----------------------------------------------
ro = 7850
E = 2e11
a = 10e-3
S = a * a  # площадь сечения балки
J_inertia = a * a ** 3 / 12  # момент инерции сечения
ro = 7850  # плотность
E = 2e11  # модуль Юнга
# ------------------------------------------------
point = 200 + 1  # количество элементов балки
l1 = 0.9  # местоположение барьера
l2 = 1  # длина балки
dl = l2 / (point - 1)
# ------------------------------------------------
t_global = 0.0
t_step = 0.00001
t_end = 10.0
# ---------------------------------------------

# Задаем начальную деформацию и скорость балки
[disp_start, vel_start] = initial_conditions()

# Определяем параметры балки без барьера
[roots1, D1, forms1, omega1] = beam_without_barrier()

# Определяем параметры балки с барьером
[roots1, D1, forms1, omega1] = beam_with_barrier()

# Запускаем внешний цикл по времени
while True:
    # Вначале запускаем динамику балки без барьера
    beam_no_VI_vibrations(roots1, D1, forms1, omega1, disp_start, vel_start)
    #
#
beam_without_barrier()