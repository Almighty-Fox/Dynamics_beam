import numpy as np
from sympy import *
from scipy import integrate
import pylab as plt
import time
import pandas as pd
from class_beam_elements import Class_beam_elements
from create_global_matrix import *


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
    form1_second_dif = []
    for root in roots1:
        alpha_i = root / l2

        u_i = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                 np.cosh(alpha_i * z) - np.cos(alpha_i * z)))
        form1_i = np.array([u_i(ii) for ii in np.linspace(0, l2, point)])
        forms1.append(form1_i)

        u_i_second_dif = lambda z: (
                (np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                -alpha_i ** 2 * np.sin(alpha_i * z) - alpha_i ** 2 * np.sinh(alpha_i * z)) + (
                        alpha_i ** 2 * np.cosh(alpha_i * z) + alpha_i ** 2 * np.cos(alpha_i * z)))
        form1_second_dif_i = np.array([u_i_second_dif(ii) for ii in np.linspace(0, l2, point)])
        form1_second_dif.append(form1_second_dif_i)

        u_i_kv = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                    np.cosh(alpha_i * z) - np.cos(alpha_i * z))) ** 2
        D1_i = integrate.quad(u_i_kv, 0, l2)[0]
        D1.append(D1_i)

    return roots1, D1, forms1, omega1, form1_second_dif


# динамика балки без барьера
def beam_no_VI_vibrations(disp_start, vel_start):
    print('NO barrier')
    global t_global, y_barrier_global, Q_list_time, total_energy, y_end_global, time_is_up
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

    first_step = True

    while ((y_list[point_barrier] >= 0) or (first_step)) and (not time_is_up):  # ???????????????????????
        first_step = False
        t_loc += t_step
        t_global += t_step

        if t_global >= t_end:
            time_is_up = True

        y_list = np.zeros(point, dtype=float)
        for id in range(len(roots1)):
            w_i = omega1[id]

            y_i = (forms1[id] / D1[id]) * (np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])

            y_list += y_i  # суммарная функция колебания

            # --------------------ENERGY------------------------------------
            vel_i = (forms1[id] / D1[id]) * (-w_i * np.sin(w_i * t_loc) * integ_u[id] + np.cos(w_i * t_loc) * integ_u_dif[id])
            kinetic_energy_id = 0.5 * (ro * S) * (dl * sum(vel_i ** 2))

            y_i_second_dif = (form1_second_dif[id] / D1[id]) * (np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])
            potential_energy_id = 0.5 * E * J_inertia * (dl * sum(y_i_second_dif ** 2))

            total_energy_id = kinetic_energy_id + potential_energy_id
            total_energy[id] = total_energy_id
            # --------------------------------------------------------------

        y_barrier_global.append(y_list[point_barrier])
        y_end_global.append(y_list[-1])

        Q_list_time.append(0)
        graph(y_list)

    vel_list = np.zeros(point, dtype=float)
    for id in range(len(roots1)):
        w_i = omega1[id]
        vel_i = (forms1[id] / D1[id]) * (-w_i * np.sin(w_i * t_loc) * integ_u[id] + np.cos(w_i * t_loc) * integ_u_dif[id])
        vel_list += vel_i  # суммарная функция колебания

    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ С БАРЬЕРОМ
    # beam_with_VI_vibrations(y_list, vel_list)
    return y_list, vel_list


# определяем частоты, формы и коэф (D = form ** 2) для балки с барьером
def beam_with_barrier():
    k = np.arange(0.0001, 15, 0.0001)  # лист значений каппы - замененная частота

    A1 = ((k*np.cos(k*l1)-k*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k*np.sin(k*l1)-k*np.sinh(k*l1)))
    A2 = ((-k**2*np.sin(k*l1)-k**2*np.sinh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k**2*np.cos(k*l1)-k**2*np.cosh(k*l1)))

    B1 = ((k*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)) -
          (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (-k*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)))

    B2 = ((-k**2*np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)) -
          (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (-k**2*np.cos(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)))

    A1_fun = lambda k: ((k*np.cos(k*l1)-k*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k*np.sin(k*l1)-k*np.sinh(k*l1)))
    A2_fun = lambda k: ((-k**2*np.sin(k*l1)-k**2*np.sinh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k**2*np.cos(k*l1)-k**2*np.cosh(k*l1)))

    B1_fun = lambda k: ((k*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)) -
          (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (-k*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)))

    B2_fun = lambda k: ((-k**2*np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)) -
          (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (-k**2*np.cos(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)))

    # для нахождения силы взаимодействия с барьером необходимо найти разрыв 3ей производной формы балки в точке барьера
    # третья производная формы левой части балки в точке барьера
    A3_fun = lambda k: ((-k**3*np.cos(k*l1)-k**3*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(k**3*np.sin(k*l1)-k**3*np.sinh(k*l1)))
    # третья производная формы правой части балки в точке барьера
    B3_fun = lambda k: ((-k**3*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k**3*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k**3*np.sinh(k*l1)) -
          (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (k**3*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k**3*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k**3*np.sinh(k*l1)))

    fun2 = -A1 * B2 + A2 * B1
    roots2 = []
    omega2 = []
    for i in range(len(fun2) - 1):
        if (fun2[i] * fun2[i+1] < 0) and (abs(fun2[i] * fun2[i+1]) < 300):
            roots2_i = (k[i] + k[i+1]) / 2
            roots2.append(roots2_i)
            omega2.append(np.sqrt(E * J_inertia / ro / S) * np.array(roots2_i) ** 2)

    x5 = np.linspace(0, l2, point)[:round(l1 * point)]
    x6 = np.linspace(0, l2, point)[round(l1 * point):]

    D2 = []
    forms2 = []
    v1 = []  # лист форм левой части балки при различных частотах
    v2 = []  # тоже для правой части
    for i in range(len(roots2)):
        k = roots2[i]
        x = np.linspace(0, l2, point)[:round(l1 * point)]
        alpha1 = 1
        v1.append(alpha1 * ((np.sin(k*x) - np.sinh(k*x)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1)) * (np.cos(k*x) - np.cosh(k*x))))
        x = np.linspace(0, l2, point)[round(l1 * point):]
        beta1 = alpha1 * A1_fun(k) / B1_fun(k)
        v2.append(beta1 * ((np.sin(k*x) + (np.cos(k*l2)*np.cosh(k*l2) - np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*x) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*x))
                  - (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (np.cos(k*x)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*x)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*x))))

        v1v2 = np.concatenate((v1[i], v2[i]))
        forms2.append(v1v2)  # форма всей балки как сумма формы левой и правой частей
        x = np.linspace(0, l2, point)

        integ_v1v2_2 = 0
        for ii in range(1, len(v1v2)):
            integ_v1v2_2 += (v1v2[ii] ** 2 + v1v2[ii - 1] ** 2) / 2
        integ_v1v2_2 *= dl

        D2.append(integ_v1v2_2)

    Q_lst = [(A3_fun(roots2[ii]) - A1_fun(roots2[ii]) / B1_fun(roots2[ii]) * B3_fun(roots2[ii])) for ii in range(len(roots2))]

    return roots2, D2, forms2, omega2, Q_lst


# Динамика балки с барьером
def beam_with_VI_vibrations(disp_start, vel_start):
    print('with barrier')
    global t_global, y_barrier_global, Q_list_time, y_end_global, time_is_up
    integ_u = np.zeros(len(roots2), dtype=float)
    integ_u_dif = np.zeros(len(roots2), dtype=float)
    for id in range(len(roots2)):

        form2_i = forms2[id]

        integ_u_body = disp_start * form2_i
        integ_u_dif_body = vel_start * form2_i

        for ii in range(1, len(integ_u_body)):
            integ_u[id] += (integ_u_body[ii] + integ_u_body[ii - 1]) / 2
            integ_u_dif[id] += (integ_u_dif_body[ii] + integ_u_dif_body[ii - 1]) / 2
        integ_u[id] *= dl
        integ_u_dif[id] *= dl

    t_loc = 0
    y_list = disp_start
    y_list_previous = disp_start

    Q_t = -1
    while (Q_t < 0) and (not time_is_up):  # ???????????????????????
        Q_t = 0
        t_loc += t_step
        t_global += t_step

        if t_global >= t_end:
            time_is_up = True

        y_list_previous = y_list
        y_list = np.zeros(point, dtype=float)

        for id in range(len(roots2)):
            w_i = omega2[id]

            y_i = (forms2[id] / D2[id]) * (
                    np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])

            y_list += y_i  # суммарная функция колебания

            Q_t += (Q_lst[id] / D2[id]) * (
                    np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])


        Q_list_time.append(Q_t)

        y_barrier_global.append(y_list[point_barrier])
        y_end_global.append(y_list[-1])

        if Q_t < 0:
            graph(y_list)

    # ----------------------------------------------------
    # последний шаг откатываем, поскольку Q>0 мы не хотим
    y_list = y_list_previous
    t_loc -= t_step
    t_global -= t_step
    Q_list_time.pop()
    y_barrier_global.pop()
    y_end_global.pop()
    # ----------------------------------------------------

    vel_list = np.zeros(point, dtype=float)
    for id in range(len(roots2)):
        w_i = omega2[id]
        vel_i = (forms2[id] / D2[id]) * (
                    -w_i * np.sin(w_i * t_loc) * integ_u[id] + w_i * np.cos(w_i * t_loc) / w_i * integ_u_dif[id])
        vel_list += vel_i  # суммарная функция колебания

    # y_list[point_barrier] = 0
    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ БЕЗ БАРЬЕРА
    # beam_no_VI_vibrations(y_list, vel_list)
    return y_list, vel_list


# Задачем начальные условия
def initial_conditions():
    natural_fr = False
    if natural_fr:
        # выбираем начальное перемещение в точности по собственной форме
        disp_start = forms1[0] / 1.2
        vel_start = np.zeros(point, dtype=float)
    else:
        # ---------------------------- НУ из МКЭ ---------------------
        MaxNode = 200 + 1
        elements = [Class_beam_elements(i, l2, MaxNode, E, J_inertia, ro) for i in
                    range(MaxNode - 1)]  # создаем массив балочных элементов

        # формируем матрицы для дифура
        global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ (матрица жесткости)
        global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ в МЖ

        global_force = np.zeros((2 * MaxNode, 1))  # создаем размер глобального вектора сил
        global_force = create_global_force(global_force, f_ampl=1)  # создаем начальный вектор сил

        disp_start_column = np.matmul(np.linalg.inv(global_stiffness), global_force)
        disp_start = [disp_start_column[i * 2, 0] for i in range(MaxNode)]
        vel_start = np.zeros(point, dtype=float)
        # ----------------------------------------------------------------------------

    return disp_start, vel_start, max(disp_start)


def graph(y_list):
    # pass
    if len(Q_list_time) % 500 == 0:
        fig.suptitle('Time=' + str('%.2f' % t_global) + 's=' + str('%.2f' % (t_global * 1e3)) + 'ms')

        axs[0][0].set_title('Beam shape')
        axs[1][0].set_title('VI force')
        axs[1][1].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10)

        axs[0][0].plot(np.linspace(0, l2, point), y_list, 'k', linewidth=1)
        axs[0][0].plot([l1], [0], 'ro', linewidth=2)
        axs[0][0].grid()
        axs[0][0].set_ylim(-max_disp_start*1.0, max_disp_start*1.0)

        Q_list_time_format = -np.array(Q_list_time) * E * J_inertia
        axs[1][0].plot(np.linspace(0, t_global, len(Q_list_time_format)), Q_list_time_format, 'k', linewidth=1)
        axs[1][0].grid()

        axs[1][1].plot(np.linspace(0, t_global, len(y_barrier_global)), y_barrier_global, 'g', linewidth=1)
        axs[1][1].plot(np.linspace(0, t_global, len(y_end_global)), y_end_global, 'k', linewidth=1)
        axs[1][1].plot(np.linspace(0, t_global, len(y_end_global)), [0] * len(y_end_global), 'r--', linewidth=1)
        axs[1][1].grid()

        # ------------- plot ENERGY -----------------------------
        df = pd.DataFrame(total_energy[:4], index=range(1, 5), columns=['Value'])
        df['Percentage'] = (df['Value'] / df['Value'].sum()) * 100
        # Plot the bar chart
        axs[0][1].bar(x=df.index, height=df['Percentage'], color='skyblue')

        # Annotate each bar with percentages
        for p in axs[0][1].patches:
            axs[0][1].annotate(f'{p.get_height():.2f}%',
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom', fontsize=8, color='black')

        axs[0][1].set_title('Energy mode')
        axs[0][1].set_xticks(np.arange(1, 5))
        # ---------------------------------------------------------

        plt.pause(0.001)
        axs[0][0].clear()
        axs[1][0].clear()
        axs[0][1].clear()
        axs[1][1].clear()


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
l1 = 0.7  # местоположение барьера
l2 = 1  # длина балки
dl = l2 / (point - 1)
point_barrier = round((point - 1) * l1)
# ------------------------------------------------
t_global = 0.0
t_step = 0.000001
t_end = 0.15
time_is_up = False  # флаг на то, закончилось ли время
# ---------------------------------------------

y_barrier_global = []
y_end_global = []
Q_list_time = []

print('Начинаем фигачить')

# Определяем параметры балки без барьера
[roots1, D1, forms1, omega1, form1_second_dif] = beam_without_barrier()
print('Определили параметры балки без барьера')

total_energy = [0] * len(roots1)  # Лист energy мод

# Определяем параметры балки с барьером
[roots2, D2, forms2, omega2, Q_lst] = beam_with_barrier()
print('Определили параметры балки с барьером')

# Задаем начальную деформацию и скорость балки
[disp_start, vel_start, max_disp_start] = initial_conditions()
print('Задали НУ')

print('Запускаем динамику')

fig, axs = plt.subplots(2, 2, squeeze=False)  # создаем саб плот из 2 графиков
# plt.subplots_adjust(wspace=0.4, hspace=0.7)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.pause(4)

# while True:
while not time_is_up:
    print('Time = ', str(t_global))
    # Вначале запускаем динамику балки без барьера
    [disp_start, vel_start] = beam_no_VI_vibrations(disp_start, vel_start)

    print('Time = ', str(t_global))
    [disp_start, vel_start] = beam_with_VI_vibrations(disp_start, vel_start)


Q_list_time_format = -np.array(Q_list_time) * E * J_inertia
file_name = 'Analytics_write_VI_force.txt'
with open(r'./plots/' + file_name, 'w') as cur_file:
    cur_file.write(str(list(Q_list_time_format)))

write_time = np.linspace(0, t_global, len(Q_list_time_format))
file_name = 'Analytics_write_time.txt'
with open(r'./plots/' + file_name, 'w') as cur_file:
    cur_file.write(str(list(write_time)))

file_name = 'Analytics_write_disp.txt'
with open(r'./plots/' + file_name, 'w') as cur_file:
    cur_file.write(str(list(y_end_global)))
