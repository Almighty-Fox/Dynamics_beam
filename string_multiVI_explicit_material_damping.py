import numpy as np
from sympy import *
import pylab as plt
import time


# динамика балки без барьера
def beam_no_VI_vibrations(x_vals, y_vals, v_vals, dt=6e-4, gamma_=0.002):
    global t_global, time_is_up, t_global_lst

    t_step_id = 0
    dt = t_step_lst[t_step_id]

    # Инициализация массивов решения
    u = np.zeros(Nx)
    u_prev = np.zeros(Nx)
    u_next = np.zeros(Nx)

    # Начальные условия
    u[:] = y_vals.copy()
    u_prev[:] = u - v_vals * dt

    # Функция для расчёта энергии
    def total_energy(u, u_prev):
        kinetic = np.sum((u - u_prev) ** 2) / (2 * dt ** 2)
        potential = np.sum((np.diff(u) / dx) ** 2) / 2
        return (kinetic + potential) * dx

    energy_data = []
    flag = True

    t_loc_lst = []
    print('NO barrier')
    is_first_no_VI = True


    t_loc = 0
    first_step = True


    # while ((u[point_barrier] >= 0) or (first_step)) and (not time_is_up):  # ???????????????????????
    while True:
        t_step = t_step_lst[t_step_id]
        first_step = False
        t_loc += t_step
        t_global += t_step
        # t_global_lst.append(t_global)
        # t_loc_lst.append(t_loc)

        if t_global >= t_end:
            time_is_up = True


        # Основной вычислительный цикл
        for i in range(1, Nx - 1):
            # Вычисление второй производной по пространству для текущего и предыдущего слоя
            d2u_dx2_n = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2
            d2u_dx2_prev = (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]) / dx ** 2

            # Явная схема для:
            # u_{n+1}[i] = 2 u_n[i] - u_{n-1}[i]
            #               + dt^2 * (u_xx^n[i])
            #               + gamma_ * dt * (u_xx^n[i] - u_xx^{n-1}[i])
            # Объединяя:
            u_next[i] = (2 * u[i] - u_prev[i]
                         + dt ** 2 * d2u_dx2_n
                         + gamma_ * dt * (d2u_dx2_n - d2u_dx2_prev))

        # Граничные условия
        u_next[0] = 0
        u_next[-1] = 0

        # Обновляем слои
        u_prev[:] = u[:]
        u[:] = u_next[:]

        current_energy = total_energy(u, u_prev)
        energy_data.append(current_energy)
        # ------------------------------------------------------


        # # если зашли в барьер, откатываем шаг назад и уменьшаем шаг на меньший из листа шагов
        # if (t_step_id > 0) and (u[point_barrier] < 0):
        #     t_loc -= t_step
        #     t_global -= t_step
        #     t_global_lst.pop()
        #     u = u_prev.copy()
        #     t_loc_lst.pop()
        #
        #     t_step_id -= 1
        #
        #     continue

        if (t_step_id < len(t_step_lst)-1) and (u[point_barrier] > 0):
            t_step_id += 1


        v_cur = (u - u_prev) / dt
        graph(u, v_cur)



    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ С БАРЬЕРОМ

    return u, v_cur, t_loc_lst



# # Динамика балки с барьером
# def beam_with_VI_vibrations(disp_start, vel_start):
#     t_loc_lst = []
#     print('with barrier')
#
#     is_first_no_VI = True
#     global t_global, y_barrier_global, Q_list_time, y_end_global, time_is_up, t_global_lst, global_en_lst, kinetic_energy_id_sum_lst, potential_energy_id_sum_lst
#     integ_u = np.zeros(len(roots2), dtype=float)
#     integ_u_dif = np.zeros(len(roots2), dtype=float)
#     for id in range(len(roots2)):
#
#         form2_i = forms2[id]
#
#         integ_u_body = disp_start * form2_i
#         integ_u_dif_body = vel_start * form2_i
#
#         for ii in range(1, len(integ_u_body)):
#             integ_u[id] += (integ_u_body[ii] + integ_u_body[ii - 1]) / 2
#             integ_u_dif[id] += (integ_u_dif_body[ii] + integ_u_dif_body[ii - 1]) / 2
#         integ_u[id] *= dl
#         integ_u_dif[id] *= dl
#
#     t_loc = 0
#     y_list = disp_start
#
#     Q_t = -1
#     t_step_id = 0
#     first_step = True
#     while ((Q_t < 0) and (not time_is_up)): # or first_step:  # ???????????????????????
#         # first_step = False
#         t_step = t_step_lst[t_step_id]
#         Q_t_previous = Q_t
#         Q_t = 0
#         t_loc += t_step
#         t_global += t_step
#         t_global_lst.append(t_global)
#         t_loc_lst.append(t_loc)
#
#         if t_global >= t_end:
#             time_is_up = True
#
#         y_list_previous = y_list.copy()
#         y_list = np.zeros(point, dtype=float)
#
#         kinetic_energy_id_sum, potential_energy_id_sum = 0, 0
#         vel_list = np.zeros(point, dtype=float)
#         for id in range(len(roots2)):
#             w_i = omega2[id]
#
#             y_i = (forms2[id] / D2[id]) * (
#                     np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])
#
#             y_list += y_i  # суммарная функция колебания
#
#             Q_t += (Q_lst[id] / D2[id]) * (
#                     np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])
#
#             # --------------------ENERGY------------------------------------
#             vel_i = (forms2[id] / D2[id]) * (
#                     -w_i * np.sin(w_i * t_loc) * integ_u[id] + np.cos(w_i * t_loc) * integ_u_dif[id])
#             vel_list += vel_i
#
#             kinetic_energy_id = 0.5 * (ro * S) * (dl * sum(vel_i ** 2))
#             kinetic_energy_id_sum += kinetic_energy_id
#
#             y_i_second_dif = (form_barrier_second_dif_lst[id] / D2[id]) * (
#                         np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])
#
#             potential_energy_id = 0.5 * E * J_inertia * (dl * sum(y_i_second_dif ** 2))
#             potential_energy_id_sum += potential_energy_id
#
#             total_energy_id = kinetic_energy_id + potential_energy_id
#             total_energy[id] = total_energy_id
#             # --------------------------------------------------------------
#
#         # если вышли из барьера, откатываем шаг назад и уменьшаем шаг на меньший из листа шагов
#         if (t_step_id > 0) and (Q_t >= 0):
#             t_loc -= t_step
#             t_global -= t_step
#             t_global_lst.pop()
#             Q_t = Q_t_previous
#             t_loc_lst.pop()
#             y_list = y_list_previous.copy()
#
#             t_step_id -= 1
#
#             continue
#
#         if (t_step_id < len(t_step_lst)-1) and (Q_t < 0):
#             t_step_id += 1
#         # --------------------------------------------------------------
#
#         if first_step:
#             # print('VI, kin = ')
#             # print(kinetic_energy_id_sum)
#             # print('VI, pot = ')
#             # print(potential_energy_id_sum)
#
#             with open('./plots/vel_lst_VI.txt', 'w') as cur_file:
#                 cur_file.write(str(vel_list))
#             with open('./plots/dis_lst_VI.txt', 'w') as cur_file:
#                 cur_file.write(str(y_list))
#             # pass
#             print('log pause')
#
#         first_step = False
#
#         # global_en_lst.append(sum(total_energy))
#         global_en_lst.append(kinetic_energy_id_sum + potential_energy_id_sum)
#         kinetic_energy_id_sum_lst.append(kinetic_energy_id_sum)
#         potential_energy_id_sum_lst.append(potential_energy_id_sum)
#
#         Q_list_time.append(Q_t)
#
#         y_barrier_global.append(y_list[point_barrier])
#         y_end_global.append(y_list[-1])
#
#         if Q_t < 0:
#             graph(y_list, vel_list)
#
#     # ----------------------------------------------------
#     # # последний шаг откатываем, поскольку Q>0 мы не хотим
#     # # if t_loc > 2e-8:
#     # y_list = y_list_previous
#     t_loc -= t_step
#     t_global -= t_step
#     t_global_lst.pop()
#     Q_list_time.pop()
#     y_barrier_global.pop()
#     y_end_global.pop()
#     global_en_lst.pop()
#     kinetic_energy_id_sum_lst.pop()
#     potential_energy_id_sum_lst.pop()
#     t_loc_lst.pop()
#     # # ----------------------------------------------------
#
#     vel_list = np.zeros(point, dtype=float)
#     for id in range(len(roots2)):
#         w_i = omega2[id]
#         y_i = (forms2[id] / D2[id]) * (
#                 np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])
#         y_list += y_i  # суммарная функция колебания
#         vel_i = (forms2[id] / D2[id]) * (
#                     -w_i * np.sin(w_i * t_loc) * integ_u[id] + np.cos(w_i * t_loc) * integ_u_dif[id])
#         vel_list += vel_i  # суммарная функция колебания
#
#     if len(t_loc_lst) == 0:
#         return disp_start, vel_start, t_loc_lst
#
#     # y_list[point_barrier] = 0
#     # ПЕРЕКЛЮЧАЕМ НА БАЛКУ БЕЗ БАРЬЕРА
#     # beam_no_VI_vibrations(y_list, vel_list)
#     return y_list, vel_list, t_loc_lst


# Задачем начальные условия
def initial_conditions():
    disp_start = np.zeros(Nx)
    vel_start = np.sin(np.pi * x_vals) + 2 * np.sin(3 * np.pi * x_vals)

    return disp_start, vel_start, max(disp_start)


def graph(y_list, vel_list):
    # pass
    # if len(t_global_lst) % 200 == 0:
    if round(t_global * 1e5) % 200 == 0:
        # print(total_energy)
        # print(sum(total_energy))

        fig.suptitle('Time=' + str('%.2f' % t_global) + 's=' + str('%.2f' % (t_global * 1e3)) + 'ms')

        axs[0][0].set_title('Beam shape')
        axs[0][0].plot(x_vals, y_list, 'k', linewidth=1)
        axs[0][0].grid()
        # axs[0][0].set_ylim(-max_disp_start*1.0, max_disp_start*1.0)

        axs[2][1].set_title('Beam vel')
        # axs[2][1].plot(np.linspace(0, l2, point), vel_list, 'k', linewidth=1)
        axs[2][1].plot(x_vals, vel_list, 'g', linewidth=1)
        axs[2][1].grid()

        plt.pause(0.01)
        axs[0][0].clear()
        axs[1][0].clear()
        axs[0][1].clear()
        axs[1][1].clear()
        axs[2][0].clear()
        axs[2][1].clear()


# -----------------------------------------------
pi = np.pi
Nx = 1000 + 1    # Number of x points
x_vals = np.linspace(0, 1, Nx)
a_bar = 0.5
point_barrier = round((Nx - 1) * a_bar)

L = 1
# Параметры задачи
T = 3.0  # Время моделирования
c = 1.0  # Скорость волны
dx = L / (Nx - 1)
# ------------------------------------------------
t_global = 0.0
t_global_lst = []

t_step_lst = [1e-4, 1e-4, 1e-4]  # лист временных шагов, которые будем динамически менять
t_step_id = 0
t_step = t_step_lst[t_step_id]

t_end = 0.55
time_is_up = False  # флаг на то, закончилось ли время
# ---------------------------------------------

print('Начинаем фигачить')

# Задаем начальную деформацию и скорость балки
[disp_start, vel_start, max_disp_start] = initial_conditions()
print('Задали НУ')

print('Запускаем динамику')

fig, axs = plt.subplots(3, 2, squeeze=False)  # создаем саб плот из 2 графиков
# plt.subplots_adjust(wspace=0.4, hspace=0.7)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.pause(2)

# while True:
while not time_is_up:
    print('Time = ', str(t_global))
    # Вначале запускаем динамику балки без барьера
    [disp_start, vel_start, t_loc_lst] = beam_no_VI_vibrations(x_vals, disp_start, vel_start)
    print('no VI = ' + str(len(t_loc_lst)))
    #
    # print('Time = ', str(t_global))
    # [disp_start, vel_start, t_loc_lst] = beam_with_VI_vibrations(disp_start, vel_start)
    # print('VI = ' + str(len(t_loc_lst)))
    # # print(t_loc_lst)


