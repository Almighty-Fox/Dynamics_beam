import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# находим собственные частоты и формы консольной балки
def beam_without_barrier():
    al_l = np.arange(0, 25, 0.00001)
    # al_l = np.arange(0, 30, 0.00001)
    fun1 = np.sinh(al_l) * np.cos(al_l) + 1
    roots1 = []
    omegas1 = []
    omegas2 = []
    for i in range(len(fun1) - 1):
        if fun1[i] * fun1[i+1] < 0:
            roots1_i = (al_l[i] + al_l[i+1]) / 2
            roots1.append(roots1_i)

    roots1 = np.array(roots1)
    omegas1 = roots1 ** 2 / l_length ** 2 * np.power((E_young * J_inertia / ro_density / F_section), (1 / 2))
    omegas2 = E_young * J_inertia / ro_density / F_section * roots1 ** 4 / l_length ** 4

    print('no VI om2 = ', str(omegas2))
    print('original roots1 = ' + str(roots1 / l_length))
    print(f'len(roots1) = {len(roots1)}')
    print(f'omegas1 = {omegas1}')

    print(f'Period = { 2 * np.pi / omegas1}c')

    D1 = []
    forms1 = []
    form1_second_dif = []
    for i, root in enumerate(roots1):
        alpha_i = root / l_length

        u_i = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                 np.cosh(alpha_i * z) - np.cos(alpha_i * z)))
        form1_i = np.array([u_i(ii) for ii in np.linspace(0, l_length, points)])
        forms1.append(form1_i)

        u_i_kv = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                    np.cosh(alpha_i * z) - np.cos(alpha_i * z))) ** 2
        D1_i = integrate.quad(u_i_kv, 0, l_length)[0]
        D1.append(D1_i)

    return roots1 / l_length, D1, forms1, omegas1, omegas2, form1_second_dif


def det_AB_before_VI(disp_start, vel_start):
    A_lst = np.zeros(len(omegas1), dtype=float)
    for i in range(len(omegas1)):
        integ_body = forms1[i] * disp_start
        for ii in range(1, len(integ_body)):
            A_lst[i] += (integ_body[ii] + integ_body[ii - 1]) / 2
        A_lst[i] /= D1[i]
    A_lst *= dl
    # for i in range(len(A_lst)):
    #     A_lst[i] = 0

    B_lst = np.zeros(len(omegas1), dtype=float)
    for i in range(len(omegas1)):
        integ_body = forms1[i] * vel_start
        for ii in range(1, len(integ_body)):
            B_lst[i] += (integ_body[ii] + integ_body[ii - 1]) / 2
        B_lst[i] /= (D1[i] * omegas1[i])
    B_lst *= dl
    # for i in range(1, len(B_lst)):
    #     B_lst[i] = 0

    return A_lst, B_lst


# динамика балки без барьера
def beam_no_VI_vibrations(disp_start, vel_start):
    print('NO barrier')
    time_is_up = False
    y_list, vel_list = disp_start, vel_start

    global Pq_global, y_end_global, time_global

    [A_lst, B_lst] = det_AB_before_VI(disp_start, vel_start)

    t_loc = 0

    first_step = True

    # while ((y_list[point_barrier] >= 0) or (first_step)) and (not time_is_up):
    # while True:
    while y_list[point_barrier] > 1:
        first_step = False
        t_loc += tau
        # t_loc += 1e-4
        time_global.append(time_global[-1] + tau)

        if time_global[-1] >= t_end:
            time_is_up = True


        y_list = np.zeros(points, dtype=float)
        vel_list = np.zeros(points, dtype=float)

        for k in range(len(omegas1)):
            y_list_add = forms1[k] * (
                    A_lst[k] * np.cos(omegas1[k] * t_loc) + B_lst[k] * np.sin(omegas1[k] * t_loc))
            y_list += y_list_add

            vel_list_add = forms1[k] * (
                    -A_lst[k] * omegas1[k] * np.sin(omegas1[k] * t_loc) + B_lst[k] * omegas1[k] * np.cos(omegas1[k] * t_loc))
            vel_list += vel_list_add

        Pq_global.append(0)

        graph(y_list)


    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ С БАРЬЕРОМ
    # beam_with_VI_vibrations(y_list, vel_list)

    return y_list, vel_list, A_lst, B_lst, t_loc



def D_const_delta_function(A_lst, B_lst, first_tau=True, time_free=0):
    if first_tau:
        slag_free = 0
        for k in range(len(omegas1)):
            slag_free_add = forms1_barrier[k] * (
                    A_lst[k] * np.cos(omegas1[k] * tau) + B_lst[k] * np.sin(omegas1[k] * tau))
            slag_free += slag_free_add

        slag_koef = l_length / delta2 / F_section / ro_density
        slag_delta = 0
        for k in range(len(omegas1)):
            slag_delta_add = 1 / rk_lst[k]**2 * forms1_barrier[k]**2 / D1[k]**2 * np.sin(omegas1[k] * tau)
            slag_delta += slag_delta_add

    else:
        slag_free = 0
        for k in range(len(omegas1)):
            slag_free_add = forms1_barrier[k] * (
                    A_lst[k] * np.cos(omegas1[k] * time_free) + B_lst[k] * np.sin(omegas1[k] * time_free))
            slag_free += slag_free_add

        slag_koef = l_length / delta2 / F_section / ro_density
        slag_delta = 0
        for k in range(len(omegas1)):
            slag_delta_add = 1 / rk_lst[k] ** 2 * forms1_barrier[k] ** 2 / D1[k] ** 2 * np.sin(omegas1[k] * tau)
            slag_delta += slag_delta_add

    return -slag_free / slag_koef / slag_delta



# Динамика балки с барьером
def beam_with_VI_vibrations(A_lst, B_lst, t_loc_free):
    print('with barrier')
    time_is_up = False

    if t_loc_free == 0:
        D_const_delta = D_const_delta_function(A_lst, B_lst)
    else:
        D_const_delta = D_const_delta_function(A_lst, B_lst, False, t_loc_free+tau)

    global Pq_global, y_end_global, time_global

    # ------------------------------
    disp_slag_free_shape = np.zeros(points)
    vel_slag_free_shape = np.zeros(points)
    for k in range(len(omegas1)):
        disp_slag_free_shape_add = forms1[k] * (
                A_lst[k] * np.cos(omegas1[k] * t_loc_free) + B_lst[k] * np.sin(omegas1[k] * t_loc_free))
        disp_slag_free_shape += disp_slag_free_shape_add

        vel_slag_free_shape_add = forms1[k] * (
                -A_lst[k] * omegas1[k] * np.sin(omegas1[k] * t_loc_free) + B_lst[k] * omegas1[k] * np.cos(
            omegas1[k] * t_loc_free))
        vel_slag_free_shape += vel_slag_free_shape_add

    disp_new = disp_slag_free_shape
    vel_new = vel_slag_free_shape
    # ------------------------------

    t_lst_loc = [0, tau]
    Pq_new = 1
    Pq_lst_loc = []

    slag_koef = l_length / delta2 / F_section / ro_density
    slag2 = slag_koef * sum([1 / rk_lst[ii]**2 / omegas1[ii] * forms1_barrier[ii]**2 / D1[ii]**2 * (1 - np.cos(omegas1[ii]*tau)) for ii in range(len(omegas1))])


    while ((Pq_new >= 0) and (not time_is_up)): # or first_step:
    # while True:
        # first_step = False
        t_loc_free += tau
        time_global.append(time_global[-1] + tau)


        if time_global[-1] >= t_end:
            time_is_up = True

        print(A_lst)
        print(B_lst)

        disp_slag_free = 0
        for k in range(len(omegas1)):
            disp_slag_free_add = forms1_barrier[k] * (
                    A_lst[k] * np.cos(omegas1[k] * t_loc_free) + B_lst[k] * np.sin(omegas1[k] * t_loc_free))
            disp_slag_free += disp_slag_free_add


        disp_slag_delta = 0
        for k in range(len(omegas1)):
            disp_slag_delta_add = 1 / rk_lst[k]**2 * forms1_barrier[k] ** 2 / D1[k] ** 2 * np.sin(omegas1[k] * t_lst_loc[-1])
            disp_slag_delta += disp_slag_delta_add

        disp_slag_delta *= D_const_delta * slag_koef

        disp_prev_sum = np.zeros(len(omegas1))
        vel_prev_sum = np.zeros(len(omegas1))
        for k in range(len(omegas1)):
            for i in range(len(Pq_lst_loc)):
                disp_prev_sum[k] += Pq_lst_loc[i] * np.sin(omegas1[k] * (t_lst_loc[-2] - t_lst_loc[i]))
                # vel_prev_sum[k] += -Pq_lst_loc[i] * omegas1[k] * np.cos(omegas1[k] * (t_lst_loc[-2] - t_lst_loc[i]))
                vel_prev_sum[k] += Pq_lst_loc[i] * omegas1[k] * np.cos(omegas1[k] * (t_lst_loc[-2] - t_lst_loc[i]))
            disp_prev_sum[k] *= tau
            vel_prev_sum[k] *= tau

        slag1 = slag_koef * sum([1 / rk_lst[ii]**2 * forms1_barrier[ii]**2 / D1[ii]**2 * disp_prev_sum[ii] for ii in range(len(omegas1))])
        Pq_new = (-disp_slag_free - disp_slag_delta - slag1) / slag2
        print(f'disp_list_free = {disp_slag_free}')
        print(f'disp_slag_delta = {disp_slag_delta}')
        print(f'slag1 = {slag1}')
        print(f'slag2 = {slag2}')
        print(f'Pq_new = {Pq_new}')

        Pq_lst_loc.append(Pq_new)
        Pq_global.append(Pq_new)

        disp_slag_free_shape = np.zeros(points)
        vel_slag_free_shape = np.zeros(points)
        for k in range(len(omegas1)):
            disp_slag_free_shape_add = forms1[k] * (
                    A_lst[k] * np.cos(omegas1[k] * t_loc_free) + B_lst[k] * np.sin(omegas1[k] * t_loc_free))
            disp_slag_free_shape += disp_slag_free_shape_add

            vel_slag_free_shape_add = forms1[k] * (
                    -A_lst[k] * omegas1[k] * np.sin(omegas1[k] * t_loc_free) + B_lst[k] * omegas1[k] * np.cos(omegas1[k] * t_loc_free))
            vel_slag_free_shape += vel_slag_free_shape_add

        disp_slag_delta_shape = np.zeros(points)
        vel_slag_delta_shape = np.zeros(points)
        for k in range(len(omegas1)):
            disp_slag_delta_shape_add = 1 / rk_lst[k] ** 2 * forms1[k] * forms1_barrier[k] / D1[k] ** 2 * np.sin(
                omegas1[k] * t_lst_loc[-1])
            disp_slag_delta_shape += disp_slag_delta_shape_add

            vel_slag_delta_shape_add = 1 / rk_lst[k] ** 2 * forms1[k] * forms1_barrier[k] / D1[k] ** 2 * omegas1[k] * np.cos(
                omegas1[k] * t_lst_loc[-1])
            vel_slag_delta_shape += vel_slag_delta_shape_add

        disp_slag_delta_shape *= D_const_delta * slag_koef
        vel_slag_delta_shape *= D_const_delta * slag_koef

        disp_slag_VI_shape = np.zeros(points)
        vel_slag_VI_shape = np.zeros(points)
        for k in range(len(omegas1)):
            disp_slag_VI_shape_add = 1 / rk_lst[k] ** 2 * forms1[k] * forms1_barrier[k] / D1[k] ** 2 * (
                        disp_prev_sum[k] + Pq_lst_loc[-1] / omegas1[k] * (1 - np.cos(omegas1[k] * tau)))
            disp_slag_VI_shape += disp_slag_VI_shape_add

            # vel_slag_VI_shape_add = 1 / rk_lst[k] ** 2 * forms1[k] * forms1_barrier[k] / D1[k] ** 2 * (
            #             vel_prev_sum[k] + Pq_lst_loc[-1] / omegas1[k] * (1 + omegas1[k] * np.sin(omegas1[k] * tau)))
            # vel_slag_VI_shape += vel_slag_VI_shape_add

            vel_slag_VI_shape_add = 1 / rk_lst[k] ** 2 * forms1[k] * forms1_barrier[k] / D1[k] ** 2 * (
                    vel_prev_sum[k] + Pq_lst_loc[-1] * np.sin(omegas1[k] * tau))
            vel_slag_VI_shape += vel_slag_VI_shape_add

        disp_slag_VI_shape *= slag_koef
        vel_slag_VI_shape *= slag_koef

        disp_new = disp_slag_free_shape + disp_slag_VI_shape + disp_slag_delta_shape
        y_end_global.append(disp_new[-1])

        vel_new = vel_slag_free_shape + vel_slag_VI_shape + vel_slag_delta_shape


        graph(disp_new)

        t_lst_loc.append(t_lst_loc[-1] + tau)

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
    return disp_new, vel_new


# Задачем начальные условия
def initial_conditions():
    disp_start = np.zeros(points)
    vel_start = -omegas1[0] * forms1[0] / 100000

    return disp_start, vel_start


def graph(disp_list):
    # pass
    if len(time_global) % 1 == 0:
        # print(total_energy)
        # print(sum(total_energy))

        fig.suptitle('Time=' + str('%.2f' % time_global[-1]) + 's=' + str('%.2f' % (time_global[-1] * 1e3)) + 'ms')

        axs[0][0].set_title('Beam shape')
        axs[1][0].set_title(f'VI force')
        axs[2][0].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10)

        axs[0][0].plot(np.linspace(0, l_length, points), disp_list, 'k', linewidth=1)
        axs[0][0].plot([l_barrier], [0], 'ro', linewidth=2)
        axs[0][0].grid()
        # axs[0][0].set_ylim(-max_disp_start*1.0, max_disp_start*1.0)

        axs[1][0].plot(time_global, Pq_global, 'g', linewidth=1)
        axs[1][0].grid()

        # axs[2][0].plot(time_global, y_end_global, 'g', linewidth=1)
        # axs[2][0].grid()


        plt.pause(0.01)
        axs[0][0].clear()
        axs[1][0].clear()
        axs[2][0].clear()



# -----------------------------------------------
E_young = 2e11
section_side = 10e-3
J_inertia = section_side ** 4 / 12
ro_density = 7850
l_length = 1
l_barrier = 0.5
F_section = section_side ** 2

points = 100 + 1
dl = l_length / (points - 1)
point_barrier = round((points - 1) * l_barrier)
Hertz_koef = 1
tau = 6e-6
# ------------------------------------------------

[alpha, D1, forms1, omegas1, omegas2, form1_second_dif] = beam_without_barrier()
forms1_barrier = [forms1[i][point_barrier] for i in range(len(omegas1))]
delta2 = E_young * J_inertia / ro_density / F_section
delta = np.power(delta2, 1/2)
rk_lst = np.sqrt(np.array(omegas1) * l_length / delta)
print('rk = {}'.format(rk_lst))
ak = rk_lst**2 * delta / l_length**2


t_end = 0.02
time_is_up = False  # флаг на то, закончилось ли время
# ---------------------------------------------


y_end_global = [0.]
Pq_global = [0.]
time_global = [0.]

print('Начинаем фигачить')

# Задаем начальную деформацию и скорость балки
disp_start, vel_start = initial_conditions()
print('Задали НУ')

print('Запускаем динамику')

fig, axs = plt.subplots(3, 1, squeeze=False)  # создаем саб плот из 2 графиков
# plt.subplots_adjust(wspace=0.4, hspace=0.7)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.pause(2)

A_lst_initial, B_lst_initial = det_AB_before_VI(disp_start, vel_start)


[disp_start, vel_start] = beam_with_VI_vibrations(A_lst_initial, B_lst_initial, t_loc_free=0)
# while True:
while not time_is_up:
    print('Time = ', str(time_global[-1]))
    # Вначале запускаем динамику балки без барьера
    [y_list, vel_list, A_lst, B_lst, t_loc_free] = beam_no_VI_vibrations(disp_start, vel_start)

    print('Time = ', str(time_global[-1]))
    [disp_start, vel_start] = beam_with_VI_vibrations(A_lst, B_lst, t_loc_free)

# Q_list_time_format = -np.array(Q_list_time) * E * J_inertia
# file_name = 'Analytics_write_VI_force.txt'
# with open(r'./plots/' + file_name, 'w') as cur_file:
#     cur_file.write(str(list(Q_list_time_format)))
#
# write_time = np.linspace(0, t_global, len(Q_list_time_format))
# file_name = 'Analytics_write_time.txt'
# with open(r'./plots/' + file_name, 'w') as cur_file:
#     cur_file.write(str(list(t_global_lst)))
#
# file_name = 'Analytics_write_disp_end.txt'
# with open(r'./plots/' + file_name, 'w') as cur_file:
#     cur_file.write(str(list(y_end_global)))
#
# file_name = 'Analytics_write_disp_barrier.txt'
# with open(r'./plots/' + file_name, 'w') as cur_file:
#     cur_file.write(str(list(y_barrier_global)))
