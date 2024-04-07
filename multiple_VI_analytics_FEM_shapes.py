from sympy import *
import pylab as plt
import pandas as pd
from class_beam_elements import Class_beam_elements
from create_global_matrix import *


def shape_interpolation(roots, natural_shapes):
    shape_interpolation_lst = []
    x_lst = np.linspace(0, l2, point)
    for i in range(len(roots)):
        k = roots[i]
        natural_shape = natural_shapes[i]

        i_0 = [0, round(len(natural_shape) / 3), round(len(natural_shape) * 2 / 3), -1]
        [x1, x2, x3, x4] = [x_lst[i_0[0]], x_lst[i_0[1]], x_lst[i_0[2]], x_lst[i_0[3]]]
        [y1, y2, y3, y4] = [natural_shape[i_0[0]], natural_shape[i_0[1]], natural_shape[i_0[2]], natural_shape[i_0[3]]]

        sin_cos_tensor = np.array([[np.sin(k * x1), np.cos(k * x1), np.sinh(k * x1), np.cosh(k * x1)],
                          [np.sin(k * x2), np.cos(k * x2), np.sinh(k * x2), np.cosh(k * x2)],
                          [np.sin(k * x3), np.cos(k * x3), np.sinh(k * x3), np.cosh(k * x3)],
                          [np.sin(k * x4), np.cos(k * x4), np.sinh(k * x4), np.cosh(k * x4)]])

        y_tensor = np.array([[y1], [y2], [y3], [y4]])

        A_tensor = np.matmul(np.linalg.inv(sin_cos_tensor), y_tensor)
        print(A_tensor)

        shape_interpolation_i = lambda x, kk=k: A_tensor[0][0] * np.sin(kk * x) + A_tensor[1][0] * np.cos(kk * x) + A_tensor[2][0] * np.sinh(kk * x) + A_tensor[3][0] * np.cosh(kk * x)
        shape_interpolation_lst.append(shape_interpolation_i)


    # plt.figure()
    print(roots)
    print(f'k = {roots[0]}')
    print(shape_interpolation_lst)
    print(shape_interpolation_lst[0])
    print(len(shape_interpolation_lst))
    print()
    for i in range(len(roots)):
        plt.figure()
        plt.plot(x_lst, [shape_interpolation_lst[i](x) for x in x_lst])
        # plt.plot(x_lst, natural_shapes[i])

    # plt.plot(x_lst, [shape_interpolation[6](x) for x in x_lst])
    # plt.plot(x_lst, shape_interpolation[0])
    # plt.plot([0, 1], [(shape_interpolation[0](x)) for x in [0, 1]])
    plt.show()

    return shape_interpolation_lst



# определяем частоты, формы и коэф (D = form ** 2) для балки без барьера
def beam_without_barrier():
    # -------------------------------------------------------------------------------------
    cur_path = r'./plots/saving_beam_shapes_noVI.txt'
    [roots1, natural_shapes1] = reading_beam_shapes(cur_path)
    omega1 = []
    for roots1_i in roots1:
        omega1.append(np.sqrt(E * J_inertia / ro / S) * (roots1_i ** 2))
    print(f'Рассматриваем частот {len(roots1)} без ВИ')
    print(f'Максимальная частота без ВИ = {roots1[-1]}')
    # -------------------------------------------------------------------------------------

    # попытка из полученных численно форм вывести аналитическое уравнение, к которому применимы аналитические методы
    # сложность, что значения аргументов в гиперболических функциях остаются большими при высоких частотах, что ведет к неправильному виду полученной формы
    # shape_interpolation(roots1, natural_shapes1)


    print('Freq no VI = ' + str(roots1))
    D1 = []
    forms1 = []
    form1_second_dif = []
    for i, root in enumerate(roots1):
        alpha_i = root / l2

        form1_second_dif.append(np.array([0] * point))

        v1v2 = np.array(natural_shapes1[i])
        forms1.append(v1v2)

        integ_v1v2_2 = 0
        for ii in range(1, len(v1v2)):
            integ_v1v2_2 += (v1v2[ii] ** 2 + v1v2[ii - 1] ** 2) / 2
        integ_v1v2_2 *= dl

        D1.append(integ_v1v2_2)

    return roots1, D1, forms1, omega1, form1_second_dif


# динамика балки без барьера
def beam_no_VI_vibrations(disp_start, vel_start):
    t_loc_lst = []
    print('NO barrier')
    is_first_no_VI = True
    global t_global, y_barrier_global, Q_list_time, total_energy, y_end_global, time_is_up, t_global_lst, global_en_lst, kinetic_energy_id_sum_lst, potential_energy_id_sum_lst
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

        # вручную убираем все формы кроме первой, поскольку возюуждаем чисто по первой
        # integ_u = np.array([integ_u[0]] + [0] * (len(integ_u) - 1))

    print(f'integ_u = {integ_u}')
    print(f'integ_u_dif = {integ_u_dif}')

    t_loc = 0
    y_list = disp_start

    first_step = True

    t_step_id = 0
    while ((y_list[point_barrier] >= 0) or (first_step)) and (not time_is_up):  # ???????????????????????
        t_step = t_step_lst[t_step_id]
        first_step = False
        t_loc += t_step
        t_global += t_step
        t_global_lst.append(t_global)
        t_loc_lst.append(t_loc)

        if t_global >= t_end:
            time_is_up = True

        y_list_previous = y_list.copy()

        y_list = np.zeros(point, dtype=float)
        vel_list = np.zeros(point, dtype=float)

        kinetic_energy_id_sum, potential_energy_id_sum = 0, 0
        for id in range(len(roots1)):
            w_i = omega1[id]

            y_i = (forms1[id] / D1[id]) * (np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])

            y_list += y_i  # суммарная функция колебания

            # --------------------ENERGY------------------------------------
            vel_i = (forms1[id] / D1[id]) * (-w_i * np.sin(w_i * t_loc) * integ_u[id] + np.cos(w_i * t_loc) * integ_u_dif[id])
            vel_list += vel_i
            kinetic_energy_id = 0.5 * (ro * S) * (dl * sum(vel_i ** 2))
            kinetic_energy_id_sum += kinetic_energy_id

            y_i_second_dif = (form1_second_dif[id] / D1[id]) * (np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])
            potential_energy_id = 0.5 * E * J_inertia * (dl * sum(y_i_second_dif ** 2))
            potential_energy_id_sum += potential_energy_id

            total_energy_id = kinetic_energy_id + potential_energy_id
            total_energy[id] = total_energy_id
            # --------------------------------------------------------------

        # если зашли в барьер, откатываем шаг назад и уменьшаем шаг на меньший из листа шагов
        if (t_step_id > 0) and (y_list[point_barrier] < 0):
            t_loc -= t_step
            t_global -= t_step
            t_global_lst.pop()
            y_list = y_list_previous.copy()
            t_loc_lst.pop()

            t_step_id -= 1

            continue

        if (t_step_id < len(t_step_lst)-1) and (y_list[point_barrier] > 0):
            t_step_id += 1


        # if is_first_no_VI:
        #     print('no VI, kin = ')
        #     print(kinetic_energy_id_sum)
        #     print('no VI, pot = ')
        #     print(potential_energy_id_sum)
        # is_first_no_VI = True

        # global_en_lst.append(sum(total_energy))
        global_en_lst.append(kinetic_energy_id_sum + potential_energy_id_sum)
        kinetic_energy_id_sum_lst.append(kinetic_energy_id_sum)
        potential_energy_id_sum_lst.append(potential_energy_id_sum)

        y_barrier_global.append(y_list[point_barrier])
        y_end_global.append(y_list[-1])

        Q_list_time.append(0)
        graph(y_list, vel_list)

    # vel_list = np.zeros(point, dtype=float)
    # for id in range(len(roots1)):
    #     w_i = omega1[id]
    #     vel_i = (forms1[id] / D1[id]) * (-w_i * np.sin(w_i * t_loc) * integ_u[id] + np.cos(w_i * t_loc) * integ_u_dif[id])
    #     vel_list += vel_i  # суммарная функция колебания

    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ С БАРЬЕРОМ
    # beam_with_VI_vibrations(y_list, vel_list)
    with open('./plots/vel_lst_no_VI.txt', 'w') as cur_file:
        # cur_file.write(str(vel_list))
        for item in vel_list:
            cur_file.write(f"{item} ")

    with open('./plots/dis_lst_no_VI.txt', 'w') as cur_file:
        # cur_file.write(str(y_list))
        for item in y_list:
            cur_file.write(f"{item} ")

    print('Interection = ' + str(y_list[point_barrier]) + ' step_id = ' + str(t_step_id))
    return y_list, vel_list, t_loc_lst


def reading_beam_shapes(path):
    with open(path, 'r') as cur_file:
        data = cur_file.readlines()

    roots = eval(data[0])
    natural_shapes = eval(data[1])

    return roots, natural_shapes


# определяем частоты, формы и коэф (D = form ** 2) для балки с барьером
def beam_with_barrier():
    A1_fun = lambda k: ((k*np.cos(k*l1)-k*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k*np.sin(k*l1)-k*np.sinh(k*l1)))

    B1_fun = lambda k: ((k*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)) -
          (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (-k*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)))

    # для нахождения силы взаимодействия с барьером необходимо найти разрыв 3ей производной формы балки в точке барьера
    # третья производная формы левой части балки в точке барьера
    A3_fun = lambda k: ((-k**3*np.cos(k*l1)-k**3*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(k**3*np.sin(k*l1)-k**3*np.sinh(k*l1)))
    # третья производная формы правой части балки в точке барьера
    B3_fun = lambda k: ((-k**3*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k**3*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k**3*np.sinh(k*l1)) -
          (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (k**3*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k**3*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k**3*np.sinh(k*l1)))


    # -------------------------------------------------------------------------------------
    cur_path = r'./plots/saving_beam_shapes.txt'
    [roots2, natural_shapes2] = reading_beam_shapes(cur_path)
    omega2 = []
    for roots2_i in roots2:
        omega2.append(np.sqrt(E * J_inertia / ro / S) * (roots2_i ** 2))
    print(f'Рассматриваем частот {len(roots2)} ВИ')
    print(f'Максимальная частота ВИ = {roots2[-1]}')
    # -------------------------------------------------------------------------------------

    D2 = []
    forms2 = []
    form_barrier_second_dif_lst = []  # store second diff beam shape

    for i in range(len(roots2)):
        k = roots2[i]

        v1v2 = np.array(natural_shapes2[i])
        forms2.append(v1v2)

        integ_v1v2_2 = 0
        for ii in range(1, len(v1v2)):
            integ_v1v2_2 += (v1v2[ii] ** 2 + v1v2[ii - 1] ** 2) / 2
        integ_v1v2_2 *= dl

        D2.append(integ_v1v2_2)

        form_barrier_second_dif_lst.append(np.array([0] * point))


    Q_lst = [(A3_fun(roots2[ii]) - A1_fun(roots2[ii]) / B1_fun(roots2[ii]) * B3_fun(roots2[ii])) for ii in range(len(roots2))]

    print('D2=')
    print(D2)
    print('\n')

    return roots2, D2, forms2, omega2, Q_lst, form_barrier_second_dif_lst


# Динамика балки с барьером
def beam_with_VI_vibrations(disp_start, vel_start):
    t_loc_lst = []
    print('with barrier')

    is_first_no_VI = True
    global t_global, y_barrier_global, Q_list_time, y_end_global, time_is_up, t_global_lst, global_en_lst, kinetic_energy_id_sum_lst, potential_energy_id_sum_lst
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

    Q_t = -1
    t_step_id = 0
    first_step = True
    while ((Q_t < 0) and (not time_is_up)): # or first_step:  # ???????????????????????
        # first_step = False
        t_step = t_step_lst[t_step_id]
        Q_t_previous = Q_t
        Q_t = 0
        t_loc += t_step
        t_global += t_step
        t_global_lst.append(t_global)
        t_loc_lst.append(t_loc)

        if t_global >= t_end:
            time_is_up = True

        y_list_previous = y_list.copy()
        y_list = np.zeros(point, dtype=float)

        kinetic_energy_id_sum, potential_energy_id_sum = 0, 0
        vel_list = np.zeros(point, dtype=float)
        for id in range(len(roots2)):
            w_i = omega2[id]

            y_i = (forms2[id] / D2[id]) * (
                    np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])

            y_list += y_i  # суммарная функция колебания

            Q_t += (Q_lst[id] / D2[id]) * (
                    np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])

            # --------------------ENERGY------------------------------------
            vel_i = (forms2[id] / D2[id]) * (
                    -w_i * np.sin(w_i * t_loc) * integ_u[id] + np.cos(w_i * t_loc) * integ_u_dif[id])
            vel_list += vel_i

            kinetic_energy_id = 0.5 * (ro * S) * (dl * sum(vel_i ** 2))
            kinetic_energy_id_sum += kinetic_energy_id

            y_i_second_dif = (form_barrier_second_dif_lst[id] / D2[id]) * (
                        np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])

            potential_energy_id = 0.5 * E * J_inertia * (dl * sum(y_i_second_dif ** 2))
            potential_energy_id_sum += potential_energy_id

            total_energy_id = kinetic_energy_id + potential_energy_id
            total_energy[id] = total_energy_id
            # --------------------------------------------------------------

        # если вышли из барьера, откатываем шаг назад и уменьшаем шаг на меньший из листа шагов
        if (t_step_id > 0) and (Q_t >= 0):
            t_loc -= t_step
            t_global -= t_step
            t_global_lst.pop()
            Q_t = Q_t_previous
            t_loc_lst.pop()
            y_list = y_list_previous.copy()

            t_step_id -= 1

            continue

        if (t_step_id < len(t_step_lst)-1) and (Q_t < 0):
            t_step_id += 1
        # --------------------------------------------------------------

        if first_step:
            # print('VI, kin = ')
            # print(kinetic_energy_id_sum)
            # print('VI, pot = ')
            # print(potential_energy_id_sum)

            with open('./plots/vel_lst_VI.txt', 'w') as cur_file:
                # cur_file.write(str(vel_list))
                for item in vel_list:
                    cur_file.write(f"{item} ")

            with open('./plots/dis_lst_VI.txt', 'w') as cur_file:
                # cur_file.write(str(y_list))
                for item in y_list:
                    cur_file.write(f"{item} ")

            print('log pause')

        first_step = False

        # global_en_lst.append(sum(total_energy))
        global_en_lst.append(kinetic_energy_id_sum + potential_energy_id_sum)
        kinetic_energy_id_sum_lst.append(kinetic_energy_id_sum)
        potential_energy_id_sum_lst.append(potential_energy_id_sum)

        Q_list_time.append(Q_t)

        y_barrier_global.append(y_list[point_barrier])
        y_end_global.append(y_list[-1])

        if Q_t < 0:
            graph(y_list, vel_list)

    # ----------------------------------------------------
    # # последний шаг откатываем, поскольку Q>0 мы не хотим
    # # if t_loc > 2e-8:
    # y_list = y_list_previous
    t_loc -= t_step
    t_global -= t_step
    t_global_lst.pop()
    Q_list_time.pop()
    y_barrier_global.pop()
    y_end_global.pop()
    global_en_lst.pop()
    kinetic_energy_id_sum_lst.pop()
    potential_energy_id_sum_lst.pop()
    t_loc_lst.pop()
    # # ----------------------------------------------------

    vel_list = np.zeros(point, dtype=float)
    for id in range(len(roots2)):
        w_i = omega2[id]
        y_i = (forms2[id] / D2[id]) * (
                np.cos(w_i * t_loc) * integ_u[id] + np.sin(w_i * t_loc) / w_i * integ_u_dif[id])
        y_list += y_i  # суммарная функция колебания
        vel_i = (forms2[id] / D2[id]) * (
                    -w_i * np.sin(w_i * t_loc) * integ_u[id] + np.cos(w_i * t_loc) * integ_u_dif[id])
        vel_list += vel_i  # суммарная функция колебания

    if len(t_loc_lst) == 0:
        return disp_start, vel_start, t_loc_lst

    # y_list[point_barrier] = 0
    # ПЕРЕКЛЮЧАЕМ НА БАЛКУ БЕЗ БАРЬЕРА
    # beam_no_VI_vibrations(y_list, vel_list)
    return y_list, vel_list, t_loc_lst


# Задачем начальные условия
def initial_conditions():
    natural_fr = True
    if natural_fr:
        # выбираем начальное перемещение в точности по собственной форме
        if forms1[0][-1] > 0:
            norm_scale = 1000
        else:
            norm_scale = -1000
        disp_start = forms1[0] / norm_scale
        vel_start = np.zeros(point, dtype=float)
    else:
        # ---------------------------- НУ из МКЭ ---------------------
        # MaxNode = 200 + 1
        MaxNode = point
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


def graph(y_list, vel_list):
    # pass
    if len(Q_list_time) % 500 == 0:
        # print(total_energy)
        # print(sum(total_energy))

        fig.suptitle('Time=' + str('%.2f' % t_global) + 's=' + str('%.2f' % (t_global * 1e3)) + 'ms')

        axs[0][0].set_title('Beam shape')
        axs[1][0].set_title(f'VI force, {-Q_list_time[-1] * E * J_inertia}')
        axs[1][1].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10)

        axs[0][0].plot(np.linspace(0, l2, point), y_list, 'k', linewidth=1)
        axs[0][0].plot([l1], [0], 'ro', linewidth=2)
        axs[0][0].grid()
        # axs[0][0].set_ylim(-max_disp_start*1.0, max_disp_start*1.0)

        Q_list_time_format = -np.array(Q_list_time) * E * J_inertia
        # axs[1][0].plot(t_global_lst, Q_list_time_format, 'k', linewidth=1)
        axs[1][0].plot(t_global_lst, np.array(potential_energy_id_sum_lst), 'g', linewidth=1)
        axs[1][0].grid()

        axs[1][1].plot(t_global_lst, y_barrier_global, 'g', linewidth=1)
        axs[1][1].plot(t_global_lst, y_end_global, 'k', linewidth=1)
        axs[1][1].plot(t_global_lst, [0] * len(y_end_global), 'r--', linewidth=1)
        axs[1][1].grid()

        axs[2][1].set_title('Beam vel')
        axs[2][1].plot(np.linspace(0, l2, point), vel_list, 'k', linewidth=1)
        # axs[2][1].plot(t_global_lst, np.array(kinetic_energy_id_sum_lst), 'g', linewidth=1)
        axs[2][1].grid()

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
        # -------------------- Global energy ----------------------
        axs[2][0].set_title(f'Full energy, {global_en_lst[-1]}')
        axs[2][0].plot(t_global_lst, global_en_lst, 'g', linewidth=1)
        axs[2][0].plot(t_global_lst, [0]*len(t_global_lst), 'r--', linewidth=1)
        # axs[2][0].plot(t_global_lst, np.array(potential_energy_id_sum_lst) + np.array(kinetic_energy_id_sum_lst), 'g', linewidth=1)
        axs[2][0].grid()
        # ---------------------------------------------------------

        plt.pause(0.01)
        axs[0][0].clear()
        axs[1][0].clear()
        axs[0][1].clear()
        axs[1][1].clear()
        axs[2][0].clear()
        axs[2][1].clear()


# -----------------------------------------------
ro = 7850
E = 2e11
a = 10e-3
S = a * a  # площадь сечения балки
J_inertia = a * a ** 3 / 12  # момент инерции сечения
ro = 7850  # плотность
E = 2e11  # модуль Юнга
# ------------------------------------------------
point = 1000 + 1  # количество элементов балки
l1 = 0.8  # местоположение барьера
l2 = 1  # длина балки
dl = l2 / (point - 1)
point_barrier = round((point - 1) * l1)
# ------------------------------------------------
t_global = 0.0
t_global_lst = []

t_step_lst = [1e-9, 1e-8, 1e-7, 1e-6]  # лист временных шагов, которые будем динамически менять
t_step_id = 0
t_step = t_step_lst[t_step_id]

t_end = 0.55
time_is_up = False  # флаг на то, закончилось ли время
# ---------------------------------------------

y_barrier_global = []
y_end_global = []
Q_list_time = []
global_en_lst = []

kinetic_energy_id_sum_lst = []
potential_energy_id_sum_lst = []

print('Начинаем фигачить')

# Определяем параметры балки без барьера
[roots1, D1, forms1, omega1, form1_second_dif] = beam_without_barrier()
print('Определили параметры балки без барьера')

total_energy = [0] * len(roots1)  # Лист energy мод

# Определяем параметры балки с барьером
[roots2, D2, forms2, omega2, Q_lst, form_barrier_second_dif_lst] = beam_with_barrier()
print('Определили параметры балки с барьером')

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
    [disp_start, vel_start, t_loc_lst] = beam_no_VI_vibrations(disp_start, vel_start)
    print('no VI = ' + str(len(t_loc_lst)))

    print('Time = ', str(t_global))
    [disp_start, vel_start, t_loc_lst] = beam_with_VI_vibrations(disp_start, vel_start)
    print('VI = ' + str(len(t_loc_lst)))
    # print(t_loc_lst)

# print('Time = ', str(t_global))
# # Вначале запускаем динамику балки без барьера
# [disp_start, vel_start, t_loc_lst] = beam_no_VI_vibrations(disp_start, vel_start)
# print('no VI = ' + str(len(t_loc_lst)))
#
# while True:
#     print('Time = ', str(t_global))
#     [disp_start, vel_start, t_loc_lst] = beam_with_VI_vibrations(disp_start, vel_start)
#     print('VI = ' + str(len(t_loc_lst)))
#     # print(t_loc_lst)


Q_list_time_format = -np.array(Q_list_time) * E * J_inertia
file_name = 'Analytics_write_VI_force.txt'
with open(r'./plots/' + file_name, 'w') as cur_file:
    cur_file.write(str(list(Q_list_time_format)))

write_time = np.linspace(0, t_global, len(Q_list_time_format))
file_name = 'Analytics_write_time.txt'
with open(r'./plots/' + file_name, 'w') as cur_file:
    cur_file.write(str(list(t_global_lst)))

file_name = 'Analytics_write_disp_end.txt'
with open(r'./plots/' + file_name, 'w') as cur_file:
    cur_file.write(str(list(y_end_global)))

file_name = 'Analytics_write_disp_barrier.txt'
with open(r'./plots/' + file_name, 'w') as cur_file:
    cur_file.write(str(list(y_barrier_global)))
