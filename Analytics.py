import numpy as np
from sympy import *
from scipy import integrate
import pylab as plt
import time


def find_u_i(alpha_l):  # для массива из собственных значений al*l = [1.875, 4.6941, 7.8548] ищем собственные частоты и формы
    w_sobst_i = (alpha_l / l)**2 * np.power((E * Jx / ro / F), (1/2))
    alpha_i = alpha_l / l
    # поиск контанты С4
    u_1_2_i = lambda z: ((np.cos(alpha_i * l) + np.cosh(alpha_i * l)) / (np.sin(alpha_i * l) + np.sinh(alpha_i * l)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                               np.cosh(alpha_i * z) - np.cos(alpha_i * z))) ** 2
    C4_2_i = 1 / integrate.quad(u_1_2_i, 0, l)[0] / ro / F * E * Jx
    C4_i = C4_2_i ** (1 / 2)
    # print(C4_i)
    # ---------------------
    # форма колебаний на первой собст частоте
    u_1_i = lambda z: C4_i * ((np.cos(alpha_i * l) + np.cosh(alpha_i * l)) / (np.sin(alpha_i * l) + np.sinh(alpha_i * l)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                  np.cosh(alpha_i * z) - np.cos(alpha_i * z)))

    # int_z_i, err = integrate.quad(u_1_i, 0, l)
    ans = integrate.quad(u_1_i, 0, l)
    int_z_i = ans[0]

    return w_sobst_i, alpha_i, C4_i, u_1_i, int_z_i


l = 1.0
ro = 7850
E = 2e11
a = 25e-3
b = a
F = a * b
Jx = a * b**3 / 12
P_0 = 100

point = 100  # количество элементов балки


w_sobst = (1.875 / l)**2 * np.power((E * Jx / ro / F), (1/2))
w_vnesh = 3 * w_sobst

P_0_2 = P_0 / E / Jx

step = 0.0005
time_end = 100
list_time = [0.]
y_end = [0.]
accel_end = [0.]  # массив ускорения конца балки


alpha_l_list = np.array([1.8864, 4.6941, 7.8548, 10.9955, 14.1372, 17.2788])
# alpha_l_list = np.array([1.875])
# alpha_l_list = np.array([1.8864, 4.6941])

step_integ = np.array([0., 0.])  # тут будем хранить значения двух интегралов
list_of_step_integ = np.array([step_integ for ii in range(len(alpha_l_list))])  # тут будет массив тех значений для всех собств частот

u_list_point = []  # массив собственных форм для массива точек балки
w_sobst_list = np.zeros(len(alpha_l_list), dtype=float)
u_list = [lambda:1 for iiii in range(len(alpha_l_list))]
int_z_list = np.zeros(len(alpha_l_list), dtype=float)
for id, alpha_l in enumerate(alpha_l_list):  # для каждой собственной частоты запоминаем набор параметров
    [w_sobst_i, alpha_i, C4_i, u_i, int_z_i] = find_u_i(alpha_l)
    w_sobst_list[id] = w_sobst_i
    u_list[id] = u_i
    int_z_list[id] = int_z_i
    u_list_point.append(np.array([u_i(ii) for ii in np.linspace(0, l, point)]))  # двумерный массив, внешний по собственным частотам, внутренний по координатам

y_list = np.zeros(point, dtype=float)

betta_i = np.zeros(len(alpha_l_list), dtype=float)

for id in range(len(alpha_l_list)):  # для первого шага по времени, пришлось для него отдельно расписать цикл, в общем цикле иначе было бы деление на ноль
    w_sobst_i = w_sobst_list[id]
    int_z_i = int_z_list[id]

    expr_1_i = lambda time: P_0_2 * int_z_i * cos(w_sobst_i * time)  # подынтегральная функция для поиска бэтта
    expr_2_i = lambda time: P_0_2 * int_z_i * sin(w_sobst_i * time)

    step_integ = np.array([integrate.quad(expr_1_i, 0, step)[0], integrate.quad(expr_2_i, 0, step)[0]])

    list_of_step_integ[id] = step_integ

    betta_i[id] = 1 / w_sobst_i * (sin(w_sobst_i * step) * step_integ[0] - cos(w_sobst_i * step) * step_integ[1])

    y_i = betta_i[id] * u_list_point[id]  # суммарная функция колебания

    y_list += y_i

fig, axs = plt.subplots(2)  # создаем саб плот из 2 графиков
plt.subplots_adjust(wspace=0.6, hspace=0.4)

# начинаем цикл по времени
for t in np.arange(2*step, time_end, step):
    start_time = time.time()  # для засечки времени одной итерации цикла
    try:
        y_list = np.zeros(point, dtype=float)
        for id in range(len(alpha_l_list)):
            w_sobst_i = w_sobst_list[id]
            u_i = u_list[id]
            int_z_i = int_z_list[id]

            period = 2 * pi / w_vnesh
            if t <= period * 2 / 4:  # ТУТ МЕНЯЕМ ДЛИТЕЛЬНОСТЬ ИМПУЛЬСА
                T_vnesh_2 = np.pi / w_vnesh / 2
                expr_1_i = lambda time: P_0_2 * int_z_i * cos(w_sobst_i * time)  # подынтегральная функция для поиска бэтта
                expr_2_i = lambda time: P_0_2 * int_z_i * sin(w_sobst_i * time)

            else:
                expr_1_i = lambda time: 0
                expr_2_i = lambda time: 0

            list_of_step_integ[id] += [integrate.quad(expr_1_i, t-step, t)[0], integrate.quad(expr_2_i, t-step, t)[0]]

            betta_i[id] = 1 / w_sobst_i * (sin(w_sobst_i * t) * list_of_step_integ[id][0]
                                            - cos(w_sobst_i * t) * list_of_step_integ[id][1])
            print(betta_i[id])
            y_i = betta_i[id] * u_list_point[id]  # суммарная функция колебания

            y_list += y_i

        # fig.suptitle('Time = ' + str("%.5g" % t))
        fig.suptitle('Time = ' + str("%.4f" % t))


        # вывод графика колебания конца балки
        y_end.append(-y_list[-1])
        list_time.append(t)
        axs[1].plot(list_time, y_end, 'g', linewidth=1)
        axs[1].set_title('Координата конца')
        # axs[1].set_ylim(-4e-4, 4e-4)
        print("%.5g" % t, "%.5g" % -y_list[-1])


        # вывод графика колебания балки

        axs[0].plot(np.linspace(0, l, point), y_list, 'r', linewidth=1)
        axs[0].set_title('Балка')
        axs[0].set_xlim(0, l)
        # axs[0].set_ylim(-1e-4, 1e-4)
        axs[0].set_ylim(-max(abs(np.array(y_end))), max(abs(np.array(y_end))))

        plt.pause(0.01)
        axs[0].clear()
        axs[1].clear()

        # print("--- %s seconds ---" % (time.time() - start_time))  # время одного шага цикла
        # print(time.time() - start_time)

    except:
        break