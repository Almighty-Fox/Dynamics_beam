import numpy as np
from sympy import *
from scipy import integrate
import pylab as plt
import time
import pandas as pd


def find_u_i(alpha_l):  # для массива из собственных значений al*l = [1.875, 4.6941, 7.8548] ищем собственные частоты и формы
    w_sobst_i = (alpha_l / l)**2 * np.power((E * Jx / ro / F), (1/2))
    alpha_i = alpha_l / l
    #
    # форма колебаний на первой собст частоте
    u_i = lambda z: ((np.cos(alpha_i * l) + np.cosh(alpha_i * l)) / (np.sin(alpha_i * l) + np.sinh(alpha_i * l)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                  np.cosh(alpha_i * z) - np.cos(alpha_i * z)))

    u_i_second_dif = lambda z: (
                (np.cos(alpha_i * l) + np.cosh(alpha_i * l)) / (np.sin(alpha_i * l) + np.sinh(alpha_i * l)) * (
                    -alpha_i**2 * np.sin(alpha_i * z) - alpha_i**2 * np.sinh(alpha_i * z)) + (
                            alpha_i**2 * np.cosh(alpha_i * z) + alpha_i**2 * np.cos(alpha_i * z)))

    u_i_kv = lambda z: ((np.cos(alpha_i * l) + np.cosh(alpha_i * l)) / (np.sin(alpha_i * l) + np.sinh(alpha_i * l)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                               np.cosh(alpha_i * z) - np.cos(alpha_i * z))) ** 2
    D_i = integrate.quad(u_i_kv, 0, l)[0]

    return w_sobst_i, alpha_i, D_i, u_i, u_i_second_dif


l = 1.0
ro = 7850
E = 2e11
a = 10e-3
b = a
F = a * b
Jx = a * b**3 / 12
P_0 = 100

point = 200 + 1  # количество элементов балки
loc = 0.9  # расположение барьера

# w_sobst = (1.875 / l)**2 * np.power((E * Jx / ro / F), (1/2))

P_0_2 = P_0 / E / Jx

step_plot = 30  # строим график каждые 20 шагов
step = 0.00001
time_end = 100
list_time = []
y_end = []
y_barrier = []


alpha_l_list = np.array([1.8864, 4.6941, 7.8548, 10.9955, 14.1372, 17.2788])
# alpha_l_list = np.array([1.8864])
# alpha_l_list = np.array([1.8864, 4.6941])
amplitude_mode_oscillations = [0] * len(alpha_l_list)  # Лист амплитуд колебаний мод
kin_en_mode_oscillations = [0] * len(alpha_l_list)  # Лист максимальных кин. эн. мод
total_energy = [0] * len(alpha_l_list)  # Лист energy мод

step_integ = np.array([0., 0.])  # тут будем хранить значения двух интегралов
list_of_step_integ = np.array([step_integ for ii in range(len(alpha_l_list))])  # тут будет массив тех значений для всех собств частот

u_list_point = []  # массив собственных форм для массива точек балки
u_second_dif_list_point = []
D_i_list = []  #
w_sobst_list = np.zeros(len(alpha_l_list), dtype=float)
u_list = [lambda:1 for _ in range(len(alpha_l_list))]
u_i_second_dif_list = [lambda:1 for _ in range(len(alpha_l_list))]
for id, alpha_l in enumerate(alpha_l_list):  # для каждой собственной частоты запоминаем набор параметров
    [w_sobst_i, alpha_i, D_i, u_i, u_i_second_dif] = find_u_i(alpha_l)
    w_sobst_list[id] = w_sobst_i
    u_list[id] = u_i
    u_i_second_dif_list[id] = u_i_second_dif
    u_list_point.append(np.array([u_i(ii) for ii in np.linspace(0, l, point)]))  # двумерный массив, внешний по собственным частотам, внутренний по координатам
    u_second_dif_list_point.append(np.array([u_i_second_dif(ii) for ii in np.linspace(0, l, point)]))
    D_i_list.append(D_i)

# Начальные условия учитываются в betta0 and betta0_dif
RRR = ro * F / E / Jx

# ------------------------------------------------
# disp0 = u_list_point[0] / 10000  # выбираем начальное перемещение в точности по собственной форме

# начальное перемещение по точкам из стат. расчета МКЭ (FEM_create_initial_disp_list.py)
file_name = 'initial_disp_after_VI_loc_{}.txt'.format(round(loc * 10))
with open(r'./initial_disp/' + file_name, 'r') as cur_file:
    initial_disp_vel = []
    for line in cur_file:
        line = line.strip()
        initial_disp_vel.append(float(line))
# в листе initial_disp_vel хранятся значения перемещений и скростей точек балки в момент после отрыва от барьера

# разделяем значения входного файла на значения перемешения и скорости
disp0_long = np.array(initial_disp_vel[:round((len(initial_disp_vel) - 1) / 2)])
disp0_dif_long = np.array(initial_disp_vel[round((len(initial_disp_vel) - 1) / 2) + 1:])
# print(len(disp0))
# ------------------------------------------------

# берем каждую н-ную точку входного листа, что бы получилось количество точек, указанных в начале проги
disp0 = disp0_long[::(len(disp0_long)//(point-1)-1)][:point]
disp0_dif = disp0_dif_long[::(len(disp0_dif_long)//(point-1)-1)][:point]
# disp0_dif = np.zeros(point, dtype=float)
dl = l / (point - 1)

betta0 = np.zeros(len(alpha_l_list), dtype=float)
betta0_dif = np.zeros(len(alpha_l_list), dtype=float)

for id, alpha_l in enumerate(alpha_l_list):
    betta0[id] = dl * ((sum(disp0 * u_list_point[id]) + sum(disp0[:-1] * u_list_point[id][:-1])) / 2)  # считаем интеграл для betta0 по методу трапеций (среднее между верхней и нижней суммой Дарбу)
    betta0_dif[id] = dl * ((sum(disp0_dif * u_list_point[id]) + sum(disp0_dif[:-1] * u_list_point[id][:-1])) / 2)  # считаем интеграл для betta0


y_list = np.zeros(point, dtype=float)
betta_i = np.zeros(len(alpha_l_list), dtype=float)

fig, axs = plt.subplots(2, 2)  # создаем саб плот из 2 графиков
plt.subplots_adjust(wspace=0.4, hspace=0.7)

# fig.suptitle('Time = ' + str("%.5g" % t))
fig.suptitle('Time = ' + str("%.4f" % step) + ' sec')

point_barrier = round((point - 1) * loc)

# начинаем цикл по времени
for t in np.arange(step, time_end, step):
    start_time = time.time()  # для засечки времени одной итерации цикла
    # try:
    y_list = np.zeros(point, dtype=float)
    for id in range(len(alpha_l_list)):
        w_sobst_i = w_sobst_list[id]

        betta_i[id] = betta0[id] * np.cos(w_sobst_i * t) + betta0_dif[id] / w_sobst_i * np.sin(w_sobst_i * t)
        y_i = betta_i[id] * u_list_point[id] / D_i_list[id]  # функция колебания

        amplitude_mode_oscillations[id] = max(amplitude_mode_oscillations[id], max(map(abs, y_i)))  # пишем максимальную амплитуду колебаний текущей моды

        vel_i = (betta0[id] * (-w_sobst_i * np.sin(w_sobst_i * t)) + betta0_dif[id] * np.cos(w_sobst_i * t)) * u_list_point[id] / D_i_list[id]  # velocity
        kinetic_energy_id = 0.5 * (ro * F) * (dl * sum(vel_i ** 2))

        y_i_second_dif = betta_i[id] * u_second_dif_list_point[id] / D_i_list[id]
        potential_energy_id = 0.5 * E * Jx * (dl * sum(y_i_second_dif ** 2))

        total_energy_id = kinetic_energy_id + potential_energy_id
        total_energy[id] = total_energy_id

        y_list += y_i  # суммарная функция колебания

    # fig.suptitle('Time = ' + str("%.5g" % t))
    fig.suptitle('Time = ' + str("%.4f" % t) + ' sec')


    # вывод графика колебания конца балки
    y_end.append(y_list[-1])
    y_barrier.append(y_list[point_barrier])
    list_time.append(t)

    if len(list_time) % step_plot == 0:
        axs[1][0].plot(list_time, y_end, 'k', linewidth=1)
        axs[1][0].plot(list_time, y_barrier, 'g', linewidth=1)
        axs[1][0].grid()
        axs[1][0].set_title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', fontsize=10)
        axs[1][0].set(xlabel='time, sec', ylabel='displacement, m')
        # axs[1].set_ylim(-3e-4, 0)
        print("%.5g" % t, "%.5g" % y_list[-1])


    # вывод графика колебания балки
    if len(list_time) % step_plot == 0:
        axs[0][0].plot(np.linspace(0, l, point), y_list, 'r', linewidth=1)
        axs[0][0].grid()
        axs[0][0].set_title('Beam shape')
        axs[0][0].set_xlim(0, l)
        axs[0][0].set(xlabel='beam, m', ylabel='displacement, m')
        # axs[0][0].set_ylim(-1e-4, 1e-4)
        axs[0][0].set_ylim(-max(abs(np.array(y_end))), max(abs(np.array(y_end))))

    # вывод графика amplitude колебания балки
    if len(list_time) % step_plot == 0:
        df = pd.DataFrame(amplitude_mode_oscillations[:4], index=range(1, 5), columns=['Value'])
        df['Percentage'] = (df['Value'] / df['Value'].sum()) * 100
        # Plot the bar chart
        # axs[0][1].plot(df, kind='bar', y='Value', legend=False, rot=0)
        axs[0][1].bar(x=df.index, height=df['Percentage'], color='skyblue')

        # Annotate each bar with percentages
        for p in axs[0][1].patches:
            axs[0][1].annotate(f'{p.get_height():.2f}%',
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom', fontsize=8, color='black')

        axs[0][1].set_title('Mode amplitudes')
        axs[0][1].set(xlabel='mode')


        # вывод графика energy колебания балки
        if len(list_time) % step_plot == 0:
            df2 = pd.DataFrame(total_energy[:4], index=range(1, 5), columns=['Value'])
            # print(df2)
            # Plot the bar chart
            axs[1][1].bar(x=df2.index, height=df2['Value'], color='skyblue')

            # Annotate each bar with percentages
            for p in axs[1][1].patches:
                axs[1][1].annotate(f'{p.get_height():.2f}',
                                   (p.get_x() + p.get_width() / 2., p.get_height()),
                                   ha='center', va='bottom', fontsize=8, color='black')

            axs[1][1].set_title('Energy mode')
            axs[1][1].set(xlabel='mode')


        plt.pause(0.0001)
        axs[0][0].clear()
        axs[1][0].clear()
        axs[0][1].clear()
        axs[1][1].clear()

        # print("--- %s seconds ---" % (time.time() - start_time))  # время одного шага цикла
        # print(time.time() - start_time)
        # print(amplitude_mode_oscillations)
        print(sum(total_energy))

    # except Exception as e:
    #     print(e)
    #     break

print('Amplitude of mode oscillations')
print(amplitude_mode_oscillations)