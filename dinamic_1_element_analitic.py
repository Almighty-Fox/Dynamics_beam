import numpy as np
import time


# ---------определяем параметры материала образца----------
a = 10e-3  # сторона квадратного сечения
S = a * a  # площадь сечения балки
I = a * a ** 3 / 12  # момент инерции сечения
ro = 7850 * S  # погонная плотность
E = 2e11  # модуль Юнга
# ---------определяем параметры геометрии и КЭ образца----------
L = 1
MaxNode = 1 + 1  # количество узлов
dl = L
# ---------определяем параметры временного шага----------
# первая частота балки = 50 Гц, период = 0.02с
dt = 2e-5
t_end = 20
t_list = np.arange(dt, t_end, dt)
# ------------------------------------------------------

dis_i = 1
vel_i = 1
acc_i = 1
fi_i = 1
om_i = 1
bet_i = 1

for t in t_list:
    # fi_i1 = (3*L/6 - ro*L/420 * (56*L*acc_i - 7*L**2*bet_i)) * L/E/I
    # print(fi_i1)
    # dis_i1 = ((1 - ro*L/420*(156*acc_i-22*L*bet_i))*L**2/2/E/I + 3*L*fi_i1) / 6
    fi_i1 = (- ro*L/420 * (56*L*acc_i - 7*L**2*bet_i)) * L/E/I
    dis_i1 = ((- ro*L/420*(156*acc_i-22*L*bet_i))*L**2/2/E/I + 3*L*fi_i1) / 6

    om_i1 = (fi_i1 - fi_i) / dt
    vel_i1 = (dis_i1 - dis_i) / dt
    bet_i1 = (om_i1 - om_i) / dt
    acc_i1 = (vel_i1 - vel_i) / dt

    print(fi_i1)
    time.sleep(1)

    dis_i = dis_i1
    vel_i = vel_i1
    acc_i = acc_i1
    fi_i = fi_i1
    om_i = om_i1
    bet_i = bet_i1






























