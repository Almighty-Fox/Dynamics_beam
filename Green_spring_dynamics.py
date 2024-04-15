import numpy as np
import matplotlib.pyplot as plt


# определение собственной частоты и формы струны
#
#
#

def determinate_eigen_params():
    omegas = [np.pi * n for n in range(Num_modes)]
    f_out = lambda t: t if t < 1 else 0
    t_cur = 0
    while t_cur < t_limit:
        t_cur += dt_time
        u_cur = np.zeros(points_num)
        for k in range(Num_modes):
            kk = k + 1
            Ik_add = f_out(t_cur - dt_time) * np.sin(np.pi * kk * (t_cur - dt_time)) * dt_time / 2
            Ik_integral[k] += Ik_add
            u_add = 2 / np.pi / kk * np.sin(np.pi * kk * points_lst) * np.sin(np.pi * kk * loc_bar) * Ik_integral[k]
            u_cur += u_add

        # plt.figure(1)
        # plt.plot(points_lst, u_cur)
        # plt.pause(0.1)
        # plt.clean()

        axs[0].plot(points_lst, u_cur, 'g', linewidth=1)
        axs[0].set_title('P(t) = ' + str("%.4f" % f_out(t_cur - dt_time)))
        # axs[0].set_ylim(-0.0004, 0.0004)


        plt.pause(0.1)

        axs[0].clear()



Num_modes = 10
points_num = 100
points_lst = np.linspace(0, 1, points_num)
loc_bar = 0.7
u_cur = np.zeros(points_num)
Ik_integral = np.zeros(Num_modes)

t_limit = 100
dt_time = 1e-2

fig, axs = plt.subplots(2)  # создаем саб плот из 2 графиков
plt.subplots_adjust(wspace=0.6, hspace=0.5)

# fig.suptitle('Time = ' + str("%.4f" % step) + ' sec')

determinate_eigen_params()