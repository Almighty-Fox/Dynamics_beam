import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# находим собственные частоты и формы консольной балки
def beam_without_barrier():
    al_l = np.arange(0, 21, 0.00001)
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


def initial_conditions():
    disp_start = np.zeros(points)
    vel_start = -omegas1[0] * forms1[0] / 100000
    # vel_start = omegas1[0] * forms1[0] * 0

    return disp_start, vel_start


def det_AB_before_VI():
    A_lst = np.zeros(len(omegas1), dtype=float)
    for i in range(len(omegas1)):
        integ_body = forms1[i] * disp_start
        for ii in range(1, len(integ_body)):
            A_lst[i] += (integ_body[ii] + integ_body[ii - 1]) / 2
    A_lst *= dl
    # for i in range(len(A_lst)):
    #     A_lst[i] = 0

    B_lst = np.zeros(len(omegas1), dtype=float)
    for i in range(len(omegas1)):
        integ_body = forms1[i] * vel_start
        for ii in range(1, len(integ_body)):
            B_lst[i] += (integ_body[ii] + integ_body[ii - 1]) / 2
    B_lst *= dl
    # for i in range(1, len(B_lst)):
    #     B_lst[i] = 0

    return A_lst, B_lst


def dynamic_beam_recalc_integ():
    tau = 1e-4
    t_lst = [0, tau]
    slag_koef = l_length / delta2 / F_section / ro_density

    slag2_koef_norm = 1
    slag2 = slag2_koef_norm * slag_koef * sum([1 / rk_lst[ii]**2 / omegas1[ii] * forms1_barrier[ii]**2 / D1[ii]**2 * (1 - np.cos(omegas1[ii]*tau)) for ii in range(len(omegas1))])

    Pq_lst = []

    while len(Pq_lst) < 2000:
        # print(f'N = {len(t_lst)}')
        slag_free = 0
        for k in range(len(omegas1)):
            slag_free_add = forms1_barrier[k] / D1[k] * (
                    A_lst[k] * np.cos(omegas1[k] * t_lst[-1]) + B_lst[k] * np.sin(omegas1[k] * t_lst[-1]))
            slag_free += slag_free_add

        prev_sum = np.zeros(len(omegas1))
        for k in range(len(omegas1)):
            for i in range(len(Pq_lst)):
                prev_sum[k] += Pq_lst[i] * np.sin(omegas1[k] * (t_lst[-2] - t_lst[i]))
            prev_sum[k] *= tau

        # print(f'prev_sum = {prev_sum}')

        slag1 = slag_koef * sum([1 / rk_lst[ii]**2 * forms1_barrier[ii]**2 / D1[ii]**2 * prev_sum[ii] for ii in range(len(omegas1))])
        # slag2 = 1
        Pq_new = (-slag_free - slag1) / slag2
        # Pq_new = sum(prev_sum)

        print(f'N = {len(Pq_lst)}')
        print(f'slag_free = {slag_free}')
        print(f'slag1 = {slag1}')
        print(f'slag2 = {slag2}')

        Pq_lst.append(Pq_new * slag2_koef_norm)
        axs[0].plot(t_lst[:-1], Pq_lst, 'b', linewidth=1, marker='')

        slag_free_shape = np.zeros(points)
        for k in range(len(omegas1)):
            slag_free_add = forms1[k] / D1[k] * (
                    A_lst[k] * np.cos(omegas1[k] * t_lst[-1]) + B_lst[k] * np.sin(omegas1[k] * t_lst[-1]))
            slag_free_shape += slag_free_add

        slag_VI_shape = np.zeros(points)
        for k in range(len(omegas1)):
            slag_VI_shape_add = 1/rk_lst[k]**2 * forms1[k] * forms1_barrier[k] / D1[k] ** 2 * (prev_sum[k] + Pq_lst[-1] / omegas1[k] * (1 - np.cos(omegas1[k]*tau)))
            slag_VI_shape += slag_VI_shape_add
        slag_VI_shape *= slag_koef

        u_new = slag_free_shape + slag_VI_shape
        # u_new = slag_VI_shape
        axs[1].plot(np.linspace(0, l_length, points), u_new, 'b', linewidth=1, marker='')
        axs[1].plot([0, 1], [0, 0], 'r', linewidth=1, marker='')
        axs[1].plot([l_barrier], [0], 'r', linewidth=1, marker='.')
        plt.pause(0.3)
        axs[1].clear()

        t_lst.append(t_lst[-1] + tau)

        # print(f'Pq_new = {Pq_new}')

    print(f'slag_free = {slag_free}')
    print(f'slag1 = {slag1}')
    print(f'slag2 = {slag2}')
    print(f'prev_sum = {prev_sum}')
    axs[0].plot(t_lst[:-2], Pq_lst, 'b', linewidth=1, marker='')
    plt.show()


E_young = 2e11
section_side = 10e-3
J_inertia = section_side ** 4 / 12
ro_density = 7850
l_length = 1
l_barrier = 0.8
F_section = section_side ** 2

points = 100 + 1
dl = l_length / (points - 1)
point_barrier = round((points - 1) * l_barrier)
Hertz_koef = 1

fig, axs = plt.subplots(2)  # создаем саб плот из 2 графиков
plt.subplots_adjust(wspace=0.6, hspace=0.5)

[alpha, D1, forms1, omegas1, omegas2, form1_second_dif] = beam_without_barrier()
forms1_barrier = [forms1[i][point_barrier] for i in range(len(omegas1))]
delta2 = E_young * J_inertia / ro_density / F_section
delta = np.power(delta2, 1/2)
rk_lst = np.sqrt(np.array(omegas1) * l_length / delta)
print('rk = {}'.format(rk_lst))
ak = rk_lst**2 * delta / l_length**2

disp_start, vel_start = initial_conditions()
A_lst, B_lst = det_AB_before_VI()

Pq_global = []
time_global = []

dynamic_beam_recalc_integ()