import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# находим собственные частоты и формы консольной балки
def beam_without_barrier():
    al_l = np.arange(0, 21, 0.00001)
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
    vel_start = omegas1[0] * forms1[0]

    return disp_start, vel_start


def det_AB_before_VI():
    A_lst = np.zeros(len(omegas1), dtype=float)
    for i in range(len(omegas1)):
        integ_body = forms1[i] * disp_start
        for ii in range(1, len(integ_body)):
            A_lst[i] += (integ_body[ii] + integ_body[ii - 1]) / 2
    A_lst *= dl

    B_lst = np.zeros(len(omegas1), dtype=float)
    for i in range(len(omegas1)):
        integ_body = forms1[i] * vel_start
        for ii in range(1, len(integ_body)):
            B_lst[i] += (integ_body[ii] + integ_body[ii - 1]) / 2
    B_lst *= dl

    return A_lst, B_lst


def iterative_solution_search():
    global Pq_global
    # Hertz_force = lambda e: Hertz_koef ** (-3/2) * (-e) ** (3/2)
    # Pt_prev = Hertz_force(vel_start[point_barrier] * t_step)
    # Pt_next = Pt_0

    t_cur = 0
    Pq = 1
    Pq_prev = 0
    Iq_prev = [0] * len(omegas1)

    q = 0  # целое число, номер шага по времени
    out_time = 0  # внешнее время
    tau = 1e-8  # шаг по времени
    in_time = 0  # время между дискретными временными шагами
    out_time = q * tau + in_time

    in_time_Nstep = 1
    in_time_step = tau / in_time_Nstep

    while (Pq > 0) or True:
        print(Pq)
        q += 1
        print('q = {}'.format(q))
        in_time = 0
        for i in range(in_time_Nstep):
            in_time += in_time_step
            out_time = (q-1) * tau + in_time
            summ1, summ2 = 0, 0
            t_cur += t_step
            for ii in range(len(omegas1)):
                summ1 += 1 / rk_lst[ii] ** 4 * forms1_barrier[ii]**2 / D1[ii] ** 2 * l_length ** 2 / delta * (
                            np.cos(rk_lst[ii] ** 2 * delta / l_length ** 2 * out_time * (1 - q)) - np.cos(
                        rk_lst[ii] ** 2 * delta / l_length ** 2 * out_time * (2 - q)))


                summ2 += 1 / rk_lst[ii] ** 2 * forms1_barrier[ii] ** 2 / D1[ii] ** 2 * (1 / ak[ii] * (
                            q * out_time * np.cos(ak[ii] * out_time * (1 - q)) - (q - 1) * out_time * np.cos(
                        ak[ii] * out_time * (2 - q))) + 1 / ak[ii] ** 2 * (np.sin(ak[ii] * out_time * (1 - q)) - np.sin(
                    ak[ii] * out_time * (2 - q))))

            # расписываем все слагаемые в Pq отдельно, что бы не писать сразу крокодила
            slag_koef = l_length / delta2 / F_section / ro_density
            slag1 = sum([(forms1_barrier[ii] * (A_lst[ii] * np.cos(omegas1[ii] * out_time) + B_lst[ii] * np.sin(omegas1[ii] * out_time))) for ii in range(len(omegas1))])

            slag2 = slag_koef * sum([(forms1_barrier[ii] ** 2 / rk_lst[ii] ** 2 * Iq_prev[ii]) for ii in range(len(omegas1))])
            slag3 = slag_koef * Pq_prev * q * summ1
            slag4 = slag_koef * Pq_prev / tau * summ2
            slag5 = slag_koef * (1 - q) * summ1 + slag_koef / tau * summ2

            Pq = -(slag1 + slag2 + slag3 + slag4) / slag5
            Pq_global.append(Pq)

            plt.figure(1)
            plt.plot(Pq_global, 'k')
            plt.pause(0.1)

            Iq = []
            for ii in range(len(omegas1)):
                slag1 = (np.cos(rk_lst[ii]**2 * delta / l_length**2 * out_time * (1 - q)) - np.cos(rk_lst[ii]**2 * delta / l_length**2 * out_time * (2 - q)))
                slag2 = (Pq - Pq_prev) / tau * (1/ak[ii] * (q*out_time*np.cos(ak[ii]*out_time*(1-q)) - (q-1)*out_time*np.cos(ak[ii]*out_time*(2-q))) + 1/ak[ii]**2 * (np.sin(ak[ii]*out_time*(1-q)) - np.sin(ak[ii]*out_time*(2-q))))

                Iq.append(Iq_prev[ii] + (Pq * (1 - q) - Pq_prev * q) * l_length**2 / rk_lst[ii]**2 / delta * slag1 + slag2)
                print('Iq = {}'.format(Iq[-1]))

            # change next ans prev values
            Iq_prev = Iq.copy()
            Pq_prev = Pq






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
t_step = 1e-6

[alpha, D1, forms1, omegas1, omegas2, form1_second_dif] = beam_without_barrier()
forms1_barrier = [forms1[i][point_barrier] for i in range(len(omegas1))]
delta2 = E_young * J_inertia / ro_density / F_section
delta = np.power(delta2, 1/2)
rk_lst = np.sqrt(np.array(omegas1) * l_length / delta)
ak = rk_lst**2 * delta / l_length**2
# r_i_4 = l_length ** 3 * omegas2 / delta2
# r_i_2 = np.power(r_i_4, 1/2)

disp_start, vel_start = initial_conditions()
A_lst, B_lst = det_AB_before_VI()

Pq_global = []
iterative_solution_search()