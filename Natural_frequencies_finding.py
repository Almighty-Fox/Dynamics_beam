import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def natural_freq_barrier():
    k = np.arange(0.0001, 18, 0.00001)  # лист значений каппы - замененная частота
    # k = np.arange(0.0001, 14, 0.00001)

    A1 = ((k * np.cos(k * l1) - k * np.cosh(k * l1)) - (np.sin(k * l1) - np.sinh(k * l1)) / (
                np.cos(k * l1) - np.cosh(k * l1)) * (-k * np.sin(k * l1) - k * np.sinh(k * l1)))
    A2 = ((-k ** 2 * np.sin(k * l1) - k ** 2 * np.sinh(k * l1)) - (np.sin(k * l1) - np.sinh(k * l1)) / (
                np.cos(k * l1) - np.cosh(k * l1)) * (-k ** 2 * np.cos(k * l1) - k ** 2 * np.cosh(k * l1)))

    B1 = ((k * np.cos(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * k * np.cosh(
        k * l1) + (np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * k * np.sinh(k * l1)) -
          (np.sin(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * np.sinh(k * l1) + (
                      np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) /
          (np.cos(k * l1) - (np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * np.sinh(k * l1) + (
                      np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) *
          (-k * np.sin(k * l1) - (np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * k * np.cosh(
              k * l1) + (np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * k * np.sinh(k * l1)))

    B2 = ((-k ** 2 * np.sin(k * l1) + (
                np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * k ** 2 * np.sinh(k * l1) + (
                       np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * k ** 2 * np.cosh(
        k * l1)) -
          (np.sin(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * np.sinh(k * l1) + (
                      np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) /
          (np.cos(k * l1) - (np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * np.sinh(k * l1) + (
                      np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) *
          (-k ** 2 * np.cos(k * l1) - (
                      np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * k ** 2 * np.sinh(
              k * l1) + (np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * k ** 2 * np.cosh(
              k * l1)))

    fun2 = -A1 * B2 + A2 * B1
    roots2 = []
    omega2 = []
    for i in range(len(fun2) - 1):
        if (fun2[i] * fun2[i + 1] < 0) and (abs(fun2[i] * fun2[i + 1]) < 300):
            roots2_i = (k[i] + k[i + 1]) / 2
            if (len(roots2) == 0) or (abs(roots2_i - roots2[-1]) > 1):
                roots2.append(roots2_i)
                omega2.append(np.sqrt(E * J_inertia / ro / S) * (roots2_i ** 2))

    print('Freq VI = ' + str(roots2))

    plt.plot(k, fun2)
    plt.grid()
    return roots2


def natural_freq_barrier_fix():
    k = np.arange(0.0001, 18, 0.00001)  # лист значений каппы - замененная частота
    # k = np.arange(0.0001, 14, 0.00001)

    A1 = ((k * np.cos(k * l1) - k * np.cosh(k * l1)) * (np.cos(k * l1) - np.cosh(k * l1)) - (
                np.sin(k * l1) - np.sinh(k * l1)) * (-k * np.sin(k * l1) - k * np.sinh(k * l1)))
    A2 = ((-k ** 2 * np.sin(k * l1) - k ** 2 * np.sinh(k * l1)) * (np.cos(k * l1) - np.cosh(k * l1)) - (
                np.sin(k * l1) - np.sinh(k * l1)) * (-k ** 2 * np.cos(k * l1) - k ** 2 * np.cosh(k * l1)))

    B1 = ((k * np.cos(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * k * np.cosh(
        k * l1) + (np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * k * np.sinh(k * l1)) * (
                      np.cos(k * l1) - (np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * np.sinh(
                  k * l1) + (np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) - (
                      np.sin(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * np.sinh(
                  k * l1) + (np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) * (
                      -k * np.sin(k * l1) - (
                          np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * k * np.cosh(k * l1) + (
                                  np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * k * np.sinh(
                  k * l1)))

    B2 = ((-k ** 2 * np.sin(k * l1) + (
                np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * k ** 2 * np.sinh(k * l1) + (
                       np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * k ** 2 * np.cosh(
        k * l1)) * (np.cos(k * l1) - (np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * np.sinh(
        k * l1) + (np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) - (
                      np.sin(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * np.sinh(
                  k * l1) + (np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) * (
                      -k ** 2 * np.cos(k * l1) - (
                          np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * k ** 2 * np.sinh(
                  k * l1) + (np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * k ** 2 * np.cosh(
                  k * l1)))

    fun2 = -A1 * B2 + A2 * B1
    roots2 = []
    omega2 = []
    # for i in range(len(fun2) - 1):
    #     if (fun2[i] * fun2[i + 1] < 0) and (abs(fun2[i] * fun2[i + 1]) < 300):
    #         roots2_i = (k[i] + k[i + 1]) / 2
    #         if (len(roots2) == 0) or (abs(roots2_i - roots2[-1]) > 1):
    #             roots2.append(roots2_i)
    #             omega2.append(np.sqrt(E * J_inertia / ro / S) * (roots2_i ** 2))

    for i in range(len(fun2) - 1):
        if (fun2[i] * fun2[i + 1] < 0):
            roots2_i = (k[i] + k[i + 1]) / 2
            if (len(roots2) == 0) or (abs(roots2_i - roots2[-1]) > 0.1):
                roots2.append(roots2_i)
                omega2.append(np.sqrt(E * J_inertia / ro / S) * (roots2_i ** 2))

    print('Freq VI fix = ' + str(roots2))

    A1_fun = lambda k: ((k * np.cos(k * l1) - k * np.cosh(k * l1)) * (np.cos(k * l1) - np.cosh(k * l1)) - (
            np.sin(k * l1) - np.sinh(k * l1)) * (-k * np.sin(k * l1) - k * np.sinh(k * l1)))
    A2_fun = lambda k: ((-k ** 2 * np.sin(k * l1) - k ** 2 * np.sinh(k * l1)) * (np.cos(k * l1) - np.cosh(k * l1)) - (
            np.sin(k * l1) - np.sinh(k * l1)) * (-k ** 2 * np.cos(k * l1) - k ** 2 * np.cosh(k * l1)))

    B1_fun = lambda k: ((k * np.cos(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * k * np.cosh(
        k * l1) + (np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * k * np.sinh(k * l1)) * (
                  np.cos(k * l1) - (np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * np.sinh(
              k * l1) + (np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) - (
                  np.sin(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * np.sinh(
              k * l1) + (np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) * (
                  -k * np.sin(k * l1) - (
                  np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * k * np.cosh(k * l1) + (
                          np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * k * np.sinh(
              k * l1)))

    B2_fun = lambda k: ((-k ** 2 * np.sin(k * l1) + (
            np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * k ** 2 * np.sinh(k * l1) + (
                   np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * k ** 2 * np.cosh(
        k * l1)) * (np.cos(k * l1) - (np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * np.sinh(
        k * l1) + (np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) - (
                  np.sin(k * l1) + (np.cos(k * l2) * np.cosh(k * l2) - np.sin(k * l2) * np.sinh(k * l2)) * np.sinh(
              k * l1) + (np.cosh(k * l2) * np.sin(k * l2) - np.cos(k * l2) * np.sinh(k * l2)) * np.cosh(k * l1)) * (
                  -k ** 2 * np.cos(k * l1) - (
                  np.sin(k * l2) * np.cosh(k * l2) + np.cos(k * l2) * np.sinh(k * l2)) * k ** 2 * np.sinh(
              k * l1) + (np.cosh(k * l2) * np.cos(k * l2) + np.sin(k * l2) * np.sinh(k * l2)) * k ** 2 * np.cosh(
              k * l1)))

    fun2_fun = lambda k: -A1_fun(k) * B2_fun(k) + A2_fun(k) * B1_fun(k)
    roots2_fsolve = fsolve(fun2_fun, np.array(roots2), xtol=1e-10)
    print('roots2_fsolve = ' + str(roots2_fsolve))

    check_roots = [fun2_fun(k_i) for k_i in roots2]
    print('check_roots = ' + str(check_roots))
    check_roots_fsolve = [fun2_fun(k_i) for k_i in roots2_fsolve]
    print('check_roots_fsolve = ' + str(check_roots_fsolve))

    roots2_ansys = (ro * S / E / J_inertia) ** (1/4) * (np.array([50.821, 121.27, 219.32, 409.51, 668.86, 952.95, 1132.0, 1461.1]) * 2 * np.pi) ** (1/2)
    print('roots2_ansys = ' + str(roots2_ansys))
    check_roots_ansys = [fun2_fun(k_i) for k_i in roots2_ansys]
    print('check_roots_fsolve = ' + str(check_roots_ansys))

    plt.plot(k, fun2)
    plt.grid()
    return roots2


# -----------------------------------------------
ro = 7850
E = 2e11
a = 10e-3
S = a * a  # площадь сечения балки
J_inertia = a * a ** 3 / 12  # момент инерции сечения
ro = 7850  # плотность
E = 2e11  # модуль Юнга
# ------------------------------------------------
point = 500 + 1  # количество элементов балки
l1 = 0.8  # местоположение барьера
l2 = 1  # длина балки
dl = l2 / (point - 1)
point_barrier = round((point - 1) * l1)
# ------------------------------------------------

# roots_VI = natural_freq_barrier()
roots_VI = natural_freq_barrier_fix()

plt.show()

