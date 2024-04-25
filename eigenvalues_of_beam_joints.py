import numpy as np
import pylab as plt


def beam_with_barrier():
    k = np.arange(0.0001, 40, 0.00001)  # лист значений каппы - замененная частота
    # k = np.arange(0.0001, 14, 0.00001)

    A1 = (np.cos(k*l1)-np.sin(k*l1) / np.tanh(k*l1))
    B2 = (-np.sin(k*l1)+np.tan(k*l2)*np.cos(k*l1))

    A2 = np.sin(k*l1)
    B1 = ((np.cos(k*l1) + np.tan(k*l2)*np.sin(k*l1)) - (np.sin(k*l1)-np.tan(k*l2)*np.cos(k*l1))/(np.sinh(k*l1)-np.tanh(k*l2)*np.cosh(k*l1))*(np.cosh(k*l1)-np.tanh(k*l2)*np.sinh(k*l1)))

    fun2 = A1 * B2 + A2 * B1
    roots2 = []
    omega2 = []

    for i in range(len(fun2) - 1):
        if (fun2[i] * fun2[i+1] < 0) and (abs(fun2[i] * fun2[i+1]) < 300):
            roots2_i = (k[i] + k[i+1]) / 2
            # if (len(roots2) == 0) or (abs(roots2_i - roots2[-1]) > 1):
            roots2.append(roots2_i)
            omega2.append(np.sqrt(E * J_inertia / ro / S) * (roots2_i ** 2))

    print(roots2)


def beam_with_barrier_multiply():
    k = np.arange(0.0001, 21, 0.00001)  # лист значений каппы - замененная частота
    # k = np.arange(0.0001, 14, 0.00001)

    A1 = ((np.cos(k*l1)*np.tanh(k*l1)) - np.sin(k*l1)) * (np.sinh(k*l1)-np.tanh(k*l2)*np.cosh(k*l1))
    B2 = (-np.sin(k*l1)+np.tan(k*l2)*np.cos(k*l1))

    A2 = np.sin(k*l1)
    B1 = ((np.cos(k*l1) + np.tan(k*l2)*np.sin(k*l1)) * (np.sinh(k*l1)-np.tanh(k*l2)*np.cosh(k*l1)) - (np.sin(k*l1)-np.tan(k*l2)*np.cos(k*l1))*(np.cosh(k*l1)-np.tanh(k*l2)*np.sinh(k*l1))) * np.tanh(k*l1)

    fun2 = A1 * B2 + A2 * B1
    roots2 = []
    omega2 = []

    for i in range(len(fun2) - 1):
        if (fun2[i] * fun2[i+1] < 0) and (abs(fun2[i] * fun2[i+1]) < 300):
            roots2_i = (k[i] + k[i+1]) / 2
            # if (len(roots2) == 0) or (abs(roots2_i - roots2[-1]) > 1):
            roots2.append(roots2_i)
            omega2.append(np.sqrt(E * J_inertia / ro / S) * (roots2_i ** 2))

    print(roots2)
    plt.plot(k, fun2)
    plt.grid()
    plt.show()


def plot_shapes():
    A1_fun = lambda k: k * (np.cos(k*l1) - np.sin(k*l1)/np.tanh(k*l1))
    B1_fun = lambda k: k * (np.cos(k*l1) + np.tan(k*l2)*np.sin(k*l1)-(np.sin(k*l1) - np.tan(k*l2)*np.cos(k*l1))/(np.sinh(k*l1) - np.tanh(k*l2)*np.cosh(k*l1))*(np.cosh(k*l1) - np.tanh(k*l2)*np.sinh(k*l1)))

    # roots_k = [4.618325, 8.391544999999999, 12.161734999999998, 14.137165, 15.707964999999998, 17.278754999999997, 17.872544999999995, 20.420355, 20.461185]
    roots_k = [4.618325, 8.391544999999999, 12.161734999999998, 14.137165, 15.707964999999998, 17.278754999999997, 17.872544999999995]

    v1, v2 = [], []
    forms2 = []
    for i in range(len(roots_k)):
        k = roots_k[i]
        x = np.linspace(0, l2, point)[:round(l1 * point)]
        alpha1 = 1
        v1.append(alpha1 * (np.sin(k*x) - np.sin(k*l1)/np.sinh(k*l1)*np.sinh(k*x)))
        x = np.linspace(0, l2, point)[round(l1 * point):]
        beta1 = alpha1 * A1_fun(k) / B1_fun(k)
        v2.append(beta1 * (np.sin(k*x) - np.tan(k*l2)*np.cos(k*x)-(np.sin(k*l1) - np.tan(k*l2)*np.cos(k*l1))/(np.sinh(k*l1) - np.tanh(k*l2)*np.cosh(k*l1))*(np.sinh(k*x) - np.tanh(k*l2)*np.cosh(k*x))))

        v1v2 = np.concatenate((v1[i], v2[i]))
        forms2.append(v1v2)  # форма всей балки как сумма формы левой и правой частей

    for i in range(len(forms2)):
        plt.figure(2)
        plt.plot(np.linspace(0, l2, point), forms2[i])
        plt.grid()
        plt.show()

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

# beam_with_barrier()
# beam_with_barrier_multiply()
plot_shapes()