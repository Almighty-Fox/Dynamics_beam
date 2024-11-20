import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


# Определение функции Хевисайда
def heaviside(t):
    return np.heaviside(t, 1)  # Хевисайда возвращает 1 при t>=0, и 0 при t<0


# Функция для численного интегрирования
def integrand(t1, t, k, a):
    result = (-alpha_n * np.cos(np.pi * alpha_n * t1) * np.sin(np.pi * alpha_n * a) +
              beta_n * np.sin(np.pi * beta_n * t1) * np.sin(np.pi * beta_n * a))

    return result * np.pi * k * np.cos(np.pi * k * (t - t1))


alpha_n, beta_n = 1, 4  # по какой моде начальная скорость (alpha_n) и по какой начальная форма (beta_n)


# Основная функция для y(x,t)
def y(x, t, a, N=100):
    sum_term = 0

    for k in range(1, N+1):
        integral_value, _ = quad(integrand, 0, t, args=(t, k, a))
        sum_term += (4 / k) * np.sin(np.pi * k * x) * np.sin(np.pi * k * a) * integral_value

    init_cond = (np.sin(np.pi * alpha_n * x) * np.pi * alpha_n * np.cos(np.pi * alpha_n * t) -
                 np.sin(np.pi * beta_n * x) * np.pi * beta_n * np.sin(np.pi * beta_n * t))

    # return np.sin(np.pi * x) * np.sin(np.pi * t) - sum_term * np.sin(np.pi * a)
    return init_cond + sum_term

# Параметры
x = np.linspace(0, 1, 1000)
t_values = np.linspace(0, 0.5, 100)
a = 0.75  # Параметр a

# Создание фигуры и осей для анимации
fig, ax = plt.subplots()
line, = ax.plot(x, y(x, 0, a), lw=2)
ax.set_xlim(0, 1)
ax.set_ylim(-3, 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)  # Добавление сетки

# Обновление графика в цикле
for t in t_values:
    y_values = y(x, t, a)
    line.set_ydata(-y_values)
    ax.set_title(f't={t:.2f}')
    plt.pause(0.1)  # Задержка между кадрами

plt.show()

# # Функция для численного интегрирования
# def integrand(t1, t, k):
#     return np.cos(np.pi * t1) * np.cos(np.pi * k * (t - t1))
#
# # Основная функция для y(x,t)
# def y(x, t, a, N=300):
#     sum_term = 0
#     for k in range(1, N+1):
#         integral_value, _ = quad(integrand, 0, t, args=(t, k))
#         sum_term += (4 * np.pi) * np.sin(np.pi * k * x) * np.sin(np.pi * k * a) * integral_value
#
#     alpha_n, beta_n = 1, 2  # по какой моде начальная скорость (alpha_n) и по какой начальная форма (beta_n)
#     init_cond = (np.sin(np.pi * alpha_n * x) * np.pi * alpha_n * np.cos(np.pi * alpha_n * t) -
#                  np.sin(np.pi * beta_n * x) * np.pi * beta_n * np.sin(np.pi * beta_n * t))
#
#     # return np.pi * np.sin(np.pi * x) * np.cos(np.pi * t) - sum_term * np.sin(np.pi * a)
#     return init_cond - sum_term * np.sin(np.pi * a)
#
#
# # Параметры
# x = np.linspace(0, 1, 1000)
# t_values = np.linspace(0, 0.5, 100)
# a = 0.5  # Параметр a
#
# # Создание фигуры и осей для анимации
# fig, ax = plt.subplots()
# line, = ax.plot(x, y(x, 0, a), lw=2)
# ax.set_xlim(0, 1)
# ax.set_ylim(-4, 4)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.grid(True)  # Добавление сетки
# plt.pause(2)
#
# # Обновление графика в цикле
# for t in t_values:
#     y_values = y(x, t, a)
#     line.set_ydata(y_values)
#     ax.set_title(f't={t:.2f}')
#     plt.pause(0.1)  # Задержка между кадрами
#
# plt.show()
#
# # np.savetxt(r'./plots/string_break_contact/x_green_07_test.txt', x)
# # np.savetxt(r'./plots/string_break_contact/vel_green_07_test.txt', y_values)
