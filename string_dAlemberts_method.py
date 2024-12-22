# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def odd_extension(f, L):
#     """Создает нечетное продолжение функции f(x) на отрезке [-L, L]."""
#     n = len(f)
#     extended_f = np.zeros(2 * n)
#     extended_f[:n] = f
#     extended_f[n:] = -f[::-1]
#     return extended_f
#
#
# def initial_conditions(x, L):
#     """Задает начальные условия смещения u(x, 0) и скорости u_t(x, 0)."""
#     u0 = np.sin(2 * np.pi * x / L)  # Пример начального смещения
#     # u0 = np.zeros_like(x)
#     # Задаем начальную скорость v0 как ступенчатую функцию
#     v0 = np.zeros_like(x)
#     # v0 = -np.sin(2*np.pi * x / L)  # Пример начального смещения
#     v0[x < L / 3] = 0.0  # На первом участке x < L/3, скорость равна 1.0
#     v0[(x >= L / 3) & (x < 2 * L / 3)] = -1.0  # На втором участке L/3 <= x < 2L/3, скорость равна -1.0
#     v0[x >= 2 * L / 3] = 0.0  # На третьем участке x >= 2L/3, скорость равна 0.5
#
#     return u0, v0
#
#
# def d_alembert_solution(u0, v0, x, t, c, L):
#     """Вычисляет решение уравнения колебания струны методом Даламбера."""
#     # Создаем нечетное продолжение для u0 и v0
#     u0_ext = odd_extension(u0, L)
#     v0_ext = odd_extension(v0, L)
#
#     # Для каждого x вычисляем u(x, t) по формуле Даламбера
#     u = np.zeros_like(x)
#     for i in range(len(x)):
#         x_plus = x[i] + c * t
#         x_minus = x[i] - c * t
#
#         # Используем периодическое продолжение по длине 2L
#         x_plus_idx = int((x_plus % (2 * L)) / (2 * L) * len(u0_ext))
#         x_minus_idx = int((x_minus % (2 * L)) / (2 * L) * len(u0_ext))
#
#         u[i] = 0.5 * (u0_ext[x_plus_idx] + u0_ext[x_minus_idx]) + \
#                0.5 * (1 / c) * (v0_ext[x_plus_idx] - v0_ext[x_minus_idx]) * t
#     return u
#
#
# def plot_string_evolution(x, L, c, t_max, dt):
#     """Анимация колебаний струны с течением времени."""
#     u0, v0 = initial_conditions(x, L)
#     plt.figure(figsize=(10, 6))
#
#     times = np.arange(0, t_max, dt)
#     for t in times:
#         u = d_alembert_solution(u0, v0, x, t, c, L)
#         plt.clf()
#         plt.plot(x, u, label=f't = {t:.2f} s')
#         plt.ylim([-1.5, 1.5])
#         plt.xlabel('x')
#         plt.ylabel('u(x, t)')
#         plt.title('Колебание струны методом Даламбера')
#         plt.grid(True)
#         plt.legend()
#         plt.pause(0.05)
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     L = 1.0  # Длина струны
#     c = 1.0  # Скорость распространения волны
#     n_points = 1000  # Количество точек для дискретизации струны
#     x = np.linspace(0, L, n_points)
#     t_max = 2.0  # Время моделирования
#     dt = 0.005  # Шаг по времени
#
#     plot_string_evolution(x, L, c, t_max, dt)

# --------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 1. Параметры задачи
L = 1  # длина струны
c = 1  # скорость распространения волны
dx = 0.0001  # шаг по пространству
dt = 0.0001  # шаг по времени
T = 3  # максимальное время

# 2. Сетка по пространству и времени
x = np.arange(0, L, dx)  # сетка по x
n_points = len(x)

# 3. Начальные условия
# Начальная форма струны f(x)
f = np.sin(np.pi * x / L)  # f(x) = sin(pi * x / L) на [0, L]

# Начальное поле скоростей g(x)
g = np.zeros_like(x)  # начальная скорость равна нулю


def odd_extension(x, L):
    """
    Нечетное и периодическое продолжение для координаты x на интервале [0, 2L].
    """
    x_mod = x % (2 * L)  # остаток от деления на 2L
    if x_mod <= L:
        return x_mod
    else:
        return 2 * L - x_mod


def compute_f_extended(xi, f, L):
    """
    Вычисление значения f(x) с учетом нечетного и периодического продолжения.
    """
    r = odd_extension(xi, L)  # приведение к интервалу [0, 2L]
    if r <= L:
        return np.interp(r, x, f)  # интерполяция на [0, L]
    else:
        return -np.interp(2 * L - r, x, f)  # нечетное продолжение


def compute_g_integral(xi, t, g, x, L):
    """
    Вычисление интеграла от g(x) на интервале [xi - ct, xi + ct] с учетом нечетного и периодического продолжения.
    """
    start = odd_extension(xi - c * t, L)
    end = odd_extension(xi + c * t, L)
    if start < end:
        # Если интеграл лежит внутри одного периода [0, 2L]
        integral = np.trapz(g[np.logical_and(x >= start, x <= end)], x[np.logical_and(x >= start, x <= end)])
    else:
        # Интеграл пересекает границу [0, L]
        u1 = np.trapz(g[np.logical_and(x >= start, x <= L)], x[np.logical_and(x >= start, x <= L)])
        u2 = np.trapz(g[np.logical_and(x >= 0, x <= end)], x[np.logical_and(x >= 0, x <= end)])
        integral = u1 + u2
    return integral


def dalambert_solution(x, t, f, g, L, c):
    """
    Метод Даламбера для решения колебаний струны.
    """
    u = np.zeros_like(x)  # массив для значений перемещений u(x, t)
    for i, xi in enumerate(x):
        # Вычисление f(x + ct) и f(x - ct)
        f1 = compute_f_extended(xi + c * t, f, L)
        f2 = compute_f_extended(xi - c * t, f, L)

        # Вычисление интеграла от g(x) на интервале [xi - ct, xi + ct]
        integral = compute_g_integral(xi, t, g, x, L)

        # Формула Даламбера
        u[i] = 0.5 * (f1 + f2) + (1 / (2 * c)) * integral
    return u


# 4. Визуализация начальной формы струны и её эволюции
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, np.zeros_like(x), lw=2)
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)
ax.set_title('Колебания струны методом Даламбера')
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')

time_steps = int(T / dt)

# Анимация
for t in range(0, time_steps, 10):  # каждые 10 шагов времени
    current_time = t * dt
    u = dalambert_solution(x, current_time, f, g, L, c)
    line.set_ydata(u)  # обновляем данные линии
    ax.set_title(f'Колебания струны, t = {current_time:.2f}s')
    plt.pause(0.01)  # пауза для обновления графика

plt.show()




