import mpmath
from mpmath import mp, sin, pi, coth
import matplotlib.pyplot as plt
import time
import numpy as np

# Сила ВИ для струны, посчитанная с помощью пакета, умеющего находить обратное преобразование Лапласа

plt.figure(figsize=(8, 5))

# Задаем точность расчетов
mp.dps = 25

# Задаем переменные
a = mp.mpf('0.5')  # например, a = 0.5
k = mp.mpf('1e3')    # например, k = 1

# Задаем шаг
step_size1 = mp.mpf('0.001')  # шаг до t1, например, шаг = 0.1
step_size2 = mp.mpf('0.01')  # шаг после t1, например, шаг = 0.05
t1 = mp.mpf('0.02')          # момент времени, после которого меняется шаг
t_max = mp.mpf('0.6')

# Определяем функцию для обратного преобразования Лапласа
def expr(p):
    numerator = -pi * sin(pi * a)
    denominator = (p**2 + pi**2) * (1 / (p * (coth(p * a) + coth(p * (1 - a)))) + 1 / k)
    return numerator / denominator

# Функция для инверсного преобразования Лапласа
def inverse_laplace(t):
    return mpmath.invertlaplace(expr, t, method='talbot')

# Засекаем время начала выполнения
start_time = time.time()

# Вычисляем результат для t в диапазоне от 0 до t1 с шагом step_size1
t_values1 = [step_size1 * i for i in range(1, int(t1 / step_size1) + 1)]
# Вычисляем результат для t в диапазоне от t1 до t_max с шагом step_size2
t_values2 = [t1 + step_size2 * i for i in range(1, int((t_max - t1) / step_size2) + 1)]

# Объединяем списки значений t
t_values = t_values1 + t_values2

# Вычисляем значения функции для всех t
results = [inverse_laplace(t) for t in t_values]

# Засекаем время окончания выполнения
end_time = time.time()

# Вычисляем затраченное время
execution_time = end_time - start_time
print(f"Время выполнения программы: {execution_time:.6f} секунд")

# Преобразуем значения в формат, пригодный для matplotlib
t_values_float = [float(t) for t in t_values]
results_float = [float(result) for result in results]

print(f'First value = {results_float[0]}')

# -----------------------------------------------------------------------------
def heaviside(x):
    return np.heaviside(x, 1)

def F_tr(t, a, N):
    result = np.cos(np.pi * t)
    for k in range(1, N + 1):
        result += (
            heaviside(t - 2 * k * a) * np.cos(np.pi * (t - 2 * k * a)) +
            heaviside(t - 2 * k * (1 - a)) * np.cos(np.pi * (t - 2 * k * (1 - a)))
        )
    return -2 * np.pi * np.sin(np.pi * a) * result

N = 100  # Число членов в сумме
a_barrier = float(a)
t_max_float = float(t_max)
# Генерация данных
t_lst = np.linspace(0, t_max_float, 1000)
F_tr_values = F_tr(t_lst, a_barrier, N)
# -----------------------------------------------------------------------------

# Построение графика
# plt.plot(t_values_float, results_float, 'o-', markersize=2, color='blue', markeredgecolor='red', linewidth=2)
plt.plot(t_values_float, results_float, color='blue', linewidth=2, label='Numerical')
plt.plot(t_lst, F_tr_values, '--', color='red', linewidth=2, label='Analytics')
x_lst_sin = np.linspace(0, float(t1) / 10, 1000)
plt.plot(x_lst_sin, -float(k) * np.sin(np.pi * x_lst_sin), '--', color='green', linewidth=2, label='k sin(pi t)')
# plt.plot([0, max(t_values_float)], [0, 0], '--', color='green', linewidth=2)
plt.xlabel('Time, s')
plt.ylabel('$F_{tr}$, N')
plt.title(f'Inverse Laplace Transform, k={int(k)}')
plt.grid(True)
plt.legend()
plt.show()