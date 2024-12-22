import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Определение функции Хевисайда
def heaviside(t):
    return np.heaviside(t, 1)

# Функция для численного интегрирования формы
def integrand_form(t1, t, k, a):
    result = (-alpha_coef * alpha_n * np.cos(np.pi * alpha_n * t1) * np.sin(np.pi * alpha_n * a) +
              beta_coef * beta_n * np.sin(np.pi * beta_n * t1) * np.sin(np.pi * beta_n * a))
    return result * np.sin(np.pi * k * (t - t1))

# Функция для численного интегрирования скорости
def integrand_velocity(t1, t, k, a):
    result = (-alpha_coef * alpha_n * np.cos(np.pi * alpha_n * t1) * np.sin(np.pi * alpha_n * a) +
              beta_coef * beta_n * np.sin(np.pi * beta_n * t1) * np.sin(np.pi * beta_n * a))
    return result * np.pi * k * np.cos(np.pi * k * (t - t1))

# Основная функция формы y(x, t)
def y(x, t, a, N=100):
    sum_term = 0
    for k in range(1, N + 1):
        integral_value, _ = quad(integrand_form, 0, t, args=(t, k, a))
        sum_term += (4 / k) * np.sin(np.pi * k * x) * np.sin(np.pi * k * a) * integral_value
    init_cond = (alpha_coef * np.sin(np.pi * alpha_n * x) * np.sin(np.pi * alpha_n * t) +
                 beta_coef * np.sin(np.pi * beta_n * x) * np.cos(np.pi * beta_n * t))
    return init_cond + sum_term

# Основная функция скорости vy(x, t)
def vy(x, t, a, N=100):
    sum_term = 0
    for k in range(1, N + 1):
        integral_value, _ = quad(integrand_velocity, 0, t, args=(t, k, a))
        sum_term += (4 / k) * np.sin(np.pi * k * x) * np.sin(np.pi * k * a) * integral_value
    init_cond = (alpha_coef * np.sin(np.pi * alpha_n * x) * np.pi * alpha_n * np.cos(np.pi * alpha_n * t) -
                 beta_coef * np.sin(np.pi * beta_n * x) * np.pi * beta_n * np.sin(np.pi * beta_n * t))
    return init_cond + sum_term

# Расчет полной энергии (кинетическая + потенциальная)
def total_energy(x, y_values, vy_values):
    kinetic = 0.5 * np.trapz(vy_values**2, x)  # Кинетическая энергия
    potential = 0.5 * np.trapz((np.gradient(y_values, x))**2, x)  # Потенциальная энергия
    return kinetic + potential

# Функция для вычисления модальных коэффициентов
def modal_coefficients(x, y, num_modes):
    coefficients = []
    for k in range(1, num_modes + 1):
        phi_k = np.sin(k * np.pi * x)
        a_k = 2 * np.trapz(y * phi_k, x=x)
        coefficients.append(a_k)
    return np.array(coefficients)

# Параметры
alpha_coef, beta_coef = 1, 1
alpha_n, beta_n = 5, 0
x = np.linspace(0, 1, 1000)
t_max = 10.5
t_fine = np.linspace(0, 0.51, 10000)  # Уточненное время для определения момента отрыва
a = 0.3

if alpha_coef * alpha_n * np.sin(np.pi * alpha_n * a) > 0:
    alpha_coef = -alpha_coef

# Вычисление момента отрыва
F_lst = (-alpha_coef * alpha_n * np.cos(np.pi * alpha_n * t_fine) * np.sin(np.pi * alpha_n * a) +
         beta_coef * beta_n * np.sin(np.pi * beta_n * t_fine) * np.sin(np.pi * beta_n * a))
index_detachment = np.argmax(F_lst < 0)
t_detachment = t_fine[index_detachment]

# Обрезаем массив времени до момента отрыва
t_values = np.linspace(0, t_detachment, 100)

# Инициализация графиков
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 10))
line1, = ax1.plot(x, y(x, 0, a), lw=2)
line2, = ax2.plot(x, vy(x, 0, a), lw=2)
ax3_energy, = ax3.plot([], [], lw=2)
ax4_force, = ax4.plot([], [], lw=2)

ax5.bar(range(1, 11), np.zeros(10), align='center', alpha=0.7)
ax5.set_xlim(0.5, 10.5)
ax5.set_ylim(0, 1)
ax5.set_xlabel('Mode number')
ax5.set_ylabel('Energy distribution')
ax5.set_title('Energy distribution across modes')
ax5.grid(True)

ax1.set_xlim(0, 1)
ax1.set_ylim(-2, 2)
ax1.set_xlabel('x')
ax1.set_ylabel('y (форма)')
ax1.grid(True)

ax2.set_xlim(0, 1)
ax2.set_ylim(-7, 7)
ax2.set_xlabel('x')
ax2.set_ylabel('vy (скорость)')
ax2.grid(True)

ax3.set_xlim(0, t_detachment)
ax3.set_ylim(0, 30)
ax3.set_xlabel('t')
ax3.set_ylabel('Полная энергия')
ax3.grid(True)

ax4.set_xlim(0, t_detachment * 1.2)
ax4.set_ylim(-0.1, F_lst.max() * 1.1)
ax4.set_xlabel('t')
ax4.set_ylabel('Сила F(t)')
ax4.grid(True)

energy_values = []
force_values = []

# Анимация
for t in t_values:
    y_values = y(x, t, a)
    vy_values = vy(x, t, a)

    # Обновление графиков формы и скорости
    line1.set_ydata(y_values)
    line2.set_ydata(vy_values)

    # Вычисление энергии
    energy = total_energy(x, y_values, vy_values)
    energy_values.append(energy)
    ax3_energy.set_data(t_values[:len(energy_values)], energy_values)

    # Обновление графика силы
    force = (-alpha_coef * alpha_n * np.cos(np.pi * alpha_n * t) * np.sin(np.pi * alpha_n * a) +
             beta_coef * beta_n * np.sin(np.pi * beta_n * t) * np.sin(np.pi * beta_n * a))
    force_values.append(force)
    ax4_force.set_data(t_values[:len(force_values)], force_values)

    # Обновление распределения энергии по модам
    coefficients = modal_coefficients(x, y_values, 20)
    energy_distribution = coefficients**2  # Энергия пропорциональна квадрату коэффициентов
    ax5.cla()
    ax5.bar(range(1, 21), energy_distribution, align='center', alpha=0.7)
    ax5.set_xlim(0.5, 10.5)
    # ax5.set_ylim(0, max(energy_distribution) * 1.1)
    ax5.set_xlabel('Mode number')
    ax5.set_ylabel('Energy distribution')
    ax5.set_title('Energy distribution across modes')
    ax5.set_xticks(range(1, 21))
    ax5.grid(True)

    ax1.set_title(f't = {t:.2f}')
    plt.pause(0.05)

plt.show()
