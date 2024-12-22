import numpy as np
import sys
import time
from matplotlib import pyplot as plt

# Constants
pi = np.pi
N_max = 100000  # Maximum number of terms in sums
Nx = 1000 + 1    # Number of x points


def VI_dis_vel(n0, beta_n0, m0, alpha_m0, a=0.5):
    # Input from the user
    # n0 = 0  # initial displacement (β_n0)
    # beta_n0 = 0

    # m0 = 1  # initial velocity (α_m0)
    # alpha_m0 = 1

    # a = 0.5  # position of the barrier

    # если поле скорости не в ту сторону, то меняем на противоположное
    # поле перемещений в начальный момент точно нулевое в точке барера
    if alpha_m0 * m0 * np.sin(pi * m0 * a) > 0:
        alpha_m0 = -alpha_m0

    t_lst = np.linspace(0, 0.51, 10000)
    F_lst = (-alpha_m0 * m0 * np.cos(pi * m0 * t_lst) * np.sin(pi * m0 * a) +
             beta_n0 * n0 * np.sin(pi * n0 * t_lst) * np.sin(pi * n0 * a))


    index_detachment = np.argmax(F_lst < 0)
    t = t_lst[index_detachment]
    # print(f'Time of detachment = {t}')
    # Time of detachment
    # t = 0.5


    # Prepare x values
    x_vals = np.linspace(0, 1, Nx)

    # Initialize y(x,t), v(x,t)
    y_vals = np.zeros_like(x_vals)
    v_vals = np.zeros_like(x_vals)

    # Compute initial dis and vel
    y1 = np.zeros_like(x_vals)
    v1 = np.zeros_like(x_vals)
    y1_init = np.zeros_like(x_vals)
    v1_init = np.zeros_like(x_vals)
    if beta_n0 != 0:
        y1 += beta_n0 * np.sin(pi * n0 * x_vals) * np.cos(pi * n0 * t)
        y1_init += beta_n0 * np.sin(pi * n0 * x_vals)
        v1 += -beta_n0 * np.sin(pi * n0 * x_vals) * np.pi * n0 * np.sin(pi * n0 * t)
    if alpha_m0 != 0:
        y1 += alpha_m0 * np.sin(pi * m0 * x_vals) * np.sin(pi * m0 * t)
        v1 += alpha_m0 * np.sin(pi * m0 * x_vals) * np.pi * m0 * np.cos(pi * m0 * t)
        v1_init += alpha_m0 * np.sin(pi * m0 * x_vals) * np.pi * m0


    # Compute y2
    y2 = np.zeros_like(x_vals)
    v2 = np.zeros_like(x_vals)
    if alpha_m0 != 0:
        sin_pi_m0_a = np.sin(pi * m0 * a)
        sin_pi_m0_x = np.sin(pi * m0 * x_vals)
        sin_pi_m0_t = np.sin(pi * m0 * t)
        cos_pi_m0_t = np.cos(pi * m0 * t)

        if beta_n0 != 0:
            y_term1_beta = -(pi * n0 * t * np.cos(pi * n0 * t) - np.sin(pi * n0 * t)) / (2 * n0)
            v_term1_beta = np.sin(pi * n0 * a) * np.sin(pi * n0 * x_vals) * pi * t * np.sin(pi * n0 * t) / (-2 * n0)
        else:
            y_term1_beta = 0
            v_term1_beta = 0

        if alpha_m0 != 0:
            y_term1_alpha = - (pi * t) / (2 * m0) * sin_pi_m0_t * sin_pi_m0_a * sin_pi_m0_x
            v_term1_alpha = sin_pi_m0_a * sin_pi_m0_x * (sin_pi_m0_t / m0 + pi * t * cos_pi_m0_t)
        else:
            y_term1_alpha = 0
            v_term1_alpha = 0

        # Sum over k, excluding k = m0
        sum_k_y_alpha, sum_k_y_beta = np.zeros_like(x_vals), np.zeros_like(x_vals)
        sum_k_v_alpha, sum_k_v_beta = np.zeros_like(x_vals), np.zeros_like(x_vals)

        for k in range(1, N_max + 1):
            sin_pi_k_a = np.sin(pi * k * a)
            sin_pi_k_x = np.sin(pi * k * x_vals)
            sin_pi_k_t = np.sin(pi * k * t)
            cos_pi_k_t = np.cos(pi * k * t)

            if k != m0:
                numerator = sin_pi_k_a * sin_pi_k_x * (cos_pi_k_t - cos_pi_m0_t)
                denominator = k ** 2 - m0 ** 2
                sum_k_y_alpha += numerator / denominator

                sum_k_v_alpha += sin_pi_k_a * sin_pi_k_x * (
                            1 / (m0 + k) * (sin_pi_m0_t + sin_pi_k_t) + 1 / (m0 - k) * (
                                sin_pi_m0_t - sin_pi_k_t))

            if k != n0:
                sum_k_y_beta += 1 / k * sin_pi_k_a * sin_pi_k_x / (n0**2 - k**2) * (n0 * sin_pi_k_t - k * np.sin(pi * n0 * t))
                sum_k_v_beta += sin_pi_k_a * sin_pi_k_x / (n0**2 - k**2) * (np.cos(pi * n0 * t) - np.cos(pi * k * t))




        y2 = ((4 / pi) * alpha_m0 * m0 * sin_pi_m0_a * (y_term1_alpha + sum_k_y_alpha) +
              (4 / pi) * beta_n0 * n0 * np.sin(pi * n0 * a) * (y_term1_beta + sum_k_y_beta))
        v2 = (-2 * alpha_m0 * m0 * sin_pi_m0_a * (v_term1_alpha + sum_k_v_alpha) -
              4 * beta_n0 * n0 * np.sin(pi * n0 * a) * (v_term1_beta + sum_k_v_beta))

    # Total y(x,t)
    y_vals = y1 + y2
    v_vals = v1 + v2


    # energy
    def mode_decomposition_energy():
        # Число мод
        num_modes = 30

        # Функция для вычисления коэффициентов модального разложения
        def modal_coefficients(x, y, num_modes):
            coefficients = []
            for k in range(1, num_modes + 1):
                phi_k = np.sin(k * np.pi * x)
                a_k = 2 * np.trapz(y * phi_k, x=x)
                coefficients.append(a_k)

            return np.array(coefficients)

        xx = x_vals.copy()
        y = y_vals
        v = v_vals
        modal_coords = modal_coefficients(xx, y, num_modes)
        modal_velocities = modal_coefficients(xx, v, num_modes)

        eigenvalues = np.array([(k * np.pi) ** 2 for k in range(1, num_modes + 1)])

        #     full_en_mode = 1 / 2 * modal_coords ** 2 + 1 / 2 * eigenvalues * modal_velocities ** 2
        full_en_mode = 1 / 2 * eigenvalues * modal_coords ** 2 + 1 / 2 * modal_velocities ** 2

        # Визуализация распределения энергии по модам
        modes_plot = 20
        modes = np.arange(1, modes_plot + 1)

    mode_decomposition_energy()
    return x_vals, y_vals, v_vals


# ------------------------------------------
# Уравнение: u_tt + λu_t - u_xx = 0
# ----- НЕЯВНАЯ СХЕМА ------------------------
def implicit_scheme(x_vals, y_vals, v_vals, dt=6e-4, lambd=0.1):
    # Параметры задачи
    L = 1.0
    T = 100.0
    # Nx = 50
    Nx = len(x_vals)
    # dt = 6e-4
    # lambd = 0.1

    # Сетка
    dx = L / (Nx - 1)
    Nt = round(T / dt + 1)
    r = dt ** 2 / dx ** 2

    # Начальные условия
    # x = np.linspace(0, L, Nx)
    x = x_vals.copy()
    # u0 = np.sin(np.pi * x)  # u(x, 0) = sin(pi*x)
    u0 = y_vals.copy()
    # v0 = np.cos(np.pi * x)  # Начальная скорость v(x, 0) = cos(pi*x)
    v0 = v_vals.copy()
    u1 = u0 + v0 * dt  # u(x, Δt) = u(x, 0) + v(x, 0) * Δt
    # u1 = u0.copy()  # u(x, Δt) = u(x, 0)
    boundary_conditions = (0, 0)  # u(0, t) = u(L, t) = 0

    # Коэффициенты уравнения
    a = -r
    b = 1 + lambd * dt + 2 * r
    c = -r

    # Матрица коэффициентов
    A = np.zeros((Nx, Nx))
    for i in range(1, Nx - 1):
        A[i, i - 1] = a
        A[i, i] = b
        A[i, i + 1] = c
    A[0, 0] = A[-1, -1] = 1  # Граничные условия

    # Начальные условия
    u = np.zeros((Nt, Nx))
    u[0, :] = u0
    u[1, :] = u1

    # Граничные условия
    u[:, 0], u[:, -1] = boundary_conditions

    # Решение уравнения с построением графиков
    energy = np.zeros(Nt)
    time_array = np.zeros(Nt)

    # Вычисление полной энергии на нулевом временном шаге
    kinetic_0 = ((u[1, :] - u[0, :]) / dt) ** 2  # Кинетическая энергия на шаге 0
    potential_0 = (np.gradient(u[0, :], dx)) ** 2  # Потенциальная энергия на шаге 0
    energy[0] = 0.5 * np.sum(kinetic_0 + potential_0) * dx  # Полная энергия на шаге 0

    for n in range(1, Nt - 1):
        f = 2 * u[n, :] - u[n - 1, :] + lambd * dt * u[n, :]
        f[0] = boundary_conditions[0]  # Левая граница
        f[-1] = boundary_conditions[1]  # Правая граница
        # Решение линейной системы
        u[n + 1, :] = np.linalg.solve(A, f)

        # Вычисление полной энергии
        kinetic = ((u[n, :] - u[n - 1, :]) / dt) ** 2
        potential = (np.gradient(u[n, :], dx)) ** 2
        energy[n] = 0.5 * np.sum(kinetic + potential) * dx
        time_array[n] = n * dt  # Время для оси графика энергии

        # Вычисление поля скорости
        velocity = (u[n, :] - u[n - 1, :]) / dt

        # Обновление графиков
        time = n * dt

        sys.stdout.write(f"\rВнутреннее время текущей итерации {time:.4f}")
        sys.stdout.flush()

        if (energy[0] / energy[n] > np.exp(1)): #or (time > 700):
            plt.figure()
            plt.plot(time_array[:n+1], energy[:n+1])
            plt.grid()
            # plt.show()

            return time, time_array[:n+1], energy[:n+1]



# ------------------------------------------
# Уравнение: u_tt - u_xx - γ * u_xxt = 0
# ----- ЯВНАЯ СХЕМА ------------------------
def explicit_scheme_material_damping(dt=6e-4, gamma_=0.1, x_vals=None, y_vals=None, v_vals=None):
    T = 100  # Время моделирования
    Nt = round(T / dt)
    Nx = len(x_vals)
    L = x_vals[-1]
    c = 1.0  # Скорость волны
    dx = L / (Nx - 1)

    # Проверка условия устойчивости для классического волнового уравнения (без γ)
    # Возможно, при учёте γ могут потребоваться дополнительные условия,
    # но здесь оставим классическое:
    assert c ** 2 * dt ** 2 / dx ** 2 < 1, "Условие устойчивости не выполняется для базового волнового уравнения"

    # Инициализация массивов решения
    u = np.zeros(Nx)
    u_prev = np.zeros(Nx)
    u_next = np.zeros(Nx)

    # Начальные условия
    u[:] = y_vals.copy()
    u_prev[:] = u - v_vals * dt

    # Функция для расчёта энергии
    def total_energy(u, u_prev):
        kinetic = np.sum((u - u_prev) ** 2) / (2 * dt ** 2)
        potential = np.sum((np.diff(u) / dx) ** 2) / 2
        return (kinetic + potential) * dx


    time_data = [0]
    energy_data = []

    # Вычисление полной энергии на нулевом временном шаге
    kinetic_0 = v_vals ** 2  # Кинетическая энергия на шаге 0
    potential_0 = (np.gradient(u, dx)) ** 2  # Потенциальная энергия на шаге 0
    energy_data.append(0.5 * np.sum(kinetic_0 + potential_0) * dx)  # Полная энергия на шаге 0

    # fig, axs = plt.subplots(3)

    for n in range(1, Nt - 1):

        # Основной вычислительный цикл
        for i in range(1, Nx - 1):
            # Вычисление второй производной по пространству для текущего и предыдущего слоя
            d2u_dx2_n = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2
            d2u_dx2_prev = (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]) / dx ** 2

            # Явная схема для:
            # u_{n+1}[i] = 2 u_n[i] - u_{n-1}[i]
            #               + dt^2 * (u_xx^n[i])
            #               + gamma_ * dt * (u_xx^n[i] - u_xx^{n-1}[i])
            # Объединяя:
            u_next[i] = (2 * u[i] - u_prev[i]
                         + dt ** 2 * d2u_dx2_n
                         + gamma_ * dt * (d2u_dx2_n - d2u_dx2_prev))

        # Граничные условия
        u_next[0] = 0
        u_next[-1] = 0

        # Обновляем слои
        u_prev[:] = u[:]
        u[:] = u_next[:]

        current_energy = total_energy(u, u_prev)
        time_data.append(n * dt)
        energy_data.append(current_energy)

        # axs[0].set_title(f'Time = {round(n * dt, 7)}')
        # axs[0].plot(x_vals, u, 'g', linewidth=1)
        # axs[1].plot(x_vals, (u - u_prev) / dt, 'g', linewidth=1)
        # axs[2].plot(time_data, energy_data)
        # axs[0].grid()
        # axs[1].grid()
        # axs[2].grid()
        #
        #
        # plt.pause(0.001)
        # axs[0].clear()
        # axs[1].clear()
        # axs[2].clear()

        sys.stdout.write(f"\rВнутреннее время текущей итерации {n * dt:.4f}")
        sys.stdout.flush()

        if (energy_data[0] / energy_data[n] > np.exp(1)):  # or (time > 700):
            # plt.figure()
            # plt.plot(time_data, energy_data)
            # plt.grid()
            # plt.show()

            return n * dt, time_data, energy_data








# Функция для обновления прогресса
def update_progress(cycle_num, iteration, total_iterations):
    progress = (iteration + 1) / total_iterations * 100
    sys.stdout.write(f"\rЦикл {cycle_num}: Выполнено {progress:.2f}%")
    sys.stdout.flush()


# Функция для замера времени
def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)} мин {secs:.2f} сек" if mins > 0 else f"{secs:.2f} сек"


# Для одних параметров
start_time = time.time()  # Время начала первого цикла
tau_lst = []
a_lst = np.arange(0.25, 0.510, 0.005).tolist()
for i, loc_bar in enumerate(a_lst):
    x_vals, y_vals, v_vals = VI_dis_vel(n0=0, beta_n0=0, m0=1, alpha_m0=1, a=loc_bar)
    # tau, time_array, energy_lst = implicit_scheme(x_vals, y_vals, v_vals, dt=1e-5, lambd=0.1)
    tau, time_array, energy_lst = explicit_scheme_material_damping(dt=5e-6, gamma_=0.02, x_vals=x_vals, y_vals=y_vals, v_vals=v_vals)
    tau_lst.append(tau)
    update_progress(1, i, len(a_lst))


end_time = time.time()  # Время окончания первого цикла
cycle1_time = end_time - start_time

print(f"\nЦикл 1 завершён. Время выполнения: {format_time(cycle1_time)}.")
print(a_lst)
print(tau_lst)

plt.figure()
plt.plot(a_lst, tau_lst)
plt.grid()
plt.show()
#
# # Для одних параметров 2
# start_time = time.time()  # Время начала первого цикла
# tau_lst = []
# a_lst = np.arange(0.25, 0.505, 0.005).tolist()
# for i, loc_bar in enumerate(a_lst):
#     x_vals, y_vals, v_vals = VI_dis_vel(n0=0, beta_n0=0, m0=3, alpha_m0=1, a=loc_bar)
#     # tau, time_array, energy_lst = implicit_scheme(x_vals, y_vals, v_vals, dt=1e-5, lambd=0.1)
#     tau, time_array, energy_lst = explicit_scheme_material_damping(dt=5e-6, gamma_=0.01, x_vals=x_vals, y_vals=y_vals, v_vals=v_vals)
#     tau_lst.append(tau)
#     update_progress(1, i, len(a_lst))
#
#
# end_time = time.time()  # Время окончания первого цикла
# cycle1_time = end_time - start_time
#
# print(f"\nЦикл 1 завершён. Время выполнения: {format_time(cycle1_time)}.")
# print(a_lst)
# print(tau_lst)
#
# plt.figure()
# plt.plot(a_lst, tau_lst)
# plt.grid()
# # plt.show()
#
# # Для одних параметров 3
# start_time = time.time()  # Время начала первого цикла
# tau_lst = []
# a_lst = np.arange(0.25, 0.505, 0.005).tolist()
# for i, loc_bar in enumerate(a_lst):
#     x_vals, y_vals, v_vals = VI_dis_vel(n0=0, beta_n0=0, m0=5, alpha_m0=1, a=loc_bar)
#     # tau, time_array, energy_lst = implicit_scheme(x_vals, y_vals, v_vals, dt=1e-5, lambd=0.1)
#     tau, time_array, energy_lst = explicit_scheme_material_damping(dt=5e-6, gamma_=0.01, x_vals=x_vals, y_vals=y_vals, v_vals=v_vals)
#     tau_lst.append(tau)
#     update_progress(1, i, len(a_lst))
#
#
# end_time = time.time()  # Время окончания первого цикла
# cycle1_time = end_time - start_time
#
# print(f"\nЦикл 1 завершён. Время выполнения: {format_time(cycle1_time)}.")
# print(a_lst)
# print(tau_lst)
#
# plt.figure()
# plt.plot(a_lst, tau_lst)
# plt.grid()
# # plt.show()

# # Для одних параметров 4
# start_time = time.time()  # Время начала первого цикла
# tau_lst = []
# a_lst = np.arange(0.25, 0.506, 0.005).tolist()
# for i, loc_bar in enumerate(a_lst):
#     x_vals, y_vals, v_vals = VI_dis_vel(n0=0, beta_n0=0, m0=6, alpha_m0=1, a=loc_bar)
#     # tau, time_array, energy_lst = implicit_scheme(x_vals, y_vals, v_vals, dt=1e-5, lambd=0.1)
#     tau, time_array, energy_lst = explicit_scheme_material_damping(dt=5e-6, gamma_=0.01, x_vals=x_vals, y_vals=y_vals, v_vals=v_vals)
#     tau_lst.append(tau)
#     update_progress(1, i, len(a_lst))
#
#
# end_time = time.time()  # Время окончания первого цикла
# cycle1_time = end_time - start_time
#
# print(f"\nЦикл 1 завершён. Время выполнения: {format_time(cycle1_time)}.")
# print(a_lst)
# print(tau_lst)
#
# plt.figure()
# plt.plot(a_lst, tau_lst)
# plt.grid()
# plt.show()

print('done')












