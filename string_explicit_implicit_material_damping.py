import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from string_tauVSa_auto_plot import disp_and_vel_after_VI

# Constants
pi = np.pi
N_max = 100000  # Maximum number of terms in sums
Nx = 1000 + 1    # Number of x points

# # Prepare x values
# x_vals = np.linspace(0, 1, Nx)
#
# # Initialize y(x,t), v(x,t)
# # y_vals = np.zeros_like(x_vals)
# v_vals = np.zeros_like(x_vals)
#
# # y_vals = np.sin(np.pi * x_vals)  # Пример начального смещения
# y_vals = 0 * np.sin(np.pi * x_vals)  # Пример начального смещения
# # u0 = np.zeros_like(x)
# # Задаем начальную скорость v0 как ступенчатую функцию
# # v0 = -np.sin(2*np.pi * x / L)  # Пример начального смещения
#
# loc_bar = 0.7
# v_vals[x_vals < loc_bar - 0.5] = 0.0  # На первом участке x < L/3, скорость равна 1.0
# v_vals[(x_vals >= loc_bar - 0.5) & (x_vals < loc_bar)] = -1.0  # На втором участке L/3 <= x < 2L/3, скорость равна -1.0
# v_vals[(x_vals >= loc_bar) & (x_vals < 1.5 - loc_bar)] = 1.0
# v_vals[x_vals >= 1.5 - loc_bar] = 0.0  # На третьем участке x >= 2L/3, скорость равна 0.5

x_vals, y_vals, v_vals, n0_, m0_, a_ = disp_and_vel_after_VI(Nx)

# # ---------------- СГЛАЖИВАЕМ СКАЧКИ --------------------------------
# # Определяем "ширину" сглаживающей области вокруг точки перехода:
# delta = 0.00  # чем меньше, тем уже область сглаживания
#
# # Функция плавного перехода (smoothstep), обеспечивающая C2-гладкость
# def smooth_transition(x, start, end, f_start, f_end):
#     # Если x вне интервала сглаживания, возвращаем края
#     if x <= start:
#         return f_start
#     elif x >= end:
#         return f_end
#     # Нормализуем координату x в [0,1]
#     t = (x - start) / (end - start)
#     # Smoothstep: 3t² - 2t³
#     # Обеспечивает непрерывность значения, первой и второй производных
#     s = 3*t**2 - 2*t**3
#     return f_start + (f_end - f_start) * s
#
# # Заданные участки скорости:
# # 1) x < loc_bar - 0.5 -> v = 0.0
# # 2) loc_bar - 0.5 <= x < loc_bar -> v = -1.0
# # 3) loc_bar <= x < 1.5 - loc_bar -> v = 1.0
# # 4) x >= 1.5 - loc_bar -> v = 0.0
#
# # Определим точки разрывов:
# # Разрыв 1: переход 0.0 -> -1.0 в точке x = loc_bar - 0.5
# transition1_start = (loc_bar - 0.5) - delta
# transition1_end   = (loc_bar - 0.5) + delta
#
# # Разрыв 2: переход -1.0 -> 1.0 в точке x = loc_bar
# transition2_start = loc_bar - delta
# transition2_end   = loc_bar + delta
#
# # Разрыв 3: переход 1.0 -> 0.0 в точке x = 1.5 - loc_bar
# transition3_start = (1.5 - loc_bar) - delta
# transition3_end   = (1.5 - loc_bar) + delta
#
# # Теперь для каждого x выбираем подходящее значение:
# for i, x in enumerate(x_vals):
#     if x < transition1_start:
#         # до первого перехода
#         v_vals[i] = 0.0
#     elif transition1_start <= x <= transition1_end:
#         # область плавного перехода от 0.0 к -1.0
#         v_vals[i] = smooth_transition(x, transition1_start, transition1_end, 0.0, -1.0)
#     elif x < transition2_start:
#         # между первым и вторым переходом, здесь v = -1.0
#         v_vals[i] = -1.0
#     elif transition2_start <= x <= transition2_end:
#         # плавный переход от -1.0 к 1.0
#         v_vals[i] = smooth_transition(x, transition2_start, transition2_end, -1.0, 1.0)
#     elif x < transition3_start:
#         # участок после второго, до третьего перехода, v = 1.0
#         v_vals[i] = 1.0
#     elif transition3_start <= x <= transition3_end:
#         # плавный переход от 1.0 к 0.0
#         v_vals[i] = smooth_transition(x, transition3_start, transition3_end, 1.0, 0.0)
#     else:
#         # после третьего перехода, v = 0.0
#         v_vals[i] = 0.0
# # -------------------------------------------------------------------


# energy
def mode_decomposition_energy():
    # Число мод
    num_modes = 300

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
    plt.figure(figsize=(7, 4))
    plt.bar(modes, full_en_mode[:modes_plot], tick_label=[f'{i}' for i in modes])
    #     plt.bar(modes, full_en_mode[:modes_plot])
    plt.xlabel('Mode Number')
    plt.ylabel('Energy')
    # plt.show()

mode_decomposition_energy()


# Plot the result
plt.figure(figsize=(7, 4))
plt.plot(x_vals, y_vals, label='y(x, t=0.5)')
plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('Displacement')
plt.grid(True)
# plt.legend()


plt.figure(figsize=(7, 4))
plt.plot(x_vals, v_vals, label='v(x, t=0.5)')
plt.xlabel('x')
plt.ylabel('Velocity')
plt.grid(True)
# plt.legend()

plt.show()


# ------------------------------------------
# Уравнение: u_tt + λu_t - u_xx = 0
# ----- ЯВНАЯ СХЕМА ------------------------
def explicit_scheme(dt=6e-4, lambda_=0.1):
    # Параметры задачи
    L = 1.0               # Длина струны
    T = 3.0               # Время моделирования
    # Nx = 100 # Число пространственных узлов
    Nx = len(x_vals)
    # dt = 6e-4
    Nt = round(T / dt)  # Число временных шагов
    # lambda_ = 0.1         # Коэффициент затухания
    dx = L / (Nx - 1)     # Шаг по пространству
    c = 1.0               # Скорость распространения волны

    # Стабильность схемы
    assert c**2 * dt**2 / dx**2 < 1, "Условие устойчивости не выполняется"

    # Инициализация массивов
    # x = np.linspace(0, L, Nx)
    x = x_vals.copy()
    u = np.zeros(Nx)         # Значения функции в текущий момент времени
    u_prev = np.zeros(Nx)    # Значения функции в предыдущий момент времени
    u_next = np.zeros(Nx)    # Значения функции в следующий момент времени

    # Начальные условия
    # u[:] = np.sin(np.pi * x)       # Начальное положение струны
    u[:] = y_vals.copy()

    # Начальная скорость струны (например, косинусоида)
    # v_initial = 3 * np.sin(np.pi * x)  # Задайте вашу функцию скорости
    v_initial = v_vals.copy()

    u_prev[:] = u - v_initial * dt  # Начальная скорость равна 0


    # Энергия
    def total_energy(u, u_prev):
        kinetic = np.sum((u - u_prev)**2) / (2 * dt**2)
        potential = np.sum((np.diff(u) / dx)**2) / 2
        return (kinetic + potential) * dx

    energies = []

    # Настройка графиков
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

    line1, = ax1.plot(x, u, label="Положение струны")
    ax1.set_title("Положение струны")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, t)")
    ax1.legend()
    ax1.set_ylim([-0.3, 0.3])
    ax1.grid()

    line2, = ax2.plot(x, (u - u_prev) / dt, label="Поле скорости")
    ax2.set_title("Поле скорости")
    ax2.set_xlabel("x")
    ax2.set_ylabel("v(x, t)")
    ax2.legend()
    ax2.set_ylim([-1.5, 1.5])
    ax2.grid()

    line3, = ax3.plot([], [], label="Полная энергия")
    ax3.set_title("Полная энергия")
    ax3.set_xlabel("Время")
    ax3.set_ylabel("Энергия")
    ax3.legend()
    ax3.grid()

    time_data = []
    energy_data = []

    def update(frame):
        # global u, u_prev, u_next
        nonlocal u, u_prev, u_next

        # Вычисление нового состояния
        for i in range(1, Nx - 1):
            u_next[i] = (
                2 * u[i]
                - u_prev[i]
                + c**2 * dt**2 / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
                - lambda_ * dt * (u[i] - u_prev[i])
            )
        # Граничные условия
        u_next[0] = u_next[-1] = 0

        # Обновление массивов
        u_prev[:] = u[:]
        u[:] = u_next[:]

        # Обновление данных графиков
        line1.set_ydata(u)
        line2.set_ydata((u - u_prev) / dt)

        current_energy = total_energy(u, u_prev)
        energies.append(current_energy)
        time_data.append(frame * dt)
        energy_data.append(current_energy)
        print(time_data[-1])

        line3.set_data(time_data, energy_data)
        ax3.set_xlim(0, T)
        ax3.set_ylim(0, max(energies) * 1.1)

        # Остановить анимацию, если достигнут последний кадр
        if frame == Nt - 1:
            ani.event_source.stop()  # Остановить анимацию

        return line1, line2, line3

    ani = FuncAnimation(fig, update, frames=Nt, blit=True, interval=4)

    plt.tight_layout()
    plt.show()


# ----- НЕЯВНАЯ СХЕМА ------------------------
def implicit_scheme(dt=6e-4, lambd=0.1):
    # Параметры задачи
    L = 1.0
    T = 3.0
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

    # Построение графиков
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    fig.tight_layout(pad=4)

    # Настройки первого графика
    ax1.set_xlim(0, L)
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, t)")
    ax1.grid(True)
    line, = ax1.plot([], [], lw=2)
    time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)  # Текст для времени

    # Настройки второго графика (энергия)
    ax3.set_xlim(0, T)
    ax3.set_ylim(0, 1)  # Временно, обновится динамически
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Energy")
    ax3.grid(True)
    energy_line, = ax3.plot([], [], lw=2)

    # Настройки третьего графика (v(x, t))
    ax2.set_xlim(0, L)
    ax2.set_ylim(-1, 1)  # Временно, обновится динамически
    ax2.set_xlabel("x")
    ax2.set_ylabel("v(x, t)")
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True)
    velocity_line, = ax2.plot([], [], lw=2)

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

        # Динамическое обновление предела y на графике энергии
        ax3.set_ylim(0, 1.1 * energy[:n + 1].max())

        # Вычисление поля скорости
        velocity = (u[n, :] - u[n - 1, :]) / dt

        # Обновление графиков
        time = n * dt

        if n % 3 == 0:
            print(n)
            line.set_data(x, u[n, :])
            time_text.set_text(f'Время: t = {time:.3f}')
            energy_line.set_data(time_array[:n + 1], energy[:n + 1])
            velocity_line.set_data(x, velocity)
            # ax3.set_ylim(1.1 * velocity.min(), 1.1 * velocity.max())  # Динамический масштаб для скорости
            plt.pause(0.001)

    # Удержание последнего кадра
    plt.ioff()
    plt.show()


# ------------------------------------------
# Уравнение: u_tt - u_xx - γ * u_xxt = 0
# ----- ЯВНАЯ СХЕМА ------------------------
def explicit_scheme_material_damping(dt=6e-4, gamma_=0.1, x_vals=None, y_vals=None, v_vals=None):
    # Если не заданы, зададим некоторые тестовые данные
    if x_vals is None:
        Nx = 100
        L = 1.0
        x_vals = np.linspace(0, L, Nx)
    else:
        Nx = len(x_vals)
        L = x_vals[-1]

    if y_vals is None:
        y_vals = np.sin(np.pi * x_vals)
    if v_vals is None:
        v_vals = np.zeros_like(x_vals)

    # Параметры задачи
    T = 3.0  # Время моделирования
    c = 1.0  # Скорость волны
    dx = L / (Nx - 1)
    Nt = round(T / dt)

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

    # Настройка графиков
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

    line1, = ax1.plot(x_vals, u, label="Положение струны")
    ax1.set_title("Положение струны")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, t)")
    ax1.legend()
    ax1.set_ylim([-0.6, 0.6])
    ax1.grid()

    line2, = ax2.plot(x_vals, (u - u_prev) / dt, label="Поле скорости")
    ax2.set_title("Поле скорости")
    ax2.set_xlabel("x")
    ax2.set_ylabel("v(x, t)")
    ax2.legend()
    ax2.set_ylim([-5, 5])
    ax2.grid()

    line3, = ax3.plot([], [], label="Полная энергия")
    ax3.set_title("Полная энергия")
    ax3.set_xlabel("Время")
    ax3.set_ylabel("Энергия")
    ax3.legend()
    ax3.grid()

    time_data = []
    energy_data = []
    flag = True

    def update(frame):
        nonlocal u, u_prev, u_next, flag

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

        if flag and (frame * dt > 0.5):
            print(energy_data[-1])
            flag = False
            time_data.clear()
            energy_data.clear()
            # Принудительная перерисовка всей фигуры
            fig.canvas.draw_idle()


        # Обновление данных графиков
        line1.set_ydata(u)
        line2.set_ydata((u - u_prev) / dt)

        current_energy = total_energy(u, u_prev)
        if not flag:
            print(current_energy)
        time_data.append(frame * dt)
        # print(time_data[-1])
        energy_data.append(current_energy)

        line3.set_data(time_data, energy_data)
        ax3.set_xlim(0, T)
        ax3.set_ylim(0, max(energy_data) * 1.1 if energy_data else 1)



        # Остановить анимацию, если достигнут последний кадр
        if frame == Nt - 1:
            ani.event_source.stop()

        return line1, line2, line3

    ani = FuncAnimation(fig, update, frames=Nt, blit=True, interval=0.2)
    plt.tight_layout()
    plt.show()



def mode_decomposition_energy(x_vals, y_vals, v_vals):
    # Число мод
    num_modes = 300

    # Функция для вычисления коэффициентов модального разложения
    def modal_coefficients(x, y, num_modes):
        coefficients = []
        for k in range(1, num_modes + 1):
            phi_k = np.sin(k * np.pi * x)
            a_k = 2 * np.trapz(y * phi_k, x=x)
            coefficients.append(a_k)

        return np.array(coefficients)

    modal_coords = modal_coefficients(x_vals, y_vals, num_modes)
    modal_velocities = modal_coefficients(x_vals, v_vals, num_modes)

    eigenvalues = np.array([(k * np.pi) ** 2 for k in range(1, num_modes + 1)])

    full_en_mode = 1 / 2 * modal_coords ** 2 + 1 / 2 * eigenvalues * modal_velocities ** 2

    # Визуализация распределения энергии по модам
    modes_plot = 100
    modes = np.arange(1, modes_plot + 1)
    plt.figure(figsize=(7, 4))
    plt.bar(modes, full_en_mode[:modes_plot], tick_label=[f'{i}' for i in modes])
    plt.xlabel('Mode Number')
    plt.ylabel('Energy')
    plt.title('Energy Distribution by Mode')
    plt.show()

    # Визуализация первых 3 мод
    #     plt.figure(figsize=(12, 8))
    plt.figure()
    modal_forms = []
    for k in range(1, num_modes + 1):
        phi_k = modal_coords[k - 1] * np.sin(k * np.pi * x_vals)
        modal_forms.append(phi_k)
        plt.plot(x_vals, phi_k, label=f'{k}')

    plt.xlabel('x')
    plt.ylabel('Mode Shape')
    plt.title('First 3 Mode Shapes')
    plt.grid()
    #     plt.legend()

    plt.plot(x_vals, sum(modal_forms), '--k')
    plt.grid()

    # Визуализация первых 3 мод
    #     plt.figure(figsize=(12, 8))
    plt.figure()
    modal_vel = []
    for k in range(1, num_modes + 1):
        phi_k = modal_velocities[k - 1] * np.sin(k * np.pi * x_vals)
        modal_vel.append(phi_k)
        plt.plot(x_vals, phi_k, label=f'{k}')

    plt.xlabel('x')
    plt.ylabel('Mode Shape')
    plt.title('First 3 Mode Shapes')
    plt.grid()
    #     plt.legend()

    plt.plot(x_vals, sum(modal_vel), '--k')
    plt.grid()

    plt.show()




# explicit_scheme(dt=1e-4, lambda_=0.0)
# implicit_scheme(dt=1e-4, lambd=0.0)
explicit_scheme_material_damping(dt=1e-5, gamma_=0.0, x_vals=x_vals, y_vals=y_vals, v_vals=v_vals)
# mode_decomposition_energy(x_vals, y_vals, v_vals)












