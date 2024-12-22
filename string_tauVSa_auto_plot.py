import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def disp_and_vel_after_VI(Nx, N_max=100000):
    pi = np.pi
    # Input from the user
    print("Enter the mode number n0 for initial displacement (β_n0).")
    n0 = int(input("n0 (enter 0 if β_n0 is zero): "))
    if n0 != 0:
        beta_n0 = float(input(f"Enter the value of β_{n0}: "))
    else:
        beta_n0 = 0
        print("Initial displacement field is zero. zero mode.")

    print("\nEnter the mode number m0 for initial velocity (α_m0).")
    m0 = int(input("m0 (enter 0 if α_m0 is zero): "))
    if m0 != 0:
        alpha_m0 = float(input(f"Enter the value of α_{m0}: "))
    else:
        alpha_m0 = 0

    # Determine zeros of initial displacement field
    if beta_n0 != 0:
        x_zeros = np.array([k / n0 for k in range(0, n0 + 1)])
        print("\nThe initial shape of the string crosses the equilibrium axis at the following x positions:")
        print(x_zeros)
    else:
        x_zeros = np.linspace(0, 1, 100)

    # Input barrier position 'a' from the user
    print("\nChoose the position of the barrier 'a' from the values above (between 0 and 1).")
    a = float(input("Enter the value of 'a': "))

    # если поле скорости не в ту сторону, то меняем на противоположное
    # поле перемещений в начальный момент точно нулевое в точке барера
    if alpha_m0 * m0 * np.sin(pi * m0 * a) > 0:
        alpha_m0 = -alpha_m0

    t_lst = np.linspace(0, 0.51, 10000)
    F_lst = (-alpha_m0 * m0 * np.cos(pi * m0 * t_lst) * np.sin(pi * m0 * a) +
             beta_n0 * n0 * np.sin(pi * n0 * t_lst) * np.sin(pi * n0 * a))
    plt.figure()
    plt.plot(t_lst, F_lst)
    plt.grid()
    # plt.show()

    index_detachment = np.argmax(F_lst < 0)
    t = t_lst[index_detachment]
    print(f'Time of detachment = {t}')
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


    plt.figure()
    plt.title('Initial Displacement')
    plt.plot(x_vals, y1_init)
    plt.grid()

    plt.figure(figsize=(7, 4))
    plt.title(f'Mode = {m0}')
    plt.xlabel('x')
    plt.ylabel('Initial Velocity')
    plt.plot(x_vals, v1_init)
    plt.plot([0, 1], [0, 0], 'k')
    plt.plot([0.25, 0.25], [max(v1_init) * 1.1, min(v1_init) * 1.1], '--r')
    plt.plot([0.5, 0.5], [max(v1_init) * 1.1, min(v1_init) * 1.1], '--r')
    plt.grid()

    plt.show()


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

    return x_vals, y_vals, v_vals, n0, m0, a





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
    plt.title(f'Dis mode = {round(n0)}, Vel mode = {round(m0)}, a = {a}')
    # plt.show()


if __name__ == "__main__":
    # Constants
    # pi = np.pi
    N_max = 100000  # Maximum number of terms in sums
    Nx = 500 + 1    # Number of x points

    x_vals, y_vals, v_vals, n0, m0, a = disp_and_vel_after_VI(Nx)

    mode_decomposition_energy()


    # Plot the result
    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, y_vals, label='y(x, t=0.5)')
    plt.title(f'Displacement, Dis mode = {round(n0)}, Vel mode = {round(m0)}, a = {a}')
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('Displacement')
    plt.grid(True)
    # plt.legend()


    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, v_vals, label='v(x, t=0.5)')
    plt.title(f'Velocity, Dis mode = {round(n0)}, Vel mode = {round(m0)}, a = {a}')
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
        ax1.set_ylim([-2.5, 2.5])
        ax1.grid()

        line2, = ax2.plot(x, (u - u_prev) / dt, label="Поле скорости")
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

        ani = FuncAnimation(fig, update, frames=Nt, blit=True, interval=20)

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
        ax1.set_ylim(-2.5, 2.5)
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
        energy_line, = ax3.plot([], [], lw=2, color="black")

        # Настройки третьего графика (v(x, t))
        ax2.set_xlim(0, L)
        ax2.set_ylim(-1, 1)  # Временно, обновится динамически
        ax2.set_xlabel("x")
        ax2.set_ylabel("v(x, t)")
        ax2.set_ylim(-5, 5)
        ax2.grid(True)
        velocity_line, = ax2.plot([], [], lw=2, color="red")

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
            if (n % 100 == 0):
                line.set_data(x, u[n, :])
                time_text.set_text(f'Время: t = {time:.3f}')
                energy_line.set_data(time_array[:n + 1], energy[:n + 1])
                velocity_line.set_data(x, velocity)
                # ax3.set_ylim(1.1 * velocity.min(), 1.1 * velocity.max())  # Динамический масштаб для скорости
                plt.pause(0.001)

        # Удержание последнего кадра
        plt.ioff()
        plt.show()


    # explicit_scheme(dt=1e-5, lambda_=0.0)
    implicit_scheme(dt=1e-5, lambd=0.01)












