import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, invertlaplace, sqrt, coth, sin, pi, mpf
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

# Set the decimal precision for mpmath
mp.dps = 15  # number of decimal digits

# Parameters
gamma = mpf('0.001')  # Material damping coefficient γ
a_float = 0.5
a = mpf(a_float)  # Position parameter a (0 < a < 1)

# Compute ω_1
omega_1 = pi * sqrt(1 - (pi ** 2 * gamma ** 2) / 4)
omega_m = lambda m: (pi * m) * sqrt(1 - ((pi * m) ** 2 * gamma ** 2) / 4)
omega_m_float = lambda m: (np.pi * m) * np.sqrt(1 - ((np.pi * m) ** 2 * float(gamma) ** 2) / 4)


# Define the Laplace-transformed function F̂_tr(s)
def F_tr_s(s, m0, alpha_m):
    sqrt_expr = sqrt(s ** 2 / (1 + s * gamma))
    coth_term1 = coth(sqrt_expr * a)
    coth_term2 = coth(sqrt_expr * (1 - a))
    numerator = -sqrt_expr * (coth_term1 + coth_term2) * alpha_m * omega_m(m0) * sin(pi * m0 * a)
    denominator = (s + ((pi * m0) ** 2 * gamma) / 2) ** 2 + omega_m(m0) ** 2
    return numerator / denominator


# Define the function to compute the inverse Laplace transform at time t
def F_tr_t(t, m0, alpha_m):
    # Use the Talbot method for numerical inversion
    return invertlaplace(lambda s: F_tr_s(s, m0, alpha_m), t, method='talbot')


# Generate time values from 0 to 0.5
t_values = np.linspace(0.0001, 0.2, 1000)  # Start from a small positive number to avoid division by zero
dt = t_values[1] - t_values[0]
F_tr_values = []

# m0 = 5
# alpha_m = 1
alpha_coef_lst = np.array([1, 0, 1])

# если поле скорости не в ту сторону, то меняем на противоположное
# поле перемещений в начальный момент точно нулевое в точке барера
vel_init_sum = 0
for m0, alpha_m0 in enumerate(alpha_coef_lst, start=1):
    vel_init_sum += alpha_m0 * omega_m_float(m0) * np.sin(np.pi * m0 * a_float)
if vel_init_sum > 0:
    print('The velocity field was reversed')
    alpha_coef_lst = -alpha_coef_lst

# Compute F_tr(t) for each time value
for t in tqdm(t_values, desc="Processing values"):
    val = 0
    for m0, alpha_m0 in enumerate(alpha_coef_lst, start=1):
        val += F_tr_t(t, m0, alpha_m0)
    F_tr_values.append(float(val))

F_tr_values = np.array(F_tr_values)

# Plot the results
plt.plot(t_values, F_tr_values)
plt.xlabel('Time t')
plt.ylabel('F_tr(t)')
plt.title('Inverse Laplace Transform of F̂_tr(s)')
plt.grid(True)


# plt.show()


def force_cross_zero(t_values, F_tr_values):
    for i, f in enumerate(F_tr_values[:-1]):
        if F_tr_values[i + 1] * F_tr_values[i] < 0:
            time_f_zero = t_values[i]
            t_values_new = t_values[:i + 1]
            F_tr_values_new = F_tr_values[:i + 1]

            return time_f_zero, t_values_new, F_tr_values_new


def trapezoidal_integration(t, t_values, f_values, k):
    under_integral_disp = f_values * np.sin(np.pi * k * (t - t_values))
    under_integral_vel = np.pi * k * f_values * np.cos(np.pi * k * (t - t_values))

    integral_disp = 0.0
    integral_vel = 0.0
    for i in range(len(t_values) - 1):
        under_mid_disp = (under_integral_disp[i] + under_integral_disp[i + 1]) / 2
        integral_disp += under_mid_disp * dt

        under_mid_vel = (under_integral_vel[i] + under_integral_vel[i + 1]) / 2
        integral_vel += under_mid_vel * dt
    return integral_disp, integral_vel


def y_xt(t, t_values, f_values):
    result_disp = 0.0
    result_vel = 0.0
    for k in range(1, k_terms + 1):
        coefficient = 2 / (np.pi * k)
        sin_x = np.sin(np.pi * k * x_values)
        sin_a = np.sin(np.pi * k * a)

        integral_disp, integral_vel = trapezoidal_integration(t, t_values, f_values, k)
        result_disp += coefficient * sin_x * sin_a * integral_disp
        result_vel += coefficient * sin_x * sin_a * integral_vel

    # print(f'VI = {result_disp[round((Nx - 1) * a_float)]}')

    result_disp_2 = 0

    for m0, alpha_m0 in enumerate(alpha_coef_lst, start=1):
        # result_disp += np.sin(np.pi * m0 * x_values) * alpha_m0 * np.sin(np.pi * m0 * t)
        # result_vel += np.sin(np.pi * m0 * x_values) * alpha_m0 * np.pi * m0 * np.cos(np.pi * m0 * t)

        # omega_m_float = lambda m: 0.87 * np.pi * m * np.sqrt(1 - ((np.pi * m) ** 2 * float(gamma) ** 2) / 4)

        result_disp += np.exp(-(np.pi * m0) ** 2 * float(gamma) / 2 * t) * alpha_m0 * np.sin(
            omega_m_float(m0) * t) * np.sin(np.pi * m0 * x_values)

        result_vel += ((alpha_m0 * np.exp(-(np.pi * m0) ** 2 * float(gamma) / 2 * t) *
                       (omega_m_float(m0) * np.cos(omega_m_float(m0) * t) -
                        (np.pi * m0) ** 2 * float(gamma) / 2 * np.sin(omega_m_float(m0) * t)))
                       * np.sin(np.pi * m0 * x_values))

        result_disp_2 += np.exp(-(np.pi * m0) ** 2 * float(gamma) / 2 * t) * alpha_m0 * np.sin(
            omega_m_float(m0) * t) * np.sin(np.pi * m0 * x_values)

    # print(f'free = {result_disp_2[round((Nx - 1) * a_float)]}')

    return result_disp, result_vel


Nx = 1000 + 1
x_values = np.linspace(0, 1, Nx)
k_terms = 1000
a = 0.5
time_f_zero, t_values, F_tr_values = force_cross_zero(t_values, F_tr_values)
print(f'detachment time = {time_f_zero}')


def plots_detachment_time():
    # поля перемещений и скорости в момент отрыва
    disp_cur, vel_cur = y_xt(time_f_zero, t_values, F_tr_values)

    print(f'interaction disp = {round(abs(disp_cur[round((Nx - 1) * a_float)] / max(abs(disp_cur)) * 100), 2)} %')
    print(disp_cur[round((Nx - 1) * a_float)])
    print(f'interaction vel = {round(abs(vel_cur[round((Nx - 1) * a_float)] / max(abs(vel_cur)) * 100), 2)} %')
    print(vel_cur[round((Nx - 1) * a_float)])

    plt.figure()
    plt.plot(x_values, disp_cur)
    plt.xlabel('x')
    plt.ylabel('disp')
    plt.title('disp')
    plt.grid(True)

    plt.figure()
    plt.plot(x_values, vel_cur)
    plt.xlabel('x')
    plt.ylabel('vel')
    plt.title('vel')
    plt.grid(True)

    plt.show()


plots_detachment_time()




# ------ dynamics ----------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Dynamic Time Visualization')

# Настраиваем первую ось для disp_cur
ax1.set_xlim(0, 1)
# ax1.set_ylim(-1.5, 1.5)  # Задайте свои пределы, если они известны
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel('x')
ax1.set_ylabel('disp_cur')
ax1.grid(True)
line1, = ax1.plot([], [], lw=2, label='Displacement')
ax1.legend()

# Настраиваем вторую ось для vel_cur
ax2.set_xlim(0, 1)
# ax2.set_ylim(-20, 20)  # Задайте свои пределы, если они известны
ax2.set_ylim(-4, 4)
ax2.set_xlabel('x')
ax2.set_ylabel('vel_cur')
ax2.grid(True)
line2, = ax2.plot([], [], lw=2, label='Velocity')
ax2.legend()


# Функция инициализации (устанавливает начальное состояние графиков)
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


# Функция обновления данных для каждого кадра
def update(frame):
    t_cur = t_values[frame]
    disp_cur, vel_cur = y_xt(t_cur, t_values[:frame + 1], F_tr_values[:frame + 1])

    # if t_cur >= 0.11205820:
    #     print('123')

    # Обновляем данные для графиков
    line1.set_data(x_values, disp_cur)
    line2.set_data(x_values, vel_cur)

    if frame % 10 == 0:
        progress_bar.update(10)

    if frame == num_frames - 1:
        print()
        print(f'interaction disp = {round(abs(disp_cur[round((Nx - 1) * a_float)] / max(abs(disp_cur)) * 100), 2)} %')
        print(f'interaction vel = {round(abs(vel_cur[round((Nx - 1) * a_float)] / max(abs(vel_cur)) * 100), 2)} %')

    return line1, line2


# Количество кадров равно количеству временных шагов
num_frames = len(t_values)

# Создаем tqdm-объект для отображения прогресса
progress_bar = tqdm(total=num_frames, desc="Animating Frames")


# После завершения анимации закрываем прогресс-бар
def finalize_progress_bar():
    progress_bar.close()


# Создаем анимацию
ani = FuncAnimation(
    fig, update, frames=num_frames, init_func=init, blit=True, interval=30, repeat=False
)


# После завершения анимации показать последний кадр
def show_last_frame():
    disp_cur, vel_cur = y_xt(t_values[-1], t_values, F_tr_values)
    line1.set_data(x_values, disp_cur)
    line2.set_data(x_values, vel_cur)
    plt.draw()


# Привязываем показ последнего кадра к закрытию анимации
ani._stop = lambda: (finalize_progress_bar(), show_last_frame())

# def plots_dynamic_time():
#     for t_id, t_cur in enumerate(t_values):
#         disp_cur, vel_cur = y_xt(t_cur, t_values[:t_id+1], F_tr_values[:t_id+1])
#
# plots_dynamic_time()


plt.show()
