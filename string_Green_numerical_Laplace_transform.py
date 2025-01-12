import numpy as np
from matplotlib import pyplot as plt

# Constants
pi = np.pi
N_max = 100000  # Maximum number of terms in sums
Nx = 1000 + 1    # Number of x points


def VI_dis_vel(t, x_vals, n0, beta_n0, m0, alpha_m0, a=0.5):
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

    # print(f'Time of detachment = {t}')
    # Time of detachment
    # t = 0.5

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

    return y_vals, v_vals, y1_init, v1_init


loc_bar = 0.5
alpha_coef_lst = [0, 0, 0, 0, 1]
beta_coef_lst = [0]

t_lst = np.linspace(0, 0.51, 10000)

F_lst = np.zeros(len(t_lst))
for m0_, alpha_m0 in enumerate(alpha_coef_lst):
    m0 = m0_ + 1
    F_lst += 2 * np.pi * (-alpha_m0 * m0 * np.cos(pi * m0 * t_lst) * np.sin(pi * m0 * loc_bar))
for n0_, beta_n0 in enumerate(beta_coef_lst):
    n0 = n0_ + 1
    F_lst += 2 * np.pi * (beta_n0 * n0 * np.sin(pi * n0 * t_lst) * np.sin(pi * n0 * loc_bar))

plt.figure(figsize=(7, 4))
plt.title('Force')
plt.plot(t_lst, F_lst)
plt.grid()
plt.xlabel('time')
plt.ylabel('Force')

index_detachment = 0
for i in range(len(F_lst) - 1):
    if F_lst[i+1] * F_lst[i] < 0:
        index_detachment = i
        break
t = t_lst[index_detachment]
# t = 0.139
print(f't = {t}')

y_vals_sum = np.zeros(Nx)
v_vals_sum = np.zeros(Nx)
y1_init_sum = np.zeros(Nx)
v1_init_sum = np.zeros(Nx)
# Prepare x values
x_vals = np.linspace(0, 1, Nx)

for m0, alpha_m0 in enumerate(alpha_coef_lst):
    y_vals, v_vals, y1_init, v1_init = VI_dis_vel(t, x_vals, n0=0, beta_n0=0, m0=m0+1, alpha_m0=alpha_m0, a=loc_bar)
    y_vals_sum += y_vals
    v_vals_sum += v_vals
    y1_init_sum += y1_init
    v1_init_sum += v1_init

for n0, beta_n0 in enumerate(beta_coef_lst):
    y_vals, v_vals, y1_init, v1_init = VI_dis_vel(t, x_vals, n0=n0+1, beta_n0=beta_n0, m0=0, alpha_m0=0, a=loc_bar)
    y_vals_sum += y_vals
    v_vals_sum += v_vals
    y1_init_sum += y1_init
    v1_init_sum += v1_init

plt.figure()
plt.title('Initial Displacement')
plt.plot(x_vals, y1_init_sum)
plt.grid()

plt.figure(figsize=(7, 4))
plt.xlabel('x')
plt.ylabel('Initial Velocity')
plt.plot(x_vals, v1_init_sum)
plt.plot([0, 1], [0, 0], 'k')
plt.grid()

plt.show()

plt.figure(figsize=(7, 4))
plt.title('Displacement')
plt.plot(x_vals, y_vals_sum, label='y(x, t=0.5)')
if max(abs(y_vals_sum)) < 0.1:
    plt.ylim(-1, 1)
# plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('Displacement')
plt.grid(True)
# plt.legend()


plt.figure(figsize=(7, 4))
plt.title('Velocity')
plt.plot(x_vals, v_vals_sum, label='v(x, t=0.5)')
plt.xlabel('x')
plt.ylabel('Velocity')
plt.grid(True)
# plt.legend()

plt.show()


print('done')












