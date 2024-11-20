import numpy as np
import matplotlib.pyplot as plt

# Constants
pi = np.pi
N_max = 10000  # Maximum number of terms in sums
Nx = 1000    # Number of x points

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

# Time of detachment
t = 0.5

# Prepare x values
x_vals = np.linspace(0, 1, Nx)

# Initialize y(x,t), v(x,t)
y_vals = np.zeros_like(x_vals)
v_vals = np.zeros_like(x_vals)

# Compute y1
y1 = np.zeros_like(x_vals)
v1 = np.zeros_like(x_vals)
if beta_n0 != 0:
    y1 += beta_n0 * np.sin(pi * n0 * x_vals) * np.cos(pi * n0 * t)
    v1 += -beta_n0 * np.sin(pi * n0 * x_vals) * np.pi * n0 * np.sin(pi * n0 * t)
if alpha_m0 != 0:
    y1 += alpha_m0 * np.sin(pi * m0 * x_vals) * np.sin(pi * m0 * t)
    v1 += alpha_m0 * np.sin(pi * m0 * x_vals) * np.pi * m0 * np.cos(pi * m0 * t)

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
    plt.title('Energy Distribution by Mode')
    # plt.show()

mode_decomposition_energy()


# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(x_vals, -y_vals, label='y(x, t=0.5)')
plt.title('Displacement Field y(x, t=0.5)')
plt.xlabel('x')
plt.ylabel('y(x, t)')
plt.grid(True)
plt.legend()


plt.figure(figsize=(10, 6))
plt.plot(x_vals, -v_vals, label='v(x, t=0.5)')
plt.title('Velocity Field y(x, t=0.5)')
plt.xlabel('x')
plt.ylabel('v(x, t)')
plt.grid(True)
plt.legend()

plt.show()
