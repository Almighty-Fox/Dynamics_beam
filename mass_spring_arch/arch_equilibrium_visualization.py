import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ----------------------------------
# 1. Параметры системы
# ----------------------------------
a1 = 0.5
a2 = 0.5
a3 = 0.5

l01 = 1.0
l02 = 0.5
l03 = 1.0

k1 = 3.0e7
k2 = 3.0e6
k3 = 3.0e7
k_theta = 2.0e3

# ----------------------------------
# 2. Остатки равновесия
# ----------------------------------
def equilibrium_residuals(Y, F1, F2):
    y1, y2 = Y
    dl1 = np.sqrt(a1**2 + y1**2) - l01
    dl2 = np.sqrt(a2**2 + (y2 - y1)**2) - l02
    dl3 = np.sqrt(a3**2 + y2**2) - l03

    ddl1_dy1 = y1 / np.sqrt(a1**2 + y1**2) if y1 != 0 else 0.0
    denom2 = a2**2 + (y2 - y1)**2
    ddl2_dy1 = (y1 - y2) / np.sqrt(denom2) if denom2 != 0 else 0.0
    ddl2_dy2 = (y2 - y1) / np.sqrt(denom2) if denom2 != 0 else 0.0
    ddl3_dy2 = y2 / np.sqrt(a3**2 + y2**2) if y2 != 0 else 0.0

    theta1 = np.arctan2(y1, a1)
    dtheta1_dy1 = a1 / (a1**2 + y1**2) if y1 != 0 else 0.0

    theta2 = np.arctan2(y2, a3)
    dtheta2_dy2 = a3 / (a3**2 + y2**2) if y2 != 0 else 0.0

    eq1 = k1 * dl1 * ddl1_dy1 + k2 * dl2 * ddl2_dy1 + k_theta * theta1 * dtheta1_dy1 - F1
    eq2 = k3 * dl3 * ddl3_dy2 + k2 * dl2 * ddl2_dy2 + k_theta * theta2 * dtheta2_dy2 - F2

    # print(y1, y2)
    # print('eq1')
    # print(k1 * dl1 * ddl1_dy1 + k2 * dl2 * ddl2_dy1 + k_theta * theta1 * dtheta1_dy1)
    # print(eq1)
    # print('eq2')
    # print(k3 * dl3 * ddl3_dy2 + k2 * dl2 * ddl2_dy2 + k_theta * theta2 * dtheta2_dy2)
    # print(eq2)

    return np.array([eq1, eq2])

# ----------------------------------
# 3. Матрица жёсткости
# ----------------------------------
def stiffness_matrix(Y):
    y1, y2 = Y
    delta_y = y2 - y1
    dl1 = np.sqrt(a1**2 + y1**2) - l01
    dl2 = np.sqrt(a2**2 + delta_y**2) - l02
    dl3 = np.sqrt(a3**2 + y2**2) - l03

    denom1 = a1**2 + y1**2
    denom2 = a2**2 + delta_y**2
    denom3 = a3**2 + y2**2

    ddl1 = y1 / np.sqrt(denom1) if denom1 != 0 else 0.0
    d2dl1 = a1**2 / denom1**(3/2) if denom1 != 0 else 0.0

    ddl2_y1 = (y1 - y2) / np.sqrt(denom2) if denom2 != 0 else 0.0
    ddl2_y2 = (y2 - y1) / np.sqrt(denom2) if denom2 != 0 else 0.0
    d2dl2_y1 = a2**2 / denom2**(3/2) if denom2 != 0 else 0.0
    d2dl2_y2 = a2**2 / denom2**(3/2) if denom2 != 0 else 0.0
    d2dl2_y1y2 = -a2**2 / denom2**(3/2) if denom2 != 0 else 0.0

    ddl3 = y2 / np.sqrt(denom3) if denom3 != 0 else 0.0
    d2dl3 = a3**2 / denom3**(3/2) if denom3 != 0 else 0.0

    theta1 = np.arctan2(y1, a1)
    dtheta1 = a1 / denom1 if denom1 != 0 else 0.0
    d2theta1 = -2 * a1 * y1 / denom1**2 if denom1 != 0 else 0.0

    theta2 = np.arctan2(y2, a3)
    dtheta2 = a3 / denom3 if denom3 != 0 else 0.0
    d2theta2 = -2 * a3 * y2 / denom3**2 if denom3 != 0 else 0.0

    K11 = (k1 * (ddl1**2 + dl1 * d2dl1)
         + k2 * (ddl2_y1**2 + dl2 * d2dl2_y1)
         + k_theta * (dtheta1**2 + theta1 * d2theta1))

    K22 = (k3 * (ddl3**2 + dl3 * d2dl3)
         + k2 * (ddl2_y2**2 + dl2 * d2dl2_y2)
         + k_theta * (dtheta2**2 + theta2 * d2theta2))

    K12 = k2 * (ddl2_y1 * ddl2_y2 + dl2 * d2dl2_y1y2)
    K21 = K12

    return np.array([[K11, K12],
                     [K21, K22]])

# ----------------------------------
# 4. Решение задачи равновесия
# ----------------------------------
def solve_for_given_forces(F1, F2):
    guesses = [
        np.array([-0.5, -0.5]),
        np.array([0.5, -0.5]) if F1 > F2 else np.array([-0.5, 0.5]),
        np.array([0.5, 0.5])
    ]

    for guess in guesses:
        # print('new guess')
        # print(guess)
        sol, info, ier, _ = fsolve(lambda Y: equilibrium_residuals(Y, F1, F2), guess, xtol=1e-9, full_output=True)
        if ier == 1:
            res_norm = np.linalg.norm(equilibrium_residuals(sol, F1, F2))
            print(guess)
            print(f'res_norm = {res_norm}')
            if res_norm < 4e-3:
                K = stiffness_matrix(sol)
                stable = np.all(np.linalg.eigvals(K) > 0)
                return sol, stable
    return None, False

# ----------------------------------
# 5. Построение арочной системы
# ----------------------------------
def plot_arch_system(y1, y2):
    P0 = np.array([0, 0])
    P1 = np.array([a1, y1])
    P2 = np.array([a1 + a2, y2])
    P3 = np.array([a1 + a2 + a3, 0])

    x = [P0[0], P1[0], P2[0], P3[0]]
    y = [P0[1], P1[1], P2[1], P3[1]]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, '-o', lw=2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Форма арочной системы')
    plt.xlabel('x [м]')
    plt.ylabel('y [м]')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# ----------------------------------
# 6. Основной запуск
# ----------------------------------
F1 = 7.55e6  # укажите желаемое значение
F2 = 5.75e6

solution, is_stable = solve_for_given_forces(F1, F2)

if solution is not None:
    print(f"Решение найдено: y1 = {solution[0]:.4f}, y2 = {solution[1]:.4f}")
    print(f"Стабильность: {'устойчивое' if is_stable else 'неустойчивое'}")
    plot_arch_system(solution[0], solution[1])
else:
    print("Решение не найдено.")
