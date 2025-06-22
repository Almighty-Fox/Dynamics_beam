import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ----------------------------------
# 1. Параметры (как в вашем оригинале)
# ----------------------------------
a1 = 0.5
a2 = 0.5
a3 = 0.5

l01 = 1.0
l02 = 0.5
l03 = 1.0

k1 = 3.0e7
k2 = 3.0e3
k3 = 3.0e6
k_theta = 2.0e3

# ----------------------------------
# 2. Функция-остаток
# ----------------------------------
def equilibrium_residuals(Y, F1, F2):
    y1, y2 = Y

    # ∆l1, ddl1_dy1
    dl1 = np.sqrt(a1**2 + y1**2) - l01
    denom1 = a1**2 + y1**2
    ddl1_dy1 = (y1 / np.sqrt(denom1)) if denom1 != 0 else 0.0

    # ∆l2, ddl2_dy1, ddl2_dy2
    delta_y = y2 - y1
    dl2 = np.sqrt(a2**2 + delta_y**2) - l02
    denom2 = a2**2 + delta_y**2
    if denom2 != 0:
        ddl2_dy1 = (y1 - y2) / np.sqrt(denom2)
        ddl2_dy2 = (y2 - y1) / np.sqrt(denom2)
    else:
        ddl2_dy1 = ddl2_dy2 = 0.0

    # ∆l3, ddl3_dy2
    dl3 = np.sqrt(a3**2 + y2**2) - l03
    denom3 = a3**2 + y2**2
    ddl3_dy2 = (y2 / np.sqrt(denom3)) if denom3 != 0 else 0.0

    # θ1, dθ1/dy1
    theta1 = np.arctan2(y1, a1)
    denom_theta1 = a1**2 + y1**2
    dtheta1_dy1 = (a1 / denom_theta1) if denom_theta1 != 0 else 0.0

    # θ2, dθ2/dy2
    theta2 = np.arctan2(y2, a3)
    denom_theta2 = a3**2 + y2**2
    dtheta2_dy2 = (a3 / denom_theta2) if denom_theta2 != 0 else 0.0

    eq1 = k1 * dl1 * ddl1_dy1 + k2 * dl2 * ddl2_dy1 + k_theta * theta1 * dtheta1_dy1 - F1
    eq2 = k3 * dl3 * ddl3_dy2 + k2 * dl2 * ddl2_dy2 + k_theta * theta2 * dtheta2_dy2 - F2

    # print('eq1')
    # print(k1 * dl1 * ddl1_dy1 + k2 * dl2 * ddl2_dy1 + k_theta * theta1 * dtheta1_dy1)
    # print(eq1)
    # print('eq2')
    # print(k3 * dl3 * ddl3_dy2 + k2 * dl2 * ddl2_dy2 + k_theta * theta2 * dtheta2_dy2)
    # print(eq2)

    return np.array([eq1, eq2])

# ----------------------------------
# 3. Функция для вычисления матрицы жёсткости
# ----------------------------------
def stiffness_matrix(Y):
    y1, y2 = Y

    # dl1, ddl1, d2dl1
    dl1 = np.sqrt(a1**2 + y1**2) - l01
    denom1 = a1**2 + y1**2
    if denom1 != 0:
        ddl1_dy1 = y1 / np.sqrt(denom1)
        d2dl1_dy12 = a1**2 / denom1**(3/2)
    else:
        ddl1_dy1 = d2dl1_dy12 = 0.0

    # dl2, ddl2, d2dl2
    delta_y = y2 - y1
    dl2 = np.sqrt(a2**2 + delta_y**2) - l02
    denom2 = a2**2 + delta_y**2
    if denom2 != 0:
        ddl2_dy1 = (y1 - y2) / np.sqrt(denom2)
        ddl2_dy2 = (y2 - y1) / np.sqrt(denom2)
        d2dl2_dy12 = a2**2 / denom2**(3/2)
        d2dl2_dy22 = a2**2 / denom2**(3/2)
        d2dl2_dy1dy2 = -a2**2 / denom2**(3/2)
    else:
        ddl2_dy1 = ddl2_dy2 = d2dl2_dy12 = d2dl2_dy22 = d2dl2_dy1dy2 = 0.0

    # dl3, ddl3, d2dl3
    dl3 = np.sqrt(a3**2 + y2**2) - l03
    denom3 = a3**2 + y2**2
    if denom3 != 0:
        ddl3_dy2 = y2 / np.sqrt(denom3)
        d2dl3_dy22 = a3**2 / denom3**(3/2)
    else:
        ddl3_dy2 = d2dl3_dy22 = 0.0

    # θ1, dθ1, d2θ1
    theta1 = np.arctan2(y1, a1)
    denom_theta1 = a1**2 + y1**2
    if denom_theta1 != 0:
        dtheta1_dy1 = a1 / denom_theta1
        d2theta1_dy12 = -2 * a1 * y1 / denom_theta1**2
    else:
        dtheta1_dy1 = d2theta1_dy12 = 0.0

    # θ2, dθ2, d2θ2
    theta2 = np.arctan2(y2, a3)
    denom_theta2 = a3**2 + y2**2
    if denom_theta2 != 0:
        dtheta2_dy2 = a3 / denom_theta2
        d2theta2_dy22 = -2 * a3 * y2 / denom_theta2**2
    else:
        dtheta2_dy2 = d2theta2_dy22 = 0.0

    # Собираем K11, K22, K12 = K21
    K11 = ( k1 * (ddl1_dy1**2 + dl1 * d2dl1_dy12)
          + k2 * (ddl2_dy1**2 + dl2 * d2dl2_dy12)
          + k_theta * (dtheta1_dy1**2 + theta1 * d2theta1_dy12) )

    K22 = ( k3 * (ddl3_dy2**2 + dl3 * d2dl3_dy22)
          + k2 * (ddl2_dy2**2 + dl2 * d2dl2_dy22)
          + k_theta * (dtheta2_dy2**2 + theta2 * d2theta2_dy22) )

    K12 = k2 * (ddl2_dy2 * ddl2_dy1 + dl2 * d2dl2_dy1dy2)
    K21 = K12

    return np.array([[K11, K12],
                     [K21, K22]])

# ----------------------------------
# 4. Вспомогательная функция: попробовать найти устойчивое равновесие
# ----------------------------------
def try_find_equilibrium(F1, F2, initial_guess):
    sol, info, ier, mesg = fsolve(lambda Y: equilibrium_residuals(Y, F1, F2), initial_guess, xtol=1e-9, full_output=True)
    # print(f'ier = {ier}')
    if ier != 1:
        return None, False  # не сошёлся

    res_norm = np.linalg.norm(equilibrium_residuals(sol, F1, F2))
    # print(f'res_norm = {res_norm}')
    if res_norm > 4e-3:
        return None, False  # недостаточно точное решение

    # проверка устойчивости
    K_mat = stiffness_matrix(sol)
    eigvals = np.linalg.eigvals(K_mat)
    if np.all(eigvals > 0):
        return sol, True  # устойчивое равновесие
    else:
        return sol, False  # равновесие, но не устойчивое

# ----------------------------------
# 5. Функция: определение положения в зависимости от F1, F2
# ----------------------------------
def classify_configuration(F1, F2):

    if k1 == k3:
        # Попробуем три сценария:
        scenarios = [
            np.array([-0.5, -0.5]),  # обе массы вниз
            np.array([+0.5, -0.5]) if F1 > F2 else np.array([-0.5, +0.5]),  # одна наверху
            np.array([+0.5, +0.5])   # обе наверху
        ]

    else:
        # Попробуем four сценария:
        scenarios = [
            np.array([-0.5, -0.5]),  # обе массы вниз
            np.array([-0.5, +0.5]),
            np.array([+0.5, -0.5]),
            np.array([+0.5, +0.5])  # обе наверху
        ]

    for guess in scenarios:
        sol, stable = try_find_equilibrium(F1, F2, guess)
        # print(f'F1 = {F1}, F2 = {F2}')
        # print(sol)
        if sol is not None and stable:
            y1, y2 = sol

            # Сравниваем знаки решения и предположения
            if np.sign(y1) != np.sign(guess[0]) or np.sign(y2) != np.sign(guess[1]):
                continue  # если знаки не совпадают — переходим к следующему guess

            # Классификация по знакам y1, y2
            if y1 < 0 and y2 < 0:
                return 0  # нет перескока
            elif y1 > 0 and y2 > 0:
                return 3  # обе массы наверху
            elif y1 > 0 and y2 < 0:
                return 1  # только масса1 наверху
            elif y1 < 0 and y2 > 0:
                return 2  # только масса2 наверху
    return 4  # если не нашли равновесие, считаем, что нет перескока

# ----------------------------------
# 6. Построение карты в F1-F2
# ----------------------------------
# Задаём сетку сил
num_points = 201  # можно изменять для повышения/понижения разрешения
F_vals = np.linspace(0, 1e7, num_points)
F1_grid, F2_grid = np.meshgrid(F_vals, F_vals)

# Создаём массив категории
category = np.zeros_like(F1_grid, dtype=int)

# Пробегаем по всем точкам сетки
for i in range(num_points):
# for i in range(1):
    for j in range(num_points):
        F1_val = F1_grid[i, j]
        F2_val = F2_grid[i, j]
        category[i, j] = classify_configuration(F1_val, F2_val)

# Построим точечный график
plt.figure(figsize=(8, 7))
colors = ['blue', 'orange', 'green', 'red', 'black']
labels = ['No snap-through', 'Only mass 1 on top', 'Only mass 2 on top', 'Both masses on top', 'No solution']

for cat in range(5):
    mask = (category == cat)
    plt.scatter(F1_grid[mask], F2_grid[mask], color=colors[cat], s=6, label=labels[cat])

plt.xlabel('F1 [Н]')
plt.ylabel('F2 [Н]')
plt.title('Snap-through response map in the (F₁, F₂) parameter space')
plt.legend(loc='upper right', fontsize='small', markerscale=1.5)
plt.grid(True)
plt.show()
