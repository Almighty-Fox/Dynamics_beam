import numpy as np
from numpy import sqrt
from scipy.integrate import solve_ivp
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. User-tunable settings
# ------------------------------------------------------------
T_FINAL = 0.35#0.04  # с — суммарное время интегрирования
v_init_min = 0
v_init_max = 5e4
v_step = abs(v_init_max - v_init_min) / 50
VEL_TOL = 1e-3  # м/с — «достаточно успокоилось»
CSV_OUT = Path("final_positions_4_c_100.csv")

# ------------------------------------------------------------
# 1. Parameters (как в исходном коде)
# ------------------------------------------------------------
a1 = 0.5
a2 = 0.5
a3 = 0.5  # м
l01 = 1.0
l02 = 0.5
l03 = 1.0  # м
k1 = 3.0e7
k2 = 3.0e6
k3 = 3.0e7  # Н/м
k_theta = 2.0e3  # Н·м/рад
m = 1.0  # кг
c = 100.0  # демпфирование, Н·с/м
RTOL = 1e-7
ATOL = 1e-10
MAX_STEP = 1e-4  # стабильнее на жёсткой системе


# ------------------------------------------------------------
# 2. Geometry helper
# ------------------------------------------------------------
def geometry(y1, y2):
    dL1 = sqrt(a1 ** 2 + y1 ** 2) - l01
    dL2 = sqrt(a2 ** 2 + (y2 - y1) ** 2) - l02
    dL3 = sqrt(a3 ** 2 + y2 ** 2) - l03

    theta1 = np.arctan2(y1, a1)
    theta2 = np.arctan2(y2, a3)

    dL1_dy1 = y1 / sqrt(a1 ** 2 + y1 ** 2)
    dL2_dy1 = (y1 - y2) / sqrt(a2 ** 2 + (y2 - y1) ** 2)
    dtheta1_dy1 = a1 / (a1 ** 2 + y1 ** 2)

    dL3_dy2 = y2 / sqrt(a3 ** 2 + y2 ** 2)
    dL2_dy2 = (y2 - y1) / sqrt(a2 ** 2 + (y2 - y1) ** 2)
    dtheta2_dy2 = a3 / (a3 ** 2 + y2 ** 2)

    return (dL1, dL2, dL3, theta1, theta2,
            dL1_dy1, dL2_dy1, dtheta1_dy1,
            dL3_dy2, dL2_dy2, dtheta2_dy2)


# ------------------------------------------------------------
# 3. Template: RHS of the ODE system
# ------------------------------------------------------------
def eom(t, y):
    y1, y2, v1, v2 = y
    (dL1, dL2, dL3, th1, th2,
     dL1_dy1, dL2_dy1, dth1_dy1,
     dL3_dy2, dL2_dy2, dth2_dy2) = geometry(y1, y2)

    # Generalised internal forces
    Q1 = (k1 * dL1 * dL1_dy1 +
          k2 * dL2 * dL2_dy1 +
          k_theta * th1 * dth1_dy1)
    Q2 = (k3 * dL3 * dL3_dy2 +
          k2 * dL2 * dL2_dy2 +
          k_theta * th2 * dth2_dy2)

    a1_dd = (- Q1 - c * v1) / m
    a2_dd = (- Q2 - c * v2) / m
    return (v1, v2, a1_dd, a2_dd)


# ------------------------------------------------------------
# 4. Simulation loop over the load grid
# ------------------------------------------------------------
v_initial_values = np.arange(v_init_min, v_init_max + v_step / 2, v_step)
results = []  # list of (F1, F2, y1_final, y2_final)

initial_state = np.array([1.0, 1.0, 0.0, 0.0])

for v1_amp in v_initial_values:
    for v2_amp in v_initial_values:
        initial_state = np.array([0.866, 0.866, -v1_amp, -v2_amp])

        sol = solve_ivp(
            eom, (0.0, T_FINAL),
            initial_state,
            max_step=MAX_STEP,
            rtol=RTOL, atol=ATOL, dense_output=False
        )

        # Последний валидный шаг
        y_last = sol.y[:, -1]
        y1_fin, y2_fin = y_last[0], y_last[1]

        # Простая проверка «успокоилась ли» система
        if np.hypot(y_last[2], y_last[3]) > VEL_TOL:
            print(f"⚠️  (v1={v1_amp: .1e}, v2={v2_amp: .1e}) "
                  f"к T_FINAL скорость ещё {np.hypot(y_last[2], y_last[3]):.2e} м/с")

        results.append((v1_amp, v2_amp, y1_fin, y2_fin))

        print(f"• v1={v1_amp: .1e}, v2={v2_amp: .1e}  →  "
              f"y1={y1_fin:+.4f} м, y2={y2_fin:+.4f} м")

# ------------------------------------------------------------
# 5. Save to CSV
# ------------------------------------------------------------
with CSV_OUT.open("w", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["v1_max [N]", "v2_max [N]", "y1_final [m]", "y2_final [m]"])
    writer.writerows(results)

print(f"\n✓ Готово. Таблица результатов: {CSV_OUT.resolve()}")


# ------------------------------------------------------------
# 6. Graph
# ------------------------------------------------------------
# Загрузка CSV с разделителем ';'
file_path = 'final_positions_4_c_100.csv'  # Убедитесь, что путь корректен
df = pd.read_csv(file_path, sep=';')

# Функция для назначения цвета в зависимости от y1 и y2
def assign_color(y1, y2):
    if y1 > 0 and y2 > 0:
        return 'blue'
    elif y1 < 0 and y2 > 0:
        return 'yellow'
    elif y1 > 0 and y2 < 0:
        return 'green'
    elif y1 < 0 and y2 < 0:
        return 'red'
    else:
        return 'black'  # на случай, если y1 или y2 == 0

# Применение функции к каждой строке
df['color'] = df.apply(lambda row: assign_color(row['y1_final [m]'], row['y2_final [m]']), axis=1)

# Построение графика
plt.figure(figsize=(10, 7))
plt.scatter(df['v1_max [N]'], df['v2_max [N]'], c=df['color'])
plt.xlabel('v1_max [N]')
plt.ylabel('v2_max [N]')
plt.title('Финальные состояния: классификация по y1_final и y2_final')
plt.grid(True)
plt.show()

