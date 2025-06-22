import numpy as np
from numpy import sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
import random

# ------------------------------------------------------------
# 0. User settings
# ------------------------------------------------------------
T_FINAL = 0.04  # total integration time, s
MAX_STEP = 1e-4  # max solver step
RTOL, ATOL = 1e-7, 1e-10
sample_every = 1  # record every Nth solver step

# Define 10 initial velocity sets (v1, v2) in m/s
# initial_velocities = [
#     (-10000, -1000), (-800, -7000), (-3400, -6000),
#     (-2000, -4000), (-20300, -2000), (200, 0),
#     (1000, 2000)
# ]

random.seed(552)
initial_velocities = [
    (random.randint(4000, 15000), random.randint(15000, 25000))
    for _ in range(13)
]
print("initial_velocities = [")
for v1, v2 in initial_velocities:
    print(f"    ({v1}, {v2}),")
print("]")

# ------------------------------------------------------------
# 1. System parameters
# ------------------------------------------------------------
a1, a2, a3 = 0.5, 0.5, 0.5  # geometry, m
l01, l02, l03 = 1.0, 0.5, 1.0  # natural lengths, m
k1, k2, k3 = 3.0e7, 3.0e6, 3.0e7  # spring stiffness, N/m
k_theta = 2.0e3  # torsional stiffness, N·m/rad
c = 9500.0  # damping, N·s/m
m = 1.0  # mass, kg


# ------------------------------------------------------------
# 2. Geometry and potential energy
# ------------------------------------------------------------
def geometry(y1, y2):
    dL1 = sqrt(a1 ** 2 + y1 ** 2) - l01
    dL2 = sqrt(a2 ** 2 + (y2 - y1) ** 2) - l02
    dL3 = sqrt(a3 ** 2 + y2 ** 2) - l03

    th1 = np.arctan2(y1, a1)
    th2 = np.arctan2(y2, a3)

    dL1_dy1 = y1 / sqrt(a1 ** 2 + y1 ** 2)
    dL2_dy1 = (y1 - y2) / sqrt(a2 ** 2 + (y2 - y1) ** 2)
    dth1_dy1 = a1 / (a1 ** 2 + y1 ** 2)

    dL3_dy2 = y2 / sqrt(a3 ** 2 + y2 ** 2)
    dL2_dy2 = (y2 - y1) / sqrt(a2 ** 2 + (y2 - y1) ** 2)
    dth2_dy2 = a3 / (a3 ** 2 + y2 ** 2)

    return (dL1, dL2, dL3, th1, th2,
            dL1_dy1, dL2_dy1, dth1_dy1,
            dL3_dy2, dL2_dy2, dth2_dy2)


# Potential energy function V(y1, y2)
def potential(y1, y2):
    dL1, dL2, dL3, th1, th2, *_ = geometry(y1, y2)
    return 0.5 * (k1 * dL1 ** 2 + k2 * dL2 ** 2 + k3 * dL3 ** 2 + k_theta * (th1 ** 2 + th2 ** 2))


# ------------------------------------------------------------
# 3. Equations of motion
# ------------------------------------------------------------
def eom(t, y):
    y1, y2, v1, v2 = y
    dL1, dL2, dL3, th1, th2, dL1_dy1, dL2_dy1, dth1_dy1, dL3_dy2, dL2_dy2, dth2_dy2 = geometry(y1, y2)

    Q1 = k1 * dL1 * dL1_dy1 + k2 * dL2 * dL2_dy1 + k_theta * th1 * dth1_dy1
    Q2 = k3 * dL3 * dL3_dy2 + k2 * dL2 * dL2_dy2 + k_theta * th2 * dth2_dy2

    a1_dd = (-Q1 - c * v1) / m
    a2_dd = (-Q2 - c * v2) / m
    return [v1, v2, a1_dd, a2_dd]


# ------------------------------------------------------------
# 4. Simulate trajectories
# ------------------------------------------------------------
y1_trajs = []
y2_trajs = []
# approximate "up" equilibrium
y_eq = (0.866, 0.866)

for v1_init, v2_init in initial_velocities:
    sol = solve_ivp(eom, (0, T_FINAL), [y_eq[0], y_eq[1], -v1_init, -v2_init],
                    max_step=MAX_STEP, rtol=RTOL, atol=ATOL)
    indices = np.arange(0, sol.y.shape[1], sample_every)
    y1_trajs.append(sol.y[0, indices])
    y2_trajs.append(sol.y[1, indices])

# ------------------------------------------------------------
# 5. Plot potential & trajectories in original style
# ------------------------------------------------------------
# Grid for V
y_vals = np.linspace(-2.6, 2.6, 600)
Y1, Y2 = np.meshgrid(y_vals, y_vals)
V = np.zeros_like(Y1)
for i in range(Y1.shape[0]):
    for j in range(Y1.shape[1]):
        V[i, j] = potential(Y1[i, j], Y2[i, j])

# Identify extrema
dV_max = maximum_filter(V, size=15, mode='nearest')
dV_min = minimum_filter(V, size=15, mode='nearest')
maxima_mask = (V == dV_max)
minima_mask = (V == dV_min)
maxima_coords = np.argwhere(maxima_mask)
minima_coords = np.argwhere(minima_mask)

plt.figure(figsize=(10, 7))
# filled contours
filled = plt.contourf(Y1, Y2, V, levels=70, alpha=0.75)
# contour lines
contours = plt.contour(Y1, Y2, V, levels=70, linewidths=0.8)
plt.clabel(contours, inline=True, fontsize=8)

# mark minima and maxima
if minima_coords.size > 0:
    xs_min = Y1[minima_coords[:, 0], minima_coords[:, 1]]
    ys_min = Y2[minima_coords[:, 0], minima_coords[:, 1]]
    plt.scatter(xs_min, ys_min, marker='o', facecolors='none', edgecolors='red', label='Local minima')
if maxima_coords.size > 0:
    xs_max = Y1[maxima_coords[:, 0], maxima_coords[:, 1]]
    ys_max = Y2[maxima_coords[:, 0], maxima_coords[:, 1]]
    plt.scatter(xs_max, ys_max, marker='X', facecolors='none', edgecolors='black', label='Local maxima')

# overlay trajectories
colors = plt.cm.tab10(np.linspace(0, 1, len(y1_trajs)))
for i, (y1_t, y2_t) in enumerate(zip(y1_trajs, y2_trajs)):
    plt.plot(y1_t, y2_t, color=colors[i], label=f"set {i + 1}")

plt.xlabel('y1 (m)')
plt.ylabel('y2 (m)')
plt.title('Trajectories over 2D potential energy landscape')
plt.legend(bbox_to_anchor=(0.02, 1), loc='upper left', framealpha=0.2)
plt.colorbar(filled, label='V value')
plt.grid(True)
plt.tight_layout()
plt.show()
