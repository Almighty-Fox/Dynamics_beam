# -*- coding: utf-8 -*-
"""
Two-mass arch → «слоумо» MP4
============================
1. Интегрируем систему от t = 0 до T_final, сохраняя каждую точку решателя
2. Записываем ролик arch_dynamics_slowmo.mp4:
      • 60 fps H.264   • замедление = 1/SLOW_FACTOR раз
      • без дублирования кадров: берём одну точку из каждых `stride`
"""

# ─── imports ──────────────────────────────────────────────────────────
import numpy as np
from numpy import sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.optimize import fsolve

# ─── system parameters ────────────────────────────────────────────────
a1 = a2 = a3 = 0.5  # м
l01, l02, l03 = 1.5, 0.5, 1.5  # м
k1, k2, k3 = 3.0e7, 3.0e7, 3.0e7  # Н/м
k_theta = 2.0e3  # Н·м/рад
m = 1.0  # кг

RTOL = 1e-7
ATOL = 1e-10
dt_visual = 1e-5  # шаг, который мы *храним*
max_step_solver = 1e-5
T_final = 0.01  # с

# ─── slow-motion settings ─────────────────────────────────────────────
OUTPUT_MP4 = "arch_dynamics_slowmo.mp4"
FPS_OUT = 60  # кадров/с в файле
SLOW_FACTOR = 0.001  # 0.05 → ролик ×20 медленнее

dt_frame = SLOW_FACTOR / FPS_OUT
stride = max(1, int(round(dt_frame / dt_visual)))
print(f"➡️  stride = {stride} (каждый {stride}-й интеграционный шаг будет кадром)")


# ─── helpers ──────────────────────────────────────────────────────────
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


# no external load
f1 = lambda t: 0.0
f2 = lambda t: 0.0


def eom(t, y):
    y1, y2, v1, v2 = y
    (dL1, dL2, dL3, th1, th2,
     dL1_dy1, dL2_dy1, dth1_dy1,
     dL3_dy2, dL2_dy2, dth2_dy2) = geometry(y1, y2)

    Q1 = (k1 * dL1 * dL1_dy1 +
          k2 * dL2 * dL2_dy1 +
          k_theta * th1 * dth1_dy1)

    Q2 = (k3 * dL3 * dL3_dy2 +
          k2 * dL2 * dL2_dy2 +
          k_theta * th2 * dth2_dy2)

    c = 100.0
    a1_dd = (f1(t) - Q1 - c * v1) / m
    a2_dd = (f2(t) - Q2 - c * v2) / m
    return [v1, v2, a1_dd, a2_dd]


# ─── figure & drawing routine ─────────────────────────────────────────
plt.style.use("seaborn-notebook")
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_aspect('equal')

x0, x1, x2, x3 = 0.0, a1, a1 + a2, a1 + a2 + a3
ax.set_xlim(-0.1, x3 + 0.1)
ax.set_ylim(-1.2 * max(l01, l03), 1.2 * max(l01, l03))
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Two-mass Arch – Simulation')

anchor_style = dict(marker='s', markersize=8, color='black', linestyle='None')
mass_style = dict(marker='o', markersize=10, color='tab:red', linestyle='None')

anchor_left, = ax.plot(x0, 0.0, **anchor_style)
anchor_right, = ax.plot(x3, 0.0, **anchor_style)
mass1, = ax.plot([], [], **mass_style)
mass2, = ax.plot([], [], **mass_style)
spring1, = ax.plot([], [], '-', lw=1.8)
spring2, = ax.plot([], [], '-', lw=1.8)
spring3, = ax.plot([], [], '-', lw=1.8)
text_time = ax.text(0.02, 0.95, '', transform=ax.transAxes)


def draw_arch(state, t_current):
    y1, y2 = state[:2]
    mass1.set_data([x1], [y1])
    mass2.set_data([x2], [y2])
    spring1.set_data([x0, x1], [0.0, y1])
    spring2.set_data([x1, x2], [y1, y2])
    spring3.set_data([x2, x3], [y2, 0.0])
    text_time.set_text(f"t = {t_current:8.5f} s")


# ─── integration & history collection ────────────────────────────────
history_t, history_state = [], []


def integrate_step(t0, y0, dt):
    sol = solve_ivp(
        eom, (t0, t0 + dt), y0,
        max_step=max_step_solver, rtol=RTOL, atol=ATOL, dense_output=False
    )
    history_t.extend(sol.t[1:])  # skip duplicate first point
    history_state.extend(sol.y.T[1:])
    return sol.y[:, -1]


# state, current_time = np.array([0.866, 0.866, -13313, -24115]), 0.0
# ---------- статика (эквилибриумы) -------------------------------------
def static_residuals(Y):
    y1, y2 = Y
    (dL1, dL2, dL3, th1, th2,
     dL1_dy1, dL2_dy1, dth1_dy1,
     dL3_dy2, dL2_dy2, dth2_dy2) = geometry(y1, y2)
    Q1 = k1 * dL1 * dL1_dy1 + k2 * dL2 * dL2_dy1 + k_theta * th1 * dth1_dy1
    Q2 = k3 * dL3 * dL3_dy2 + k2 * dL2 * dL2_dy2 + k_theta * th2 * dth2_dy2
    return Q1, Q2

y_eq_up = fsolve(static_residuals, (1.1, 1.1))

current_time = 0.0
K = 3.104e+07
alpha = 0.01
v1_init = -np.sqrt(K * (1 + alpha))
v2_init = -np.sqrt(K * (1 - alpha))
state = np.array([y_eq_up[0], y_eq_up[1], v1_init, v2_init])

while current_time < T_final:
    state = integrate_step(current_time, state, dt_visual)
    current_time += dt_visual

print(f"✅ интегрировано {len(history_t)} точек")

# ─── video writing (sub-sampling) ─────────────────────────────────────
writer = FFMpegWriter(
    fps=FPS_OUT,
    metadata=dict(title="Arch slow-mo", artist="Matplotlib"),
    codec="libx264",
    bitrate=3000,
    extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'main', '-preset', 'slow']
)

with writer.saving(fig, OUTPUT_MP4, dpi=200):
    for idx in range(0, len(history_t), stride):
        draw_arch(history_state[idx], history_t[idx])
        writer.grab_frame()

print(f"🎞  Видео сохранено: {OUTPUT_MP4}")
