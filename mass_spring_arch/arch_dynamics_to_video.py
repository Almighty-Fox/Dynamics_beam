# -*- coding: utf-8 -*-
"""
Two-mass arch: full-resolution simulation → MP4 video
----------------------------------------------------
1. Интегрируем систему с solve_ivp от 0 до t_end
2. Берём все точки решателя между t_start и t_end
3. Записываем видео arch_dynamics.mp4 (60 fps, H.264)
"""

import numpy as np
from numpy import sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. Параметры
# ----------------------------------------------------------------------
a1 = a2 = a3 = 0.5
l01, l02, l03 = 1.0, 0.5, 1.0
# k1, k2, k3, k_theta = 3.0e7, 3.0e6, 3.0e7, 2.0e3
k1, k2, k3, k_theta = 3.0e7, 3.0e4, 3.0e7, 2.0e3
m = 1.0

RTOL, ATOL = 1e-7, 1e-10
max_step_solver = 1e-5           # шаг интегратора
t_start, t_end = 0.3, 0.5        # ❱❱ что записываем в ролик

# ----------------------------------------------------------------------
# 2. Геометрия и силы
# ----------------------------------------------------------------------
def geometry(y1, y2):
    dL1 = sqrt(a1**2 + y1**2) - l01
    dL2 = sqrt(a2**2 + (y2 - y1)**2) - l02
    dL3 = sqrt(a3**2 + y2**2) - l03
    theta1, theta2 = np.arctan2(y1, a1), np.arctan2(y2, a3)

    dL1_dy1 = y1 / sqrt(a1**2 + y1**2)
    dL2_dy1 = (y1 - y2) / sqrt(a2**2 + (y2 - y1)**2)
    dtheta1_dy1 = a1 / (a1**2 + y1**2)

    dL3_dy2 = y2 / sqrt(a3**2 + y2**2)
    dL2_dy2 = (y2 - y1) / sqrt(a2**2 + (y2 - y1)**2)
    dtheta2_dy2 = a3 / (a3**2 + y2**2)

    return (dL1, dL2, dL3, theta1, theta2,
            dL1_dy1, dL2_dy1, dtheta1_dy1,
            dL3_dy2, dL2_dy2, dtheta2_dy2)

def f1(t):          # внешние силы (пример: линейно растущие)
    return -np.sqrt(t) * 1e7

def f2(t):
    return -np.sqrt(t) * 1e7

# ----------------------------------------------------------------------
# 3. Уравнения движения
# ----------------------------------------------------------------------
def eom(t, y):
    y1, y2, v1, v2 = y
    (dL1, dL2, dL3, th1, th2,
     dL1_dy1, dL2_dy1, dtha1_dy1,
     dL3_dy2, dL2_dy2, dtha2_dy2) = geometry(y1, y2)

    Q1 = k1*dL1*dL1_dy1 + k2*dL2*dL2_dy1 + k_theta*th1*dtha1_dy1
    Q2 = k3*dL3*dL3_dy2 + k2*dL2*dL2_dy2 + k_theta*th2*dtha2_dy2

    c = 60.0                 # демпфирование
    a1_dd = (f1(t) - Q1 - c*v1) / m
    a2_dd = (f2(t) - Q2 - c*v2) / m
    return [v1, v2, a1_dd, a2_dd]

# ----------------------------------------------------------------------
# 4. Начальные условия (пример: обе массы в 1 м над опорами)
# ----------------------------------------------------------------------
y0 = np.array([1.0, 1.0, 0.0, 0.0])

# ----------------------------------------------------------------------
# 5. Интеграция
# ----------------------------------------------------------------------
sol = solve_ivp(
    eom, (0.0, t_end), y0,
    max_step=max_step_solver, rtol=RTOL, atol=ATOL, dense_output=False
)

print('Solution is done')

# ----------------------------------------------------------------------
# 6. Отбор кадров для ролика (каждый шаг решателя между t_start и t_end)
# ----------------------------------------------------------------------
mask = sol.t >= t_start
step = 30  # пропускаем каждый 10‑й кадр
t_frames = sol.t[mask][::step]
y_frames = sol.y[:, mask][:, ::step]

# ----------------------------------------------------------------------
# 7. Настройка фигуры
# ----------------------------------------------------------------------
x0, x1, x2, x3 = 0.0, a1, a1 + a2, a1 + a2 + a3
plt.style.use("seaborn-notebook")
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_aspect('equal')
ax.set_xlim(-0.1, x3 + 0.1)
ax.set_ylim(-1.4*max(l01, l03), 1.4*max(l01, l03))
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Two-mass Arch – simulation (video)')

anchor_style = dict(marker='s', markersize=8, color='black', linestyle='None')
mass_style   = dict(marker='o', markersize=10, color='tab:red', linestyle='None')

ax.plot(x0, 0.0, **anchor_style)
ax.plot(x3, 0.0, **anchor_style)
mass1,   = ax.plot([], [], **mass_style)
mass2,   = ax.plot([], [], **mass_style)
spring1, = ax.plot([], [], '-', lw=1.8)
spring2, = ax.plot([], [], '-', lw=1.8)
spring3, = ax.plot([], [], '-', lw=1.8)
txt_time = ax.text(0.02, 0.83, '', transform=ax.transAxes)
# txt_time = ax.text(
#     x1, -1.05 * max(l01, l03), '',  # позиция под массой 1
#     ha='left', va='top',
#     fontsize=9, color='black'
# )

def draw_frame(k):
    """Обновить геометрию для кадра №k (индекс в t_frames)."""
    y1, y2 = y_frames[0, k], y_frames[1, k]

    mass1.set_data([x1], [y1])
    mass2.set_data([x2], [y2])
    spring1.set_data([x0, x1], [0.0, y1])
    spring2.set_data([x1, x2], [y1, y2])
    spring3.set_data([x2, x3], [y2, 0.0])
    txt_time.set_text(
        f"t = {t_frames[k]:5.3f} s\n"
        f"f1(t) = {f1(t_frames[k]): .2e} N\n"
        f"f2(t) = {f2(t_frames[k]): .2e} N"
    )
# ----------------------------------------------------------------------
# 8. Запись MP4
# ----------------------------------------------------------------------
fps = 60                                 # частота кадров в итоговом видео
writer = FFMpegWriter(fps=fps, bitrate=2000,
                      metadata=dict(title='Arch dynamics', artist='SciSim'))

with writer.saving(fig, "arch_dynamics.mp4", dpi=100):
    for k in tqdm(range(t_frames.size), desc="Запись видео", unit="кадр"):
        draw_frame(k)
        writer.grab_frame()

print("✔ Видео сохранено: arch_dynamics.mp4")
