#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beam Snap Visualizer
--------------------
Независимый скрипт-проверка: задаём набор тестов (пары x_s, F0 и Ω),
интегрируем динамику бистабильной балки и показываем АНИМАЦИЮ прогиба,
а также график q1(t). По событию выводится, был ли snap-through.

Особенности:
- фикс-фикс балка, осевая сила P (>Pcr для бистабильности)
- Галёркин по формам потери устойчивости, нелинейность: γ (Σ q_k^2) q_i
- модальное демпфирование (или Релея при желании)
- робастная работа при P>Pcr: опорная частота возбуждения берётся как
  первая положительная линейная частота, иначе |ω1|

Как пользоваться:
1) Заполните список TESTS ниже кортежами (x_s [м], F0 [Н], Omega [безр.])
2) Запустите скрипт. Для каждого теста откроется окно с анимацией и
   подписью статуса (SNAP/NO SNAP). Закройте окно — начнётся следующий тест.

Зависимости: numpy, scipy, matplotlib
"""
from __future__ import annotations
import numpy as np
from numpy import sin, cos
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib import animation

# =============================================================
# Пользовательские параметры
# =============================================================
L = 0.30                 # длина балки [м]
E = 210e9                # Юнг [Па]
rho = 7800.0             # плотность [кг/м^3]
b = 0.02                 # ширина [м]
h = 0.001                # толщина [м]
S = b * h                # площадь [м^2]
I = b * h**3 / 12.0      # момент инерции [м^4]
drive_mode_index = 1    # ← 2 означает: вторая положительная собственная частота

# Критическая нагрузка Эйлера (фикс-фикс)
Pcr = 4.0*np.pi**2 * E * I / L**2
P = 1.80 * Pcr           # > 1 для бистабильности

# Число мод
N_modes = 8

# Демпфирование: модальные ζ_i
use_modal_zeta = True
zeta_default = 0.04
zeta_custom = {1: 0.01, 2: 0.01}
# Альтернатива — Релея (если use_modal_zeta=False)
rayleigh_a0 = 0.0
rayleigh_a1 = 0.0

# Тесты: список (x_s [м], F0 [Н], Omega)
TESTS = [
    (0.14, 5.0, 2.1),
    (0.075, 7.0, 4.038),
    (0.18, 0.4, 1.0),
]

# Настройки интегрирования/визуализации
n_periods = 80           # периодов возбуждения на один тест
rtol, atol = 1e-6, 1e-9
max_step_frac = 1/200.0  # не более 1/200 периода на шаг
Nx = 3001                # сетка по x для форм и анимации
anim_frames = 240        # кадров на анимацию
anim_fps = 24
start_in_well = +1       # +1 → старт в правой яме (q1=+q_eq), -1 → в левой
snap_theta = 0.5         # критерий: q1 <= -θ q_eq

# =============================================================
# Формы для балки с заделками на концах
# =============================================================
def antisymmetric_root_y(m: int) -> float:
    a = (m - 0.5) * np.pi
    b = (m + 0.5) * np.pi
    f = lambda y: np.tan(y) - y
    return brentq(f, a + 1e-6, b - 1e-6)


def build_modes(L: float, N: int, x: np.ndarray):
    Nx = x.size
    Phi = np.zeros((N, Nx))
    dPhi = np.zeros_like(Phi)
    ddPhi = np.zeros_like(Phi)

    mode_idx = 0; n_sym = 0; n_anti = 0
    while mode_idx < N:
        i = mode_idx + 1
        if i % 2 == 1:
            n_sym += 1
            n = n_sym
            k = 2.0*np.pi*n / L
            phi  = 1.0 - np.cos(k*x)
            dphi = k*np.sin(k*x)
            ddphi = k**2*np.cos(k*x)
        else:
            n_anti += 1
            y = antisymmetric_root_y(n_anti)
            k = 2.0*y / L
            A = -1.0/np.tan(y)
            C =  k/np.tan(y)
            phi  = A*np.sin(k*x) + np.cos(k*x) + C*x - 1.0
            dphi = A*k*np.cos(k*x) - k*np.sin(k*x) + C
            ddphi = -A*k**2*np.sin(k*x) - k**2*np.cos(k*x)
        # Нормировка по энергии изгиба: ∫ (φ')^2 dx = 1
        norm2 = np.trapz(dphi*dphi, x)
        s = 1.0/np.sqrt(norm2)
        Phi[mode_idx]  = phi*s
        dPhi[mode_idx] = dphi*s
        ddPhi[mode_idx]= ddphi*s
        mode_idx += 1
    return Phi, dPhi, ddPhi

# =============================================================
# Сборка модели (общая часть)
# =============================================================
x = np.linspace(0.0, L, Nx)
Phi, dPhi, ddPhi = build_modes(L, N_modes, x)
# Вектор усреднения по длине для центра масс (однородная масса → просто среднее)
cm_vec = np.trapz(Phi, x, axis=1) / L

rhoS = rho * S
M = np.zeros((N_modes, N_modes))
for i in range(N_modes):
    for j in range(i, N_modes):
        val = rhoS * np.trapz(Phi[i]*Phi[j], x)
        M[i,j] = val; M[j,i] = val

K_diag = np.array([E*I*np.trapz(ddPhi[i]**2, x) - P for i in range(N_modes)])
K = np.diag(K_diag)

# Собственные значения/векторы
w2, V_eig = eigh(K, M)
_tol = 1e-12
w2_finite = np.where(np.isfinite(w2), w2, 0.0)

# omega (>=0) для демпфирования и выводов
omega = np.sqrt(np.clip(w2_finite, 0.0, None))
# omega_signed (со знаком) — отображает линейную устойчивость мод
omega_signed = np.empty_like(w2_finite)
pos_mask = w2_finite > _tol
neg_mask = w2_finite < -_tol
zero_mask = ~(pos_mask | neg_mask)
omega_signed[pos_mask] = np.sqrt(w2_finite[pos_mask])
omega_signed[neg_mask] = -np.sqrt(-w2_finite[neg_mask])
omega_signed[zero_mask] = 0.0

# Проверка бистабильности (для q_eq нужен K11<0)
if K_diag[0] >= 0:
    raise RuntimeError("Не бистабильно: K11 >= 0 (увеличьте P/Pcr)")

# Демпфирование
if use_modal_zeta:
    zetas = np.full(N_modes, zeta_default, dtype=float)
    for k, z in zeta_custom.items():
        if 1 <= k <= N_modes:
            zetas[k-1] = z
    C_diag = 2.0 * zetas * omega * np.diag(M)
else:
    C_full = rayleigh_a0*M + rayleigh_a1*K
    C_diag = np.diag(C_full)

# Нелинейная «изотропная» связь
gamma = E*S/(2.0*L)
val = -K_diag[0]/gamma
if not np.isfinite(val) or val <= 0.0:
    raise RuntimeError("Невозможно вычислить q_eq (проверьте параметры)")
q_eq = np.sqrt(max(1e-18, val))

Minv = np.linalg.inv(M)

# =============================================================
# Общие вспомогательные
# =============================================================
def make_rhs(F0: float, phi_xs: np.ndarray, excitation_omega: float):
    def rhs(t, y):
        q  = y[:N_modes]
        qd = y[N_modes:]
        f = F0 * np.cos(excitation_omega*t)
        Fvec = phi_xs * f
        q2sum = q @ q
        nonlinear = gamma * q2sum * q
        damping = C_diag * qd
        qdd = Minv @ (Fvec - damping - (K @ q) - nonlinear)
        return np.hstack([qd, qdd])
    return rhs


# def make_snap_event(theta=snap_theta):
#     def event(t, y):
#         q1 = y[0]
#         return q1 + theta*q_eq
#     event.terminal = True
#     event.direction = -1.0
#     return event
#
# snap_event = make_snap_event(snap_theta)

def make_snap_event_cm(start_in_well_sign: int):
    """
    Событие перещёлкивания по центру масс:
    - w_cm(t) = (1/L)∫ w dx пересекает 0,
    - и направление соответствует уходу из исходной ямы:
        start_in_well=+1 → сверху вниз (direction = -1)
        start_in_well=-1 → снизу вверх (direction = +1)
    """
    want_dir = -1.0 if start_in_well_sign > 0 else +1.0

    def event(t, y):
        q  = y[:N_modes]
        # qd = y[N_modes:]  # не нужно для самого «нуля», но полезно знать:
        # wcm_dot = cm_vec @ qd
        wcm = cm_vec @ q
        return wcm

    event.terminal  = True      # остановить интегратор при первом таком пересечении
    event.direction = want_dir  # требуем правильный знак скорости w_cm' в момент пересечения
    return event


# # Выбор опорной частоты для конкретного теста
# def choose_drive_omega():
#     pos_idx = np.where(omega > 0.0)[0]
#     if pos_idx.size > 0:
#         return omega[pos_idx[0]]
#     return abs(omega_signed[0])  # если положительных нет

def choose_drive_omega():
    """
    Возвращает |ω| выбранной положительной моды (1-based) по drive_mode_index.
    Если положительных мод меньше, чем запросили, берём максимально доступную положительную.
    Если положительных вообще нет, возвращаем abs(omega_signed[0]) как фолбэк.
    """
    print(f'fr = {omega[:7]}')
    print(omega[1:7] / omega[1])
    pos_idx = np.flatnonzero(omega > 0.0)  # индексы мод с положительной частотой
    if pos_idx.size > 0:
        # желаемый 0-based индекс в массиве положительных
        k = drive_mode_index - 1
        # зажмём в допустимые границы [0, pos_idx.size-1]
        k = max(0, min(k, pos_idx.size - 1))
        return omega[pos_idx[k]]
    # фолбэк: нет положительных частот → берём |ω первой моды со знаком|
    return abs(omega_signed[0])


def linear_modal_energies(Q: np.ndarray, Qd: np.ndarray):
    """
    Линейные модальные энергии в базисе собственных векторов линеризованной задачи около w=0.
    Вход:
      Q  : (N_modes, Nt) — координаты в базисе Phi
      Qd : (N_modes, Nt) — скорости в базисе Phi
    Выход:
      E_lin : (N_modes, Nt) — энергии E_i(t) = 0.5*(zeta_dot^2 + |omega|^2 zeta^2)
      Z, Zd : (N_modes, Nt) — модальные координаты и скорости в лин. базисе (на всякий случай)
    """
    # Переход к линейным модальным координатам: q = V_eig * zeta   (V_eig^T M V_eig = I)
    # => zeta = V_eig^T M q
    Z  = V_eig.T @ (M @ Q)
    Zd = V_eig.T @ (M @ Qd)
    omega_lin = np.abs(omega_signed)       # |ω_i|, чтобы энергия была неотрицательной и осмысленной
    # E_lin = 0.5*(Zd**2 + (omega_lin[:, None]**2)*(Z**2))
    E_lin = 0.5 * ((omega_lin[:, None]**2)*(Z**2))
    return E_lin, Z, Zd


# Анимация одного теста
def animate_test(x_s: float, F0: float, Omega: float):
    # --- частота возбуждения и интегрирование ---
    omega_ref = choose_drive_omega()
    print(f'omega_ref = {omega_ref}')
    excitation_omega = Omega * max(omega_ref, 1e-12)
    if not np.isfinite(excitation_omega) or excitation_omega <= 0:
        excitation_omega = 1e-6
    T_drive = 2.0*np.pi / excitation_omega
    t_final = n_periods * T_drive
    max_step = max_step_frac * T_drive

    y0 = np.zeros(2*N_modes)
    y0[0] = np.sign(start_in_well) * q_eq

    phi_xs = np.array([np.interp(x_s, x, Phi[i]) for i in range(N_modes)])
    rhs = make_rhs(F0, phi_xs, excitation_omega)

    snap_event = make_snap_event_cm(start_in_well)

    sol = solve_ivp(rhs, (0.0, t_final), y0, method="RK45",
                    rtol=rtol, atol=atol, max_step=max_step,
                    events=snap_event)

    t  = sol.t
    Q  = sol.y[:N_modes, :]
    Qd = sol.y[N_modes:, :]

    snapped = (sol.t_events[0].size > 0)
    t_snap  = sol.t_events[0][0] if snapped else None

    # --- данные для верхней анимации ---
    ids = np.linspace(0, t.size-1, anim_frames, dtype=int)
    profiles = np.zeros((anim_frames, x.size))
    for kf, idx in enumerate(ids):
        profiles[kf] = Q[:, idx] @ Phi

    # --- энергии первых 5 мод как функции времени ---
    E_lin, _, _ = linear_modal_energies(Q, Qd)  # (N_modes, Nt)
    E5 = E_lin[:5, :]                           # (5, Nt)

    # --- фигура ---
    fig = plt.figure(figsize=(9.6, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.3], hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # Верх: профиль
    ln, = ax1.plot([], [], lw=2)
    ax1.set_xlim(0, L)
    ymin, ymax = 1.1*np.min(profiles), 1.1*np.max(profiles)
    if ymin == ymax:
        ymin -= 1e-6; ymax += 1e-6
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel("x [м]"); ax1.set_ylabel("w(x,t) [м]")
    title_text = ax1.set_title("")
    ax1.grid(True, ls='--', alpha=0.3)

    # Низ: 5 кривых E_i(t)
    lines = []
    for i in range(5):
        (li,) = ax2.plot(t, E5[i], lw=1.5, label=f"мода {i+1}")
        lines.append(li)
    ax2.set_xlabel("t [с]"); ax2.set_ylabel("Энергия E_i(t)")
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.legend(loc="upper right", ncols=2, fontsize=9, framealpha=0.9)

    # вертикальная «каретка» времени, синхронная с верхней анимацией
    cursor = ax2.axvline(t[0], color='k', lw=1.2, ls='-')
    if snapped:
        ax2.axvline(t_snap, color='r', lw=1.5, ls='--')

    status = "SNAP" if snapped else "NO SNAP"
    title_static = f"x_s/L={x_s/L:.3f}, F0={F0:.4g} N, Ω={Omega:.3f}, status: {status}"

    def init():
        ln.set_data([], [])
        title_text.set_text(title_static)
        cursor.set_xdata(t[0])
        return (ln, title_text, cursor, *lines)

    def update(i):
        ln.set_data(x, profiles[i])
        ti = t[ids[i]]
        title_text.set_text(f"{title_static} | t={ti:.5f} с")
        cursor.set_xdata(ti)
        return (ln, title_text, cursor, *lines)

    ani = animation.FuncAnimation(
        fig, update, frames=anim_frames, init_func=init,
        blit=True, interval=1000/anim_fps, repeat=False
    )

    if snapped:
        ax2.set_title(f"Энергии первых 5 мод (SNAP в t={t_snap:.5f} с)")
    else:
        ax2.set_title("Энергии первых 5 мод")

    # plt.tight_layout()
    plt.show()


# =============================================================
# Запуск всех тестов
# =============================================================
if __name__ == "__main__":
    print(f"P/Pcr={P/Pcr:.3f}; N_modes={N_modes}; zeta_default={zeta_default}")
    for (x_s, F0, Omega) in TESTS:
        print(f"Test: x_s/L={x_s/L:.3f}, F0={F0:.4g} N, Omega={Omega}")
        animate_test(x_s, F0, Omega)
    print("Done.")
