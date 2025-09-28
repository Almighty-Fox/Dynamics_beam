#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Карта минимальной амплитуды гармонической точечной силы, необходимой
для перещёлкивания (snap-through) бистабильной балки, в зависимости от
координаты приложения силы x_s.

Особенности:
- Фиксированные концы, осевая сила P; моды нормированы по ∫(φ')^2 dx = 1
- Нелинейность вида γ (∑ q_k^2) q_i (геометрическая)
- Модальное вязкое демпфирование (или Релея при желании)
- Робастная работа при P > Pcr (линейная неустойчивость первой моды)
- Выбор опорной частоты возбуждения: первая положительная линейная
  частота; если её нет — берём |ω1| с защитой от нуля
- Индикатор прогресса через tqdm (с мягким импортом)

Алгоритм:
1) Строим моды, M, K; решаем K v = ω^2 M v
2) Вычисляем ω (≥0) и omega_signed (со знаком) — устойчивость/неустойч.
3) Выбираем ω_ref для возбуждения и задаём ω_drive = Ω·ω_ref
4) Оцениваем q_eq = sqrt(-K11/γ) и запускаем интегрирование из q1 = +q_eq
5) Для каждого x_s ищем F0*, при котором q1 ≤ -θ·q_eq за N периодов
6) Сохраняем CSV/NPZ и строим графики

Зависимости: numpy, scipy, matplotlib, tqdm (необязателен)
"""
from __future__ import annotations
import numpy as np
from numpy import sin, cos
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import csv

# tqdm (индикатор прогресса) с мягким импортом
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

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

# Критическая нагрузка Эйлера для фикс-фикс: Pcr = 4π^2 E I / L^2
Pcr = 4.0*np.pi**2 * E * I / L**2
P = 1.80 * Pcr           # > 1 для бистабильности

# Число мод (4..12 обычно достаточно)
N_modes = 8

# Демпфирование: модальные ζ_i
use_modal_zeta = True
zeta_default = 0.04
zeta_custom = {1: 0.01, 2: 0.01}
# Альтернатива — демпфирование Релея (если use_modal_zeta=False)
rayleigh_a0 = 0.0
rayleigh_a1 = 0.0

# Гармонический форсинг (частота в долях опорной линейной частоты)
Omega = 10.6              # ω_drive = Omega * ω_ref

# Временные настройки интегрирования
n_periods = 120          # число периодов воздействия в одном прогоне
rtol, atol = 1e-6, 1e-9
max_step_frac = 1/200.0  # не более 1/200 периода на шаг интегратора

# Поиск по амплитуде силы
F0_min_start = 1e-4      # начальное нижнее значение для поиска [Н]
F0_max_cap   = 2.0e3     # верхняя «крышка» [Н]; если не сработает — NaN
bisection_tol = 1e-3     # относительная точность на F0

# Сетка по координате приложения силы
n_xs = 101
xmin, xmax = 0.02*L, 0.98*L
x_s_grid = np.linspace(xmin, xmax, n_xs)

# Критерий «перещёлкивания» по первой моде
theta_threshold = 0.5    # событие: q1 ≤ -θ * q_eq

# Визуализации/сохранение
make_plots = True
save_prefix = "snapmap"

# =============================================================
# Формы для балки с заделками на концах
# =============================================================
def antisymmetric_root_y(m: int) -> float:
    """m-й положительный корень tan(y) = y, m=1,2,... в ((m-0.5)π, (m+0.5)π)."""
    a = (m - 0.5) * np.pi
    b = (m + 0.5) * np.pi
    f = lambda y: np.tan(y) - y
    return brentq(f, a + 1e-6, b - 1e-6)


def build_modes(L: float, N: int, x: np.ndarray):
    """
    Возвращает формы, нормированные по ∫ (φ')^2 dx = 1:
      Phi[i,:] = φ_i(x), dPhi[i,:] = φ'_i(x), ddPhi[i,:] = φ''_i(x).
    Нечётные i: симметричные формы φ = 1 − cos(kx), k = 2πn/L.
    Чётные i: антисимметричные формы с y: tan(y)=y, y=kL/2.
    """
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
# Сборка модели
# =============================================================
x = np.linspace(0.0, L, 1201)
Phi, dPhi, ddPhi = build_modes(L, N_modes, x)

# Массовая матрица: M_ij = ρ S ∫ φ_i φ_j dx
rhoS = rho * S
M = np.zeros((N_modes, N_modes))
for i in range(N_modes):
    for j in range(i, N_modes):
        val = rhoS * np.trapz(Phi[i]*Phi[j], x)
        M[i,j] = val; M[j,i] = val

# Линейная жёсткость: K_i = E I ∫ (φ''_i)^2 dx − P
K_diag = np.array([E*I*np.trapz(ddPhi[i]**2, x) - P for i in range(N_modes)])
K = np.diag(K_diag)

# Собственные значения/векторы
w2, V_eig = eigh(K, M)

# Робастные частоты:
# 1) omega (\u22650) — для демпфирования и прочих расчётов
# 2) omega_signed — сохраняет знак (отриц. => линейная неустойчивость)
_tol = 1e-12
w2_finite = np.where(np.isfinite(w2), w2, 0.0)

omega = np.sqrt(np.clip(w2_finite, 0.0, None))  # >= 0
omega_signed = np.empty_like(w2_finite)
pos_mask = w2_finite > _tol
neg_mask = w2_finite < -_tol
zero_mask = ~(pos_mask | neg_mask)
omega_signed[pos_mask] = np.sqrt(w2_finite[pos_mask])
omega_signed[neg_mask] = -np.sqrt(-w2_finite[neg_mask])
omega_signed[zero_mask] = 0.0

# Опорная частота возбуждения: первая положительная, иначе |ω1|
pos_idx = np.where(pos_mask)[0]
if pos_idx.size > 0:
    omega_ref = omega[pos_idx[0]]  # >= 0
else:
    print("[warn] Нет положительных линейных частот; беру |ω1| как опорную.")
    omega_ref = abs(omega_signed[0])

# Проверка бистабильности: для q_eq нужен K11 < 0
if K_diag[0] >= 0:
    raise RuntimeError("Не бистабильно: K11 >= 0 (увеличьте P/Pcr)")

# Демпфирование
if use_modal_zeta:
    zetas = np.full(N_modes, zeta_default, dtype=float)
    for k, z in zeta_custom.items():
        if 1 <= k <= N_modes:
            zetas[k-1] = z
    # Диагональная аппроксимация: C_ii ≈ 2 ζ_i ω_i M_ii (берём ω_i >=0)
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

# Частота возбуждения и шаг
excitation_omega = Omega * max(omega_ref, 1e-12)
if not np.isfinite(excitation_omega) or excitation_omega <= 0.0:
    print("[warn] excitation_omega некорректна; устанавливаю маленькое положительное значение.")
    excitation_omega = 1e-6
T_drive = 2.0*np.pi / excitation_omega
max_step = max_step_frac * T_drive

# Предвычисления
Minv = np.linalg.inv(M)

# =============================================================
# RHS и событие «snap»
# =============================================================
def make_rhs(F0: float, phi_xs: np.ndarray):
    def rhs(t, y):
        q  = y[:N_modes]
        qd = y[N_modes:]
        f = F0 * np.cos(excitation_omega*t)  # точечная сила в x_s
        Fvec = phi_xs * f                    # проекция в модальные координаты
        q2sum = q @ q
        nonlinear = gamma * q2sum * q
        damping = C_diag * qd
        qdd = Minv @ (Fvec - damping - (K @ q) - nonlinear)
        return np.hstack([qd, qdd])
    return rhs


def make_snap_event(theta=theta_threshold):
    def event(t, y):
        q1 = y[0]
        return q1 + theta*q_eq
    event.terminal = True
    event.direction = -1.0
    return event

snap_event = make_snap_event(theta_threshold)

# =============================================================
# Функции для запуска и поиска F0*
# =============================================================
def run_single(F0: float, x_s: float, y0: np.ndarray, t_final: float) -> bool:
    phi_xs = np.array([np.interp(x_s, x, Phi[i]) for i in range(N_modes)])
    rhs = make_rhs(F0, phi_xs)
    sol = solve_ivp(rhs, (0.0, t_final), y0, method="RK45",
                    rtol=rtol, atol=atol, max_step=max_step,
                    events=snap_event)
    return (sol.t_events[0].size > 0)


def find_threshold_F0(x_s: float, F0_hint: float | None = None) -> float:
    t_final = n_periods * T_drive
    y0 = np.zeros(2*N_modes)
    y0[0] = +q_eq

    lo = F0_min_start if F0_hint is None else max(1e-12, 0.5*F0_hint)
    hi = lo
    snapped = run_single(hi, x_s, y0, t_final)
    grow = 1.8
    while not snapped and hi < F0_max_cap:
        lo = hi
        hi *= grow
        snapped = run_single(hi, x_s, y0, t_final)
    if not snapped:
        return np.nan
    while (hi - lo) > bisection_tol * hi:
        mid = 0.5*(lo + hi)
        if run_single(mid, x_s, y0, t_final):
            hi = mid
        else:
            lo = mid
    return hi

# =============================================================
# Основной цикл
# =============================================================
F0_min_list = []
alpha_asym_list = []
F_hint = None

print("Поиск порогов F0 по координате x_s...")
print(f"P/Pcr={P/Pcr:.3f}; Omega={Omega:.3f}; ω_ref={omega_ref:.4g} rad/s; T={T_drive:.4g} s")

for xs in tqdm(x_s_grid, desc="x_s scan"):
    alpha_x = 2.0*abs(xs/L - 0.5)
    alpha_asym_list.append(alpha_x)
    F_thr = find_threshold_F0(xs, F0_hint=F_hint)
    F0_min_list.append(F_thr)
    if not np.isnan(F_thr):
        F_hint = F_thr

F0_min = np.array(F0_min_list)
alpha_asym = np.array(alpha_asym_list)

# Сохранение результатов
np.savez_compressed(f"{save_prefix}_results.npz",
                    x_s_grid=x_s_grid, F0_min=F0_min, alpha_asym=alpha_asym,
                    L=L, P=P, Omega=Omega, n_periods=n_periods, theta=theta_threshold,
                    N_modes=N_modes, zeta_default=zeta_default)
with open(f"{save_prefix}_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["x_s", "x_s/L", "alpha_asym", "F0_min_N"])
    for xs, a, F in zip(x_s_grid, alpha_asym, F0_min):
        w.writerow([f"{xs:.6g}", f"{xs/L:.6g}", f"{a:.6g}", f"{F:.6g}"])

# =============================================================
# Визуализации
# =============================================================
if make_plots:
    # 1) F0_min(x_s)
    plt.figure(figsize=(8.2, 4.6))
    plt.plot(x_s_grid/L, F0_min, marker="o", lw=1.5)
    plt.xlabel(r"координата приложения, $x_s/L$")
    plt.ylabel(r"минимальная амплитуда силы $F_0^*$, Н")
    plt.title(r"Порог перещёлкивания $F_0^*$ vs $x_s$")
    plt.grid(True, ls='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_F_vs_xs.png", dpi=180)
    plt.show()

    # 2) F0_min(α_x)
    plt.figure(figsize=(8.2, 4.6))
    plt.plot(alpha_asym, F0_min, marker="s", lw=1.5)
    plt.xlabel(r"асимметрия $\alpha_x = 2|x_s/L - 1/2|")
    plt.ylabel(r"минимальная амплитуда силы $F_0^*$, Н")
    plt.title(r"Порог перещёлкивания $F_0^*$ vs асимметрия приложения силы")
    plt.grid(True, ls='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_F_vs_asym.png", dpi=180)
    plt.show()

print("Готово.")