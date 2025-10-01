#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupled Cantilever + Bistable Beam (Penalty coupling) — Single Animation
-----------------------------------------------------------------------
Большая консольная балка (Bernoulli–Euler, clamped-free) связана
жёсткой почти нерастяжимой связью с малой бистабильной балкой
(fixed-fixed под осевой силой Pb > Pcr_b). База большой балки возбуждается
поперечным ускорением a_base(t).

Модель:
- Галёркин по формам изгиба обеих балок
- Бистабильность малой балки через кубическую "изотропную" нелинейность
  gamma_b * (Σ q_b^2) * q_b  и отрицат. лин. жёсткость первой моды (за счёт Pb)
- Связь между точками (x_m^*, x_b^*) через штраф Fc = Kc * (w_m - w_b)
- Возбуждение — инерционная нагрузка от ускорения основания большой балки

Визуализация:
- ОДНА анимация: обе балки (сдвинуты вбок для наглядности), связь отрисована
- Без сабплотов. Можно добавить текстовый статус SNAP/NO SNAP

Прогресс:
- Лёгкий прогресс-бар в stdout (без tqdm), обновление максимум ~3–4 Гц.

Зависимости: numpy, scipy, matplotlib
"""

from __future__ import annotations
import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib import animation
import sys, time

# =============================================================
# 0) Пользовательские параметры
# =============================================================

# --- Геометрия/материалы большой консольной балки ---
Lm   = 0.80       # длина, м
Em   = 210e9      # Юнг, Па
rhom = 7800.0     # плотность, кг/м^3
bm   = 0.03       # ширина, м
hm   = 0.003      # толщина, м
Sm   = bm*hm
Im   = bm*hm**3/12

# --- Геометрия/материалы малой бистабильной балки (fixed-fixed) ---
Lb   = 0.30
Eb   = 210e9
rhob = 7800.0
bb   = 0.03
hb   = 0.0014
Sb   = bb*hb
Ib   = bb*hb**3/12

# Осевая сила в малой балке (для бистабильности Pb > Pcr_b)
Pcr_b = 4.0*np.pi**2 * Eb * Ib / (Lb**2)  # критическая Эйлера (fixed-fixed)
Pb    = 1.6 * Pcr_b                       # >1 — бистабильный режим

# --- Число мод ---
Nm = 6     # у консольной балки
Nb = 6     # у бистабильной балки

# --- Позиции крепления нерастяжимой связи ---
x_attach_m = 0.50*Lm   # по длине большой балки (от заделки)
x_attach_b = 0.50*Lb   # по длине малой балки (от нижней заделки)

# --- Штрафная жёсткость связи ---
# Kc = 5e7   # Н/м — достаточно большая, но не чрезмерная (можно варьировать)
Kc = 1e5

# --- Демпфирование ---
# модальное относительное демпфирование для ОБЕИХ балок
zeta_m = 0.01
zeta_b = 0.01

# --- Возбуждение: ускорение основания большой балки ---
a0      = 14.0      # м/с^2 (амплитуда)
Omega_a = 2*np.pi*1.2   # рад/с, частота
# можно задать произвольный закон:
def a_base(t: float) -> float:
    return a0*sin(Omega_a*t)

# --- Бистабильная нелинейность малой балки ---
gamma_b = Eb*Sb/(2.0*Lb)   # «изотропная» кубическая; можно подстроить

# --- Интегрирование ---
T_final     = 6.0       # общее время, с
rtol, atol  = 1e-6, 1e-9
max_step    = 1e-3      # ограничение шага (стабильность с Kc)
# начальная яма для малой балки: +1 (правая) или -1 (левая)
start_well_sign = +1

# --- Пост-снап интеграция ---
n_post_periods = 2.0   # сколько периодов добавить после snap
post_freq_ref  = "main"  # "main" -> первая частота консоли;
                         # "drive" -> частота возбуждения
time_margin    = 0.25   # небольшой запас (в долях периода)


# --- Визуализация ---
Nx_m   = 400     # сетки по длине для отрисовки
Nx_b   = 300
anim_fps    = 30
anim_frames = int(T_final*anim_fps)  # по времени
x_offset_b  = 0.35*Lm                # смещение малой балки вправо в кадре

# --- Прогресс ---
progress_print_hz = 3.0   # не чаще 3 раз/с
progress_ticks    = 100   # печатать по процентам (1%, 2%, ...)

# =============================================================
# 1) Математика форм
# =============================================================

# 1.1) Корни для консольной балки (clamped-free): coshβ cosβ = -1
#     первые корни ~ [1.8751, 4.6941, 7.8548, 10.9955, ...]
def beta_root_cantilever(n: int) -> float:
    """
    Находит n-й корень уравнения cosh(b)*cos(b) + 1 = 0 по переменной b = βL.
    Для n=1 берём [1, 3]; далее интервалы чередуются так, чтобы был смена знака.
    """
    import numpy as np
    from math import pi
    from scipy.optimize import brentq

    def f(b):
        return np.cosh(b)*np.cos(b) + 1.0

    # маленькие сдвиги от концов, чтобы не попадать в точки cos(...) ~ 0
    eps = 1e-8

    if n == 1:
        a, b = 1.0, 3.0
    else:
        k = n - 1
        if k % 2 == 1:  # k нечётный → интервал [kπ, (k+1/2)π]
            a = k*pi + eps
            b = (k + 0.5)*pi - eps
        else:           # k чётный → интервал [(k+1/2)π, (k+1)π]
            a = (k + 0.5)*pi + eps
            b = (k + 1.0)*pi - eps

    fa, fb = f(a), f(b)
    # на крайний случай — немного расширим интервал, если численно знаки совпали
    if fa*fb > 0:
        da = 0.05*pi
        a = max(eps, a - da)
        b = b + da
        fa, fb = f(a), f(b)
        if fa*fb > 0:
            raise RuntimeError(f"Не удалось забракетировать корень для n={n}: f(a)={fa}, f(b)={fb}")

    return brentq(f, a, b)


def build_modes_cantilever(L: float, N: int, x: np.ndarray):
    """
    Стандартные моды консоли:
    φ = cosh(βx) - cos(βx) - α[ sinh(βx) - sin(βx) ], где α = (coshβL+cosβL)/(sinhβL+sinβL)
    Нормировка по энергии изгиба: ∫ (φ')^2 dx = 1
    """
    Nx = x.size
    Phi = np.zeros((N, Nx))
    dPhi = np.zeros_like(Phi)
    ddPhi = np.zeros_like(Phi)

    for i in range(N):
        n = i+1
        beta = beta_root_cantilever(n)/L
        BL = beta*L
        denom = np.sinh(BL) + np.sin(BL)
        alpha = (np.cosh(BL) + np.cos(BL)) / (denom if abs(denom)>1e-12 else np.sign(denom)*1e-12)

        bx = beta*x
        ch, sh = np.cosh(bx), np.sinh(bx)
        c, s   = np.cos(bx),  np.sin(bx)

        phi  = (ch - c) - alpha*(sh - s)
        dphi = beta*(sh + s) - alpha*beta*(ch - c)
        ddphi = beta**2*(ch - c) - alpha*beta**2*(sh + s)

        # нормировка
        norm2 = np.trapz(dphi*dphi, x)
        sN = 1.0/np.sqrt(max(norm2,1e-18))
        Phi[i]   = phi*sN
        dPhi[i]  = dphi*sN
        ddPhi[i] = ddphi*sN
    return Phi, dPhi, ddPhi

# 1.2) Формы fixed-fixed (симм/антисимм), как в предыдущих твоих скриптах
#     Антисимм корни уравнения tan(y)=y
def antisymmetric_root_y(m: int) -> float:
    a = (m - 0.5)*np.pi
    b = (m + 0.5)*np.pi
    f = lambda y: np.tan(y) - y
    return brentq(f, a+1e-6, b-1e-6)

def build_modes_fixedfixed(L: float, N: int, x: np.ndarray):
    """
    Чередуем симметр./антисимм. набор, нормировка ∫(φ')^2 dx = 1.
    Этого достаточно для корректного Галёркина в дальней динамике.
    """
    Nx = x.size
    Phi = np.zeros((N, Nx))
    dPhi = np.zeros_like(Phi)
    ddPhi = np.zeros_like(Phi)

    mode_idx = 0; n_sym = 0; n_anti = 0
    while mode_idx < N:
        i = mode_idx + 1
        if i % 2 == 1:  # симм условная "косинусная"
            n_sym += 1
            n = n_sym
            k = 2.0*np.pi*n/L
            phi   = 1.0 - np.cos(k*x)
            dphi  = k*np.sin(k*x)
            ddphi = k**2*np.cos(k*x)
        else:           # антисимм через tan(y)=y
            n_anti += 1
            y = antisymmetric_root_y(n_anti)
            k = 2.0*y/L
            A = -1.0/np.tan(y)
            C =  k/np.tan(y)
            phi   = A*np.sin(k*x) + np.cos(k*x) + C*x - 1.0
            dphi  = A*k*np.cos(k*x) - k*np.sin(k*x) + C
            ddphi = -A*k**2*np.sin(k*x) - k**2*np.cos(k*x)
        norm2 = np.trapz(dphi*dphi, x)
        sN = 1.0/np.sqrt(max(norm2,1e-18))
        Phi[mode_idx]   = phi*sN
        dPhi[mode_idx]  = dphi*sN
        ddPhi[mode_idx] = ddphi*sN
        mode_idx += 1

    return Phi, dPhi, ddPhi

# =============================================================
# 2) Дискретизация по модам и матрицы
# =============================================================

# Сетки по длинам
x_m = np.linspace(0.0, Lm, Nx_m)
x_b = np.linspace(0.0, Lb, Nx_b)

# Формы
Phi_m, dPhi_m, ddPhi_m = build_modes_cantilever(Lm, Nm, x_m)
Phi_b, dPhi_b, ddPhi_b = build_modes_fixedfixed(Lb, Nb, x_b)

# Массовые и жёсткостные (линейные)
Mm = np.zeros((Nm, Nm))
Km = np.zeros((Nm, Nm))
Mb = np.zeros((Nb, Nb))
Kb = np.zeros((Nb, Nb))

rhoSm = rhom*Sm
rhoSb = rhob*Sb

for i in range(Nm):
    for j in range(i, Nm):
        Mij = rhoSm * np.trapz(Phi_m[i]*Phi_m[j], x_m)
        Kij = Em*Im * np.trapz(ddPhi_m[i]*ddPhi_m[j], x_m)
        Mm[i,j] = Mij; Mm[j,i] = Mij
        Km[i,j] = Kij; Km[j,i] = Kij

for i in range(Nb):
    for j in range(i, Nb):
        Mij = rhoSb * np.trapz(Phi_b[i]*Phi_b[j], x_b)
        # Линейная жёсткость с осевой силой Pb: K = EI∫φ''φ'' dx - Pb ∫φ'φ' dx
        Kij = Eb*Ib * np.trapz(ddPhi_b[i]*ddPhi_b[j], x_b) - Pb * np.trapz(dPhi_b[i]*dPhi_b[j], x_b)
        Mb[i,j] = Mij; Mb[j,i] = Mij
        Kb[i,j] = Kij; Kb[j,i] = Kij

# Проверка бистабильности малой балки: нужна отрицат. диагональ (примерно) по первой моде
Kb_diag = np.diag(Kb)
if Kb_diag[0] >= 0.0:
    print("ВНИМАНИЕ: Pb может быть недостаточен для бистабильности (Kb11>=0). Увеличьте Pb/Pcr_b.")

# Демпфирование (модальное, диагональное в модальном базисе форм Phi)
# Получим частоты из обобщённой лин. задачи; возьмём |ω| для расчёта ξ
w2m, Vm = eigh(Km, Mm); wm = np.sqrt(np.clip(w2m, 0.0, None))
w2b, Vb = eigh(Kb, Mb); wb_abs = np.sqrt(np.abs(w2b))  # могут быть отрицательные

# Опорная частота для пост-интеграции
if post_freq_ref == "drive":
    w_ref = Omega_a
else:
    # возьмём первую положительную собственную частоту консоли
    pos = np.where(wm > 1e-12)[0]
    if pos.size == 0:
        w_ref = Omega_a  # страховка
    else:
        w_ref = wm[pos[0]]

T_ref   = 2*np.pi / max(w_ref, 1e-9)
T_post  = n_post_periods * T_ref * (1.0 + time_margin)


Cm = np.diag(2.0*zeta_m*wm*np.diag(Mm))
Cb = np.diag(2.0*zeta_b*wb_abs*np.diag(Mb))

# Предвычислим вектора форм в точках крепления
phi_m_star = np.array([np.interp(x_attach_m, x_m, Phi_m[i]) for i in range(Nm)])
phi_b_star = np.array([np.interp(x_attach_b, x_b, Phi_b[i]) for i in range(Nb)])

# Векторы усреднения для «центра масс» малой балки (критерий snap)
cm_vec_b = np.trapz(Phi_b, x_b, axis=1) / Lb

# Равновесная амплитуда первой моды малой балки в яме (приближение)
# gamma_b*(q1^2)*q1 + Kb11*q1 ≈ 0 => q_eq ≈ sqrt(-Kb11/gamma_b)
Kb11 = Kb_diag[0]
if Kb11 < 0.0:
    q_eq = np.sqrt(max(-Kb11/gamma_b, 1e-18))
else:
    q_eq = 0.0

Minv_m = np.linalg.inv(Mm)
Minv_b = np.linalg.inv(Mb)

# =============================================================
# 3) Правая часть: объединённая система
# =============================================================

# Размерность: y = [q_m (Nm), q_b (Nb), qd_m (Nm), qd_b (Nb)]
Ndof = Nm + Nb

class ProgressPrinter:
    def __init__(self, t_final: float, hz: float = 3.0, ticks: int = 100):
        self.t_final = max(t_final, 1e-12)
        self.last_wall = 0.0
        self.min_dt_wall = 1.0/max(hz, 1e-6)
        self.next_tick = 0
        self.dt_tick = 1.0/max(ticks,1)
    def maybe_print(self, t: float):
        now = time.time()
        frac = min(max(t/self.t_final, 0.0), 1.0)
        tick = int(frac*100+1e-9)
        # печатаем если прошёл следующий процент И/ИЛИ прошло время
        if tick >= self.next_tick or (now - self.last_wall) >= self.min_dt_wall:
            self.last_wall = now
            self.next_tick = tick + 1
            sys.stdout.write(f"\rИнтегрирование: {tick:3d}%")
            sys.stdout.flush()
            if frac >= 1.0:
                sys.stdout.write("\n")
                sys.stdout.flush()

def make_rhs_and_events(t_final: float):
    prog = ProgressPrinter(t_final, hz=progress_print_hz, ticks=progress_ticks)
    t_snap_holder = [None]    # сюда положим время snap

    def rhs(t, y):
        prog.maybe_print(t)
        q_m   = y[0:Nm]
        q_b   = y[Nm:Nm+Nb]
        qd_m  = y[Nm+Nb:Nm+Nb+Nm]
        qd_b  = y[Nm+Nb+Nm:]

        w_m_star = phi_m_star @ q_m
        w_b_star = phi_b_star @ q_b
        delta = (w_m_star - w_b_star) - delta0
        Fc = Kc * delta

        Fm_c = -Fc * phi_m_star
        Fb_c = +Fc * phi_b_star

        Fm_base = rhoSm * a_base(t) * np.trapz(Phi_m, x_m, axis=1)

        qb2   = q_b @ q_b
        Fb_nl = -gamma_b * qb2 * q_b

        qdd_m = Minv_m @ (Fm_base + Fm_c - Cm @ qd_m - Km @ q_m)
        qdd_b = Minv_b @ (Fb_c        - Cb @ qd_b - Kb @ q_b + Fb_nl)

        return np.hstack([qd_m, qd_b, qdd_m, qdd_b])

    # 1) Событие snap: первый переход центра масс через 0 в нужном направлении
    want_dir = -1.0 if start_well_sign > 0 else +1.0
    def snap_event(t, y):
        q_b = y[Nm:Nm+Nb]
        wcm = cm_vec_b @ q_b
        # если мы близко к корню — запомним момент snap
        if abs(wcm) < 1e-9 and t_snap_holder[0] is None:
            t_snap_holder[0] = t
        return wcm
    snap_event.terminal  = False     # НЕ останавливаемся на snap
    snap_event.direction = want_dir

    # 2) Событие "post": сработает через T_post после snap
    def post_event(t, y):
        if t_snap_holder[0] is None:
            return +1.0   # отключено до snap
        return t - (t_snap_holder[0] + T_post)
    post_event.terminal  = True      # здесь и остановимся
    post_event.direction = +1.0

    return rhs, (snap_event, post_event), t_snap_holder


# =============================================================
# 4) Начальные условия
# =============================================================
y0 = np.zeros(2*Ndof)
# основная балка: в покое у нуля
# малая балка: в одной из ям (приблизим по 1-й моде)
if q_eq > 0:
    y0[Nm + 0] = start_well_sign * q_eq  # q_b[0] = ±q_eq

q_b0 = y0[Nm:Nm+Nb]
q_m0 = y0[0:Nm]
delta0 = float( (phi_m_star @ q_m0) - (phi_b_star @ q_b0) )  # сохранить глобально

# =============================================================
# 5) Интегрирование
# =============================================================
rhs, (snap_event, post_event), t_snap_holder = make_rhs_and_events(T_final)

T_final_cap = T_final + T_post + 0.5*T_ref   # верхняя граница, если snap поздний  # >>> NEW

print(f"P_b/Pcr_b = {Pb/Pcr_b:.3f},  q_eq ≈ {q_eq:.4e} м (по 1-й моде),  Kc = {Kc:.3e} Н/м")
print(f"Nm={Nm}, Nb={Nb},  T_final={T_final}s  (+{n_post_periods}T_ref post),  a0={a0} m/s^2,  Ωa={Omega_a/(2*np.pi):.2f} Hz")

sol = solve_ivp(rhs, (0.0, T_final_cap), y0, method="RK45",
                rtol=rtol, atol=atol, max_step=max_step,
                events=(snap_event, post_event))  # <<< ДВА события


t = sol.t
Y = sol.y
q_m_t  = Y[0:Nm, :]
q_b_t  = Y[Nm:Nm+Nb, :]
qd_m_t = Y[Nm+Nb:Nm+Nb+Nm, :]
qd_b_t = Y[Nm+Nb+Nm:, :]

snapped = (sol.t_events[0].size > 0)
t_snap = (sol.t_events[0][0] if snapped else None)
stopped_by_post = (len(sol.t_events) > 1 and sol.t_events[1].size > 0)  # >>> NEW

print("\nSNAP status:", "SNAP at t = %.5f s" % t_snap if snapped else "NO SNAP")
if stopped_by_post:
    print(f"Stopped after ~{n_post_periods} periods post-snap at t = {sol.t[-1]:.5f} s")


# =============================================================
# 6) Подготовка данных для анимации
# =============================================================
# Чтобы анимировать «одной лентой времени», возьмём равномерные кадры по времени
t_frames = np.linspace(t[0], t[-1], anim_frames)
# интерполяция решений по времени
from numpy import interp
def interp_states(T, arr):
    # arr: (n, Nt) — интерполируем вдоль времени
    return np.vstack([np.interp(t_frames, T, arr[i]) for i in range(arr.shape[0])])

q_m_f = interp_states(t, q_m_t)
q_b_f = interp_states(t, q_b_t)

# --- интерполяция скоростей мод консоли (для кинетической энергии) ---
qd_m_f = interp_states(t, qd_m_t)   # shape: (Nm, frames)

# --- строгие модальные энергии через собственный базис (K_m v = w^2 M_m v) ---
# У нас уже есть из выше: w2m, Vm = eigh(Km, Mm)
# SciPy нормирует так, что Vm.T @ Mm @ Vm = I, Vm.T @ Km @ Vm = diag(w2m)

n_show = min(5, Nm)
V5   = Vm[:, :n_show]                     # первые 5 собственных векторов
w2_5 = w2m[:n_show]                       # их квадрат частот (>=0 для консоли)
w_5  = np.sqrt(np.maximum(w2_5, 0.0))     # частоты

# Проекция текущих состояний в модальные координаты:
# q = Vm * η  ⇒  η = Vm^T M q;  аналогично для скоростей
Mq_frames   = Mm @ q_m_f                  # (Nm, frames)
Mqd_frames  = Mm @ qd_m_f                 # (Nm, frames)

eta_f    = V5.T @ Mq_frames               # (n_show, frames)
etad_f   = V5.T @ Mqd_frames              # (n_show, frames)

# Модальные энергии (строго линейные, "независимые"):
# E_j(t) = 1/2 * (etad_j)^2 + 1/2 * (w_j^2) * (eta_j)^2
E_modes = 0.5*(etad_f**2) + 0.5*(w2_5[:, None])*(eta_f**2)
E_modes = np.clip(E_modes, 0.0, None)     # на всякий случай отсекаем машинный шум < 0

# Пределы по оси Y для второго графика
Emin = 0.0
Emax = np.percentile(E_modes, 99.5)
if not np.isfinite(Emax) or Emax <= 0.0:
    Emax = 1.0


# Пространственные профили для каждого кадра
profiles_m = q_m_f.T @ Phi_m   # (frames, Nx_m)
profiles_b = q_b_f.T @ Phi_b   # (frames, Nx_b)

# Координаты точек крепления для связи в каждом кадре
wm_star_frames = (phi_m_star @ q_m_f).ravel()
wb_star_frames = (phi_b_star @ q_b_f).ravel()

# =============================================================
# 7) Анимация: обе балки + связь
# =============================================================
# ДВА сабплота: (1) профили балок, (2) энергии мод консоли
fig, (ax, axE) = plt.subplots(2, 1, figsize=(9.5, 8.0), gridspec_kw={'height_ratios':[3, 2]}, sharex=False)

# ==== верхняя ось (как было) ====
ax.plot(x_m, 0*x_m, lw=0.8, alpha=0.4, color='k')
ax.plot(x_offset_b + x_b, 0*x_b, lw=0.8, alpha=0.4, color='k')
line_m, = ax.plot([], [], lw=2.2, label="Основная консольная")
line_b, = ax.plot([], [], lw=2.2, label="Бистабильная (fixed-fixed)")
link_line, = ax.plot([], [], lw=2.6, alpha=0.9, label="Жёсткая связь")
pt_m, = ax.plot([], [], 'o', ms=6)
pt_b, = ax.plot([], [], 'o', ms=6)

y_all = np.hstack([profiles_m.ravel(), profiles_b.ravel()])
ymin, ymax = np.percentile(y_all, [1, 99])
y_span = ymax - ymin
if y_span < 1e-9:
    ymin -= 1e-6; ymax += 1e-6
else:
    ymin -= 0.15*y_span; ymax += 0.15*y_span
ax.set_xlim(-0.05*Lm, x_offset_b + Lb + 0.05*Lm)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Горизонтальная координата (вторая балка сдвинута), м")
ax.set_ylabel("Поперечный прогиб, м")
ax.grid(True, ls='--', alpha=0.3)
ax.legend(loc="upper right")
title_text = ax.set_title("")

status_prefix = f"SNAP: t={t_snap:.4f} с" if snapped else "NO SNAP"
xm_plot = x_m
xb_plot = x_offset_b + x_b
xm_star_plot = x_attach_m
xb_star_plot = x_offset_b + x_attach_b

# ==== нижняя ось (НОВАЯ) — энергии мод ====
lines_E = []
for i in range(E_modes.shape[0]):
    (li,) = axE.plot(t_frames, E_modes[i], lw=1.8, label=fr"$E_{{m{i+1}}}(t)$")
    lines_E.append(li)

# вертикальный курсор по времени (будем двигать в update)
cursor_time = axE.axvline(t_frames[0], ls='--', lw=2.0, alpha=0.8)

axE.set_xlim(t_frames[0], t_frames[-1])
axE.set_ylim(Emin, 1.1*Emax)
axE.set_xlabel("Время, с")
axE.set_ylabel("Энергия мод консоли, Дж")
axE.grid(True, ls='--', alpha=0.3)
axE.legend(ncol=min(5, E_modes.shape[0]), loc="upper right")

snap_line = None
if snapped and (t_snap is not None):
    snap_line = axE.axvline(t_snap, ls='--', lw=2.2, alpha=0.9, color='red')
    snap_line.set_label("Snap")
    axE.legend(ncol=min(5, E_modes.shape[0]), loc="upper right")



def init():
    line_m.set_data([], [])
    line_b.set_data([], [])
    link_line.set_data([], [])
    pt_m.set_data([], [])
    pt_b.set_data([], [])
    title_text.set_text("")
    cursor_time.set_xdata([t_frames[0], t_frames[0]])
    return line_m, line_b, link_line, pt_m, pt_b, title_text, cursor_time


def update(k):
    wm = profiles_m[k]
    wb = profiles_b[k]
    line_m.set_data(xm_plot, wm)
    line_b.set_data(xb_plot, wb)

    y_m_star = wm_star_frames[k]
    y_b_star = wb_star_frames[k]
    pt_m.set_data([xm_star_plot], [y_m_star])
    pt_b.set_data([xb_star_plot], [y_b_star])
    link_line.set_data([xm_star_plot, xb_star_plot], [y_m_star, y_b_star])

    t_now = t_frames[k]
    title_text.set_text(f"{status_prefix}   |   t = {t_now:.3f} с")

    # --- движем курсор на графике энергий ---
    cursor_time.set_xdata([t_now, t_now])

    return line_m, line_b, link_line, pt_m, pt_b, title_text, cursor_time


ani = animation.FuncAnimation(
    fig, update, frames=anim_frames, init_func=init,
    blit=True, interval=1000/anim_fps, repeat=True
)

plt.tight_layout()
plt.show()
