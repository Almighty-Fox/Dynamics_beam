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
bb   = 0.02
hb   = 0.001
Sb   = bb*hb
Ib   = bb*hb**3/12

# Осевая сила в малой балке (для бистабильности Pb > Pcr_b)
Pcr_b = 4.0*np.pi**2 * Eb * Ib / (Lb**2)  # критическая Эйлера (fixed-fixed)
Pb    = 1.6 * Pcr_b                       # >1 — бистабильный режим

# --- Число мод ---
Nm = 6     # у консольной балки
Nb = 6     # у бистабильной балки

# --- Позиции крепления нерастяжимой связи ---
x_attach_m = 0.55*Lm   # по длине большой балки (от заделки)
x_attach_b = 0.77*Lb   # по длине малой балки (от нижней заделки)

# --- Штрафная жёсткость связи ---
# Kc = 5e7   # Н/м — достаточно большая, но не чрезмерная (можно варьировать)
Kc = 1e2

# --- Демпфирование ---
# модальное относительное демпфирование для ОБЕИХ балок
zeta_m = 0.01
zeta_b = 0.01

# --- Возбуждение: ускорение основания большой балки ---
a0      = 8.0      # м/с^2 (амплитуда)
Omega_a = 2*np.pi*12.0   # рад/с, частота
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

def make_rhs_and_event(t_final: float):
    prog = ProgressPrinter(t_final, hz=progress_print_hz, ticks=progress_ticks)

    def rhs(t, y):
        # прогресс почти без накладных расходов
        prog.maybe_print(t)

        q_m   = y[0:Nm]
        q_b   = y[Nm:Nm+Nb]
        qd_m  = y[Nm+Nb:Nm+Nb+Nm]
        qd_b  = y[Nm+Nb+Nm:]

        # Связь: разность поперечных перемещений в точках
        w_m_star = phi_m_star @ q_m
        w_b_star = phi_b_star @ q_b
        delta = w_m_star - w_b_star  # хотим = 0
        Fc = Kc * delta              # сила связи (Н)

        # Обобщённые силы от связи
        Fm_c = -Fc * phi_m_star     # знак минус, т.к. Fc действует на "узел" в сторону уменьшения delta
        Fb_c = +Fc * phi_b_star

        # Инерционная нагрузка от ускорения основания большой балки:
        # q̈_m = M^-1 ( F_base - C q̇ - K q - NL - ... )
        # F_base_i = ∫ (ρS * a_base(t) * φ_i(x)) dx  = (ρS * a)* ∫ φ_i dx
        Fm_base = rhoSm * a_base(t) * np.trapz(Phi_m, x_m, axis=1)

        # Нелинейность только у малой балки (бистабильность)
        qb2 = q_b @ q_b
        Fb_nl = -gamma_b * qb2 * q_b   # переносим в правую часть со знаком "-"

        # Собираем ускорения
        qdd_m = Minv_m @ (Fm_base + Fm_c - Cm @ qd_m - Km @ q_m)
        qdd_b = Minv_b @ (Fb_c        - Cb @ qd_b - Kb @ q_b + Fb_nl)

        return np.hstack([qd_m, qd_b, qdd_m, qdd_b])

    # Событие snap-through: средний прогиб малой балки меняет знак в нужном направлении
    want_dir = -1.0 if start_well_sign > 0 else +1.0
    def snap_event(t, y):
        q_b = y[Nm:Nm+Nb]
        wcm = cm_vec_b @ q_b
        return wcm
    snap_event.terminal  = True
    snap_event.direction = want_dir

    return rhs, snap_event

# =============================================================
# 4) Начальные условия
# =============================================================
y0 = np.zeros(2*Ndof)
# основная балка: в покое у нуля
# малая балка: в одной из ям (приблизим по 1-й моде)
if q_eq > 0:
    y0[Nm + 0] = start_well_sign * q_eq  # q_b[0] = ±q_eq

# =============================================================
# 5) Интегрирование
# =============================================================
rhs, snap_event = make_rhs_and_event(T_final)
print(f"P_b/Pcr_b = {Pb/Pcr_b:.3f},  q_eq ≈ {q_eq:.4e} м (по 1-й моде),  Kc = {Kc:.3e} Н/м")
print(f"Nm={Nm}, Nb={Nb},  T_final={T_final}s,  a0={a0} m/s^2,  Ωa={Omega_a/(2*np.pi):.2f} Hz")
sol = solve_ivp(rhs, (0.0, T_final), y0, method="RK45",
                rtol=rtol, atol=atol, max_step=max_step,
                events=snap_event)

t = sol.t
Y = sol.y
q_m_t  = Y[0:Nm, :]
q_b_t  = Y[Nm:Nm+Nb, :]
qd_m_t = Y[Nm+Nb:Nm+Nb+Nm, :]
qd_b_t = Y[Nm+Nb+Nm:, :]

snapped = (sol.t_events[0].size > 0)
t_snap = sol.t_events[0][0] if snapped else None
print("SNAP status:", "SNAP at t = %.5f s" % t_snap if snapped else "NO SNAP")

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

# Пространственные профили для каждого кадра
profiles_m = q_m_f.T @ Phi_m   # (frames, Nx_m)
profiles_b = q_b_f.T @ Phi_b   # (frames, Nx_b)

# Координаты точек крепления для связи в каждом кадре
wm_star_frames = (phi_m_star @ q_m_f).ravel()
wb_star_frames = (phi_b_star @ q_b_f).ravel()

# =============================================================
# 7) Анимация: обе балки + связь
# =============================================================
fig, ax = plt.subplots(figsize=(9.5, 6.0))

# Рисуем "геом. оси" (нулевые линии) для наглядности
ax.plot(x_m, 0*x_m, lw=0.8, alpha=0.4, color='k')
ax.plot(x_offset_b + x_b, 0*x_b, lw=0.8, alpha=0.4, color='k')

# Линии профилей
line_m, = ax.plot([], [], lw=2.2, label="Основная консольная")
line_b, = ax.plot([], [], lw=2.2, label="Бистабильная (fixed-fixed)")

# Линия связи (между точками крепления, визуально — прямой отрезок)
link_line, = ax.plot([], [], lw=2.6, alpha=0.9, label="Жёсткая связь")

# Метки точек крепления
pt_m, = ax.plot([], [], 'o', ms=6)
pt_b, = ax.plot([], [], 'o', ms=6)

# Пределы осей (по наблюдаемым амплитудам)
y_all = np.hstack([profiles_m.ravel(), profiles_b.ravel()])
ymin, ymax = np.percentile(y_all, [1, 99])
y_span = ymax - ymin
if y_span < 1e-9:
    ymin -= 1e-6; ymax += 1e-6
else:
    ymin -= 0.15*y_span; ymax += 0.15*y_span

ax.set_xlim(-0.05*Lm, x_offset_b + Lb + 0.05*Lm)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Горизонтальная координата (искусственный сдвиг для второй балки), м")
ax.set_ylabel("Поперечный прогиб, м")
ax.grid(True, ls='--', alpha=0.3)
ax.legend(loc="upper right")

title_text = ax.set_title("")

# Время SNAP — вертикальная линия как текст в заголовке
status_prefix = f"SNAP: t={t_snap:.4f} с" if snapped else "NO SNAP"

# Предвычислим «жёсткое» — абсциссы
xm_plot = x_m
xb_plot = x_offset_b + x_b
xm_star_plot = x_attach_m
xb_star_plot = x_offset_b + x_attach_b

def init():
    line_m.set_data([], [])
    line_b.set_data([], [])
    link_line.set_data([], [])
    pt_m.set_data([], [])
    pt_b.set_data([], [])
    title_text.set_text("")
    return line_m, line_b, link_line, pt_m, pt_b, title_text

def update(k):
    wm = profiles_m[k]                  # (Nx_m,)
    wb = profiles_b[k]                  # (Nx_b,)
    line_m.set_data(xm_plot, wm)
    line_b.set_data(xb_plot, wb)

    # точки крепления
    y_m_star = wm_star_frames[k]
    y_b_star = wb_star_frames[k]
    pt_m.set_data([xm_star_plot], [y_m_star])
    pt_b.set_data([xb_star_plot], [y_b_star])

    # связь — прямой отрезок
    link_line.set_data([xm_star_plot, xb_star_plot], [y_m_star, y_b_star])

    title_text.set_text(f"{status_prefix}   |   t = {t_frames[k]:.3f} с")
    return line_m, line_b, link_line, pt_m, pt_b, title_text

ani = animation.FuncAnimation(
    fig, update, frames=anim_frames, init_func=init,
    blit=True, interval=1000/anim_fps, repeat=False
)

plt.tight_layout()
plt.show()
