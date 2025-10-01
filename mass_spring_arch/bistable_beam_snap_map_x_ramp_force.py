from __future__ import annotations
import numpy as np
from numpy import sin, cos
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# =============================================================
# Пользовательские параметры
# =============================================================
L = 0.30                 # длина балки [м]
E = 210e9                # Юнг [Па]
rho = 7800.0             # плотность [кг/м^3]
b = 0.02                 # ширина [м]
h = 0.002                # толщина [м]
S = b * h                # площадь [м^2]
I = b * h**3 / 12.0      # момент инерции [м^4]
drive_mode_index = 1     # индекс положительной линейной моды для справочного периода

# Критическая нагрузка Эйлера (фикс-фикс)
Pcr = 4.0*np.pi**2 * E * I / L**2
P = 6.30 * Pcr           # > 1 для бистабильности

# Число мод
N_modes = 12

# Демпфирование: модальные ζ_i
use_modal_zeta = True
zeta_default = 0.04
zeta_custom = {1: 0.01, 2: 0.01}
# Альтернатива — Релея (если use_modal_zeta=False)
rayleigh_a0 = 0.0
rayleigh_a1 = 0.0

# Интегрирование
rtol, atol = 1e-6, 1e-9
Nx = 2001                # сетка по x для форм
start_in_well = +1       # +1 → старт в правой яме (q ~ +q_eq), -1 → в левой

# Скан по x_s (только половина балки)
N_xs = 101
x_min = 0.2*L           # избегаем x_s=0 ровно (узел)
x_max = 0.5*L            # половина балки
xs_list = np.linspace(x_min, x_max, N_xs)

# Квазистатический протокол накачки
qs_eps = 0.03            # допустимая доля инерционности в recommend_quasistatic_alpha
alpha_factor = 0.50      # доля от alpha_max (меньше — «квазистатичнее»)
tmax_periods_ramp = 80   # макс. число справочных периодов для этапа ramp до события
twin_periods_after = 5   # «догоняем» после щелчка при удержании F*=const

# =============================================================
# Формы для балки с заделками на концах (clamped–clamped)
# =============================================================
def _g_cc(z: float) -> float:
    return np.cos(z) - 1.0 / np.cosh(z)

def cc_roots(n_roots: int, n_hard_cap: int = 200):
    roots = []
    n = 1
    eps = 1e-9
    while len(roots) < n_roots and n <= n_hard_cap:
        a = n * np.pi + eps
        b = (n + 1) * np.pi - eps
        ga = _g_cc(a)
        gb = _g_cc(b)
        if np.sign(ga) == np.sign(gb):
            M = 64
            xs = np.linspace(a, b, M + 1)
            gs = np.array([_g_cc(xx) for xx in xs])
            brack = False
            for i in range(M):
                if np.sign(gs[i]) == 0:
                    roots.append(xs[i]); brack = True; break
                if np.sign(gs[i]) != np.sign(gs[i+1]):
                    r = brentq(_g_cc, xs[i], xs[i+1])
                    roots.append(r); brack = True; break
            if not brack:
                n += 1
                continue
        else:
            r = brentq(_g_cc, a, b)
            roots.append(r)
        n += 1
    if len(roots) < n_roots:
        raise RuntimeError(f"Найдено {len(roots)} корней из {n_roots}. Увеличьте n_hard_cap.")
    return np.array(roots)

def build_modes_cc(L: float, N: int, x: np.ndarray):
    z = cc_roots(N)
    beta = z / L
    Nx = x.size
    Phi  = np.zeros((N, Nx))
    dPhi = np.zeros_like(Phi)
    ddPhi= np.zeros_like(Phi)
    for k in range(N):
        b = beta[k]
        sig = (np.cosh(b * L) - np.cos(b * L)) / (np.sinh(b * L) - np.sin(b * L))
        phi   =  np.cosh(b*x) - np.cos(b*x) - sig*(np.sinh(b*x) - np.sin(b*x))
        dphi  =  b*(np.sinh(b*x) + np.sin(b*x)) - sig*b*(np.cosh(b*x) - np.cos(b*x))
        ddphi =  b**2*(np.cosh(b*x) + np.cos(b*x)) - sig*b**2*(np.sinh(b*x) + np.sin(b*x))
        norm2 = np.trapz(dphi*dphi, x)
        s = 1.0/np.sqrt(norm2 if norm2 > 0 else 1.0)
        Phi[k]   = phi*s
        dPhi[k]  = dphi*s
        ddPhi[k] = ddphi*s
    return Phi, dPhi, ddPhi

# =============================================================
# Сборка модели
# =============================================================
x = np.linspace(0.0, L, Nx)
Phi, dPhi, ddPhi = build_modes_cc(L, N_modes, x)

# Центр масс: (1/L)∫ Phi dx
cm_vec = np.trapz(Phi, x, axis=1) / L

rhoS = rho * S
M = np.zeros((N_modes, N_modes))
K = np.zeros((N_modes, N_modes))
for i in range(N_modes):
    for j in range(N_modes):
        M[i, j] = rhoS * np.trapz(Phi[i]   * Phi[j],   x)
        K[i, j] =  E*I * np.trapz(ddPhi[i] * ddPhi[j], x) \
                  - P   * np.trapz(dPhi[i]  * dPhi[j],  x)

# Собственные значения/векторы
w2, V_eig = eigh(K, M)
_tol = 1e-12
w2_finite = np.where(np.isfinite(w2), w2, 0.0)

omega = np.sqrt(np.clip(w2_finite, 0.0, None))
omega_signed = np.empty_like(w2_finite)
pos_mask = w2_finite > _tol
neg_mask = w2_finite < -_tol
zero_mask = ~(pos_mask | neg_mask)
omega_signed[pos_mask] = np.sqrt(w2_finite[pos_mask])
omega_signed[neg_mask] = -np.sqrt(-w2_finite[neg_mask])
omega_signed[zero_mask] = 0.0

if w2[0] >= 0:
    raise RuntimeError("Не бистабильно около w=0: минимальное ω^2 >= 0 (увеличьте P/Pcr)")

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

# Мягкое направление и q_eq
lamK, U_K = np.linalg.eigh(K)
lamK_min  = lamK[0]
u_min     = U_K[:, 0] / np.linalg.norm(U_K[:, 0])
if lamK_min >= 0:
    raise RuntimeError("Не бистабильно около w=0: минимальное λ_K >= 0 (увеличьте P/Pcr)")
q_eq_amp = np.sqrt(max(1e-18, -lamK_min / gamma))
q_eq_vec = q_eq_amp * u_min

Minv = np.linalg.inv(M)

# =============================================================
# Вспомогательные (точно как в твоём коде, с доп. аргументами)
# =============================================================
def make_rhs(F_of_t, phi_xs: np.ndarray):
    def rhs(t, y):
        q  = y[:N_modes]
        qd = y[N_modes:]
        Fvec = phi_xs * F_of_t(t) * (-np.sign(start_in_well))
        q2sum = q @ q
        nonlinear = gamma * q2sum * q
        damping = C_diag * qd
        qdd = Minv @ (Fvec - damping - (K @ q) - nonlinear)
        return np.hstack([qd, qdd])
    return rhs

def make_ramp(alpha: float):
    return lambda t: alpha * t

def make_hold(F_star: float):
    return lambda t: F_star

def make_snap_event_cm(start_in_well_sign: int):
    want_dir = -1.0 if start_in_well_sign > 0 else +1.0
    def event(t, y):
        q = y[:N_modes]
        wcm = cm_vec @ q
        return wcm
    event.terminal  = True
    event.direction = want_dir
    return event

def choose_drive_omega():
    pos_idx = np.flatnonzero(omega > 0.0)
    if pos_idx.size > 0:
        k = max(0, min(drive_mode_index-1, pos_idx.size-1))
        return omega[pos_idx[k]]
    return abs(omega_signed[0])

def recommend_quasistatic_alpha(eps=0.03, phi_xs=None):
    if lamK_min >= 0:
        raise RuntimeError("Нет отрицательной λ_K: увеличьте P/Pcr")
    m_eff = float(u_min @ (M @ u_min))
    K_eff = -2.0 * lamK_min
    qeq   = np.sqrt(-lamK_min / gamma)
    omega_well = np.sqrt(K_eff / m_eff)
    pos_idx = np.flatnonzero(omega > 0.0)
    omega_pos1 = omega[pos_idx[0]] if pos_idx.size > 0 else abs(omega_signed[0])
    omega_star = min(omega_well, omega_pos1)
    proj = float(u_min @ phi_xs) if phi_xs is not None else 1.0
    proj_abs = max(1e-6, abs(proj))
    F_char_modal = K_eff * qeq
    F_char_ext   = F_char_modal / proj_abs
    alpha_max = eps * F_char_ext * omega_star
    return dict(alpha_max=alpha_max,
                K_eff=K_eff, m_eff=m_eff,
                omega_well=omega_well, omega_pos1=omega_pos1, omega_star=omega_star,
                F_char_ext=F_char_ext, proj=proj, q_eq=qeq)

# =============================================================
# Основная функция: F_snap(x_s) по протоколу ramp→hold
# =============================================================
def threshold_force_for_xs(x_s: float,
                           eps_qs=qs_eps,
                           alpha_mul=alpha_factor,
                           tmax_periods=tmax_periods_ramp,
                           twin_periods=twin_periods_after):
    # справочный период по первой положительной моде (для шага и окон)
    omega_ref = choose_drive_omega()
    T_ref = 2.0*np.pi / max(omega_ref, 1e-12)
    max_step = (1/200.0) * T_ref

    # форма в точке приложения
    phi_xs = np.array([np.interp(x_s, x, Phi[i]) for i in range(N_modes)])

    # рекомендация квазистатической скорости с учётом проекции
    rec = recommend_quasistatic_alpha(eps=eps_qs, phi_xs=phi_xs)
    alpha = alpha_mul * rec["alpha_max"]

    # начальные условия в «правой» яме
    y0 = np.zeros(2 * N_modes)
    y0[:N_modes] = np.sign(start_in_well) * q_eq_vec

    # этап 1: F=α t до события снапа
    rhs_ramp   = make_rhs(make_ramp(alpha), phi_xs)
    snap_event = make_snap_event_cm(start_in_well)

    t_final_1  = tmax_periods * T_ref
    sol1 = solve_ivp(rhs_ramp, (0.0, t_final_1), y0, method="RK45",
                     rtol=rtol, atol=atol, max_step=max_step, events=snap_event)

    snapped = (sol1.t_events[0].size > 0)
    if not snapped:
        return np.nan, dict(alpha=alpha, proj=rec["proj"], note="no snap within ramp window")

    t_snap = float(sol1.t_events[0][0])
    F_snap = alpha * t_snap

    # этап 2 (не влияет на F_snap): удерживаем F*=const и «догоняем»
    y_snap   = sol1.y[:, -1]
    rhs_hold = make_rhs(make_hold(F_snap), phi_xs)
    t_final_2 = t_snap + twin_periods * T_ref
    _ = solve_ivp(rhs_hold, (t_snap, t_final_2), y_snap, method="RK45",
                  rtol=rtol, atol=atol, max_step=max_step)

    return F_snap, dict(alpha=alpha, proj=rec["proj"])

# =============================================================
# Скан по x_s и построение графика
# =============================================================
Fsnaps = np.empty_like(xs_list)
meta = []

print(f"P/Pcr={P/Pcr:.3f}; N_modes={N_modes}; zeta_default={zeta_default}")
print("----- sweep x_s on [0, L/2] -----")
for i, xs in enumerate(xs_list):
    Fstar, info = threshold_force_for_xs(xs)
    Fsnaps[i] = Fstar
    meta.append(info)
    status = "OK" if np.isfinite(Fstar) else "NO SNAP"
    print(f"x_s/L={xs/L:.4f}  ->  F_snap = {Fstar:.6g} N   [{status}; alpha={info['alpha']:.3e}; proj={info['proj']:.3e}]")

# Нормируем x_s по L для оси
xi = xs_list / L

plt.figure(figsize=(9.6, 5.4))
plt.plot(xi, Fsnaps, lw=2, marker='o', ms=4)
plt.xlabel(r"$x_s/L$")
plt.ylabel(r"$F_{\mathrm{snap}}$  [N]")
plt.title("Минимальная сила перещёлкивания vs точка приложения силы (левая половина балки)")
plt.grid(True, ls='--', alpha=0.35)
plt.tight_layout()
plt.show()
