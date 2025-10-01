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
N_modes = 12

# Демпфирование: модальные ζ_i
use_modal_zeta = True
zeta_default = 0.04
zeta_custom = {1: 0.01, 2: 0.01}
# Альтернатива — Релея (если use_modal_zeta=False)
rayleigh_a0 = 0.0
rayleigh_a1 = 0.0

# Тесты: список (x_s [м], F0 [Н], Omega)
TESTS = [0.08, 0.11, 0.15]

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
# =============================================================
# Правильные линейные формы для балки с заделками (clamped–clamped)
# w(0)=w(L)=0, w'(0)=w'(L)=0; уравнение частот: cosh(z)cos(z)=1, z=βL
# =============================================================
# =============================================================
# Clamped–clamped формы: cosh(z)cos(z) = 1, z = βL
# Защита от переполнений в cosh: ограничиваем поиск разумным n.
# Этого более чем достаточно для N_modes ≤ 30.
# =============================================================
# =============================================================
# Clamped–clamped: cosh(z)cos(z) = 1  ⇔  cos(z) = sech(z) = 1/cosh(z)
# Используем g(z) = cos(z) - 1/cosh(z), чтобы избежать переполнений.
# Корни: по одному на интервале [nπ, (n+1)π], n = 1,2,...
# =============================================================
def _g_cc(z: float) -> float:
    # безопасная целевая функция: ограничена [-2, 2]
    return np.cos(z) - 1.0 / np.cosh(z)

def cc_roots(n_roots: int, n_hard_cap: int = 200):
    """
    Возвращает первые n_roots корней z_k > 0 уравнения cosh(z)cos(z)=1,
    избегая переполнений. Ищем по одному корню на [nπ, (n+1)π], n=1,2,...
    """
    roots = []
    n = 1
    eps = 1e-9
    while len(roots) < n_roots and n <= n_hard_cap:
        a = n * np.pi + eps
        b = (n + 1) * np.pi - eps
        ga = _g_cc(a)
        gb = _g_cc(b)

        if np.sign(ga) == np.sign(gb):
            # редкий случай — попробуем адаптивно раздробить
            M = 64
            xs = np.linspace(a, b, M + 1)
            gs = np.array([_g_cc(xx) for xx in xs])
            brack_found = False
            for i in range(M):
                if np.sign(gs[i]) == 0:
                    roots.append(xs[i])
                    brack_found = True
                    break
                if np.sign(gs[i]) != np.sign(gs[i+1]):
                    r = brentq(_g_cc, xs[i], xs[i+1])
                    roots.append(r)
                    brack_found = True
                    break
            # если не нашли — идём к следующему интервалу (не должно случаться)
            if not brack_found:
                n += 1
                continue
        else:
            # обычная ситуация: знаки разные на концах
            r = brentq(_g_cc, a, b)
            roots.append(r)

        n += 1

    if len(roots) < n_roots:
        raise RuntimeError(f"Найдено {len(roots)} корней из {n_roots}. "
                           f"Увеличьте n_hard_cap в cc_roots().")
    return np.array(roots)

def build_modes_cc(L: float, N: int, x: np.ndarray):
    """
    Возвращает Phi, dPhi, ddPhi для clamped–clamped балки.
    Нормировка: ∫_0^L (φ')^2 dx = 1.
    """
    z = cc_roots(N)           # z_k = β_k L, безопасно
    beta = z / L
    Nx = x.size
    Phi  = np.zeros((N, Nx))
    dPhi = np.zeros_like(Phi)
    ddPhi= np.zeros_like(Phi)

    for k in range(N):
        b = beta[k]
        # σ из граничных условий (везде безопасно)
        # Вариант 1 (из w(L)=0):
        sig = (np.cosh(b * L) - np.cos(b * L)) / (np.sinh(b * L) - np.sin(b * L))

        # Вариант 2 (из w'(L)=0):
        # sig = (np.sinh(b*L) + np.sin(b*L)) / (np.cosh(b*L) - np.cos(b*L))

        # φ, φ', φ''
        phi   =  np.cosh(b*x) - np.cos(b*x) - sig*(np.sinh(b*x) - np.sin(b*x))
        dphi  =  b*(np.sinh(b*x) + np.sin(b*x)) - sig*b*(np.cosh(b*x) - np.cos(b*x))
        ddphi =  b**2*(np.cosh(b*x) + np.cos(b*x)) - sig*b**2*(np.sinh(b*x) + np.sin(b*x))

        # нормировка ∫(φ')^2 dx = 1 (энергетически удобная)
        norm2 = np.trapz(dphi*dphi, x)
        s = 1.0/np.sqrt(norm2 if norm2 > 0 else 1.0)

        Phi[k]   = phi*s
        dPhi[k]  = dphi*s
        ddPhi[k] = ddphi*s

    return Phi, dPhi, ddPhi




# =============================================================
# Сборка модели (общая часть)
# =============================================================
x = np.linspace(0.0, L, Nx)
Phi, dPhi, ddPhi = build_modes_cc(L, N_modes, x)

# Вектор усреднения по длине для центра масс (однородная масса → просто среднее)
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

# Проверка бистабильности около w=0: первая мода должна быть неустойчива (ω1^2 < 0)
# Для этого используем отсортированный спектр w2 (eigh уже сортирует возрастанием)
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

# Эффективная "линейная" жёсткость первой моды в базисе φ (элемент K[0,0])
lamK, U_K = np.linalg.eigh(K)      # возрастание
lamK_min  = lamK[0]
u_min     = U_K[:, 0]
u_min     = u_min / np.linalg.norm(u_min)   # ||u_min||_2 = 1

if lamK_min >= 0:
    raise RuntimeError("Не бистабильно около w=0: минимальное λ_K >= 0 (увеличьте P/Pcr)")

q_eq_amp = np.sqrt(max(1e-18, -lamK_min / gamma))  # скаляр
q_eq_vec = q_eq_amp * u_min                        # вектор в Phi-базисе


Minv = np.linalg.inv(M)

# =============================================================
# Общие вспомогательные
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


def recommend_quasistatic_alpha(eps=0.03, phi_xs=None):
    """
    Квазистатическая скорость нарастания α (внешняя сила F_ext = α t).
    eps — допустимая доля инерционности (например, 0.03 = 3%).
    """
    if lamK_min >= 0:
        raise RuntimeError("Нет отрицательной λ_K: увеличьте P/Pcr")

    # Масштабы вдоль мягкого направления u_min
    m_eff = float(u_min @ (M @ u_min))           # кг
    K_eff = -2.0 * lamK_min                      # Н/м
    qeq   = np.sqrt(-lamK_min / gamma)           # м

    omega_well = np.sqrt(K_eff / m_eff)          # 1/с

    # первая положительная линейная частота (из обобщённой задачи)
    pos_idx = np.flatnonzero(omega > 0.0)
    omega_pos1 = omega[pos_idx[0]] if pos_idx.size > 0 else abs(omega_signed[0])

    omega_star = min(omega_well, omega_pos1)     # 1/с

    # Проекция точки приложения на u_min (масштаб внешней силы)
    proj = float(u_min @ phi_xs) if phi_xs is not None else 1.0
    proj_abs = max(1e-6, abs(proj))              # защита от узла

    F_char_modal = K_eff * qeq                    # Н
    F_char_ext   = F_char_modal / proj_abs        # Н

    alpha_max = eps * F_char_ext * omega_star     # Н/с

    return dict(alpha_max=alpha_max,
                K_eff=K_eff, m_eff=m_eff,
                omega_well=omega_well, omega_pos1=omega_pos1, omega_star=omega_star,
                F_char_ext=F_char_ext, proj=proj, q_eq=qeq)



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
def animate_test(x_s: float):
    """
    Протокол: F(t)=α t до события SNAP (по центру масс), затем F(t)=F* (hold).
    Пороговая сила F* = α * t_snap печатается в заголовке.
    """
    # -------- справочный период для масштаба времени/шага --------
    pos_idx = np.flatnonzero(omega > 0.0)
    omega_ref = omega[pos_idx[0]] if pos_idx.size > 0 else abs(omega_signed[0])
    T_ref = 2.0*np.pi / max(omega_ref, 1e-12)
    max_step = (1/200.0) * T_ref

    # форма в точке приложения силы
    phi_xs = np.array([np.interp(x_s, x, Phi[i]) for i in range(N_modes)])

    # -------- квазистатическая скорость нарастания (рекомендация) --------
    rec = recommend_quasistatic_alpha(eps=0.03)   # 3% инерционности
    alpha = 0.6 * rec["alpha_max"]                # небольшой запас
    print(f"[QS] alpha<= {rec['alpha_max']:.3e} N/s; using alpha={alpha:.3e} N/s")

    # -------- начальные условия --------
    y0 = np.zeros(2 * N_modes)
    y0[:N_modes] = np.sign(start_in_well) * q_eq_vec



    # -------- этап 1: растим силу F(t)=α t до события снапа --------
    rhs_ramp    = make_rhs(make_ramp(alpha), phi_xs)
    snap_event  = make_snap_event_cm(start_in_well)
    t_final_1   = 60.0 * T_ref          # «длинное» окно, обычно остановимся раньше по событию

    sol1 = solve_ivp(rhs_ramp, (0.0, t_final_1), y0, method="RK45",
                     rtol=rtol, atol=atol, max_step=max_step, events=snap_event)

    t  = sol1.t
    Q  = sol1.y[:N_modes, :]
    Qd = sol1.y[N_modes:, :]

    snapped = (sol1.t_events[0].size > 0)
    t_snap  = sol1.t_events[0][0] if snapped else None
    F_snap  = alpha * t_snap if snapped else None

    # -------- этап 2: держим силу на F* и «догоняем» систему --------
    if snapped:
        y_snap   = sol1.y[:, -1]
        rhs_hold = make_rhs(make_hold(F_snap), phi_xs)
        t_final_2 = t_snap + 5.0 * T_ref

        sol2 = solve_ivp(rhs_hold, (t_snap, t_final_2), y_snap, method="RK45",
                         rtol=rtol, atol=atol, max_step=max_step)

        # склейка
        if sol2.t.size > 0:
            t  = np.hstack([sol1.t,            sol2.t])
            Q  = np.hstack([sol1.y[:N_modes, :],  sol2.y[:N_modes, :]])
            Qd = np.hstack([sol1.y[N_modes:, :],  sol2.y[N_modes:, :]])

    # -------- подготовка данных для анимации профиля --------
    ids = np.linspace(0, t.size-1, anim_frames, dtype=int)
    profiles = np.zeros((anim_frames, x.size))
    for kf, idx in enumerate(ids):
        profiles[kf] = Q[:, idx] @ Phi

    # -------- энергии первых 5 линейных мод как функции времени --------
    E_lin, _, _ = linear_modal_energies(Q, Qd)  # (N_modes, Nt)
    E5 = E_lin[:5, :]                           # (5, Nt)

    # -------- фигуры/оси --------
    fig = plt.figure(figsize=(9.6, 6.4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.3], hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # Верх: профиль w(x,t)
    (ln,) = ax1.plot([], [], lw=2)
    ax1.set_xlim(0, L)
    ymin, ymax = 1.1*np.min(profiles), 1.1*np.max(profiles)
    if ymin == ymax:
        ymin -= 1e-6; ymax += 1e-6
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel("x [м]"); ax1.set_ylabel("w(x,t) [м]")
    title_text = ax1.text(
        0.5, 1.02, "", transform=ax1.transAxes,
        ha="center", va="bottom", fontsize=11,
    )
    ax1.grid(True, ls='--', alpha=0.3)

    # Низ: энергии E_i(t)
    lines = []
    for i in range(5):
        (li,) = ax2.plot(t, E5[i], lw=1.5, label=f"мода {i+1}")
        lines.append(li)
    ax2.set_xlabel("t [с]"); ax2.set_ylabel("Энергия E_i(t)")
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.legend(loc="upper right", ncols=2, fontsize=9, framealpha=0.9)

    # Вертикальная каретка времени; отметка момента снапа
    cursor = ax2.axvline(t[0], color='k', lw=1.2, ls='-')
    if snapped:
        ax2.axvline(t_snap, color='r', lw=1.5, ls='--')

    # Заголовки
    status = "SNAP" if snapped else "NO SNAP"
    if snapped:
        title_static = (f"x_s/L={x_s/L:.3f}, ramp α={alpha:.3e} N/s, "
                        f"F*={F_snap:.4g} N @ t*={t_snap:.4f} s, status: {status}")
        ax2.set_title(f"Энергии первых 5 мод (SNAP в t={t_snap:.5f} с)")
    else:
        title_static = (f"x_s/L={x_s/L:.3f}, ramp α={alpha:.3e} N/s, status: {status}")
        ax2.set_title("Энергии первых 5 мод")

    # Инициализация/обновление анимации
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
        blit=False, interval=1000/anim_fps, repeat=True
    )

    plt.show()


# -------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ПЕРВЫХ 5 ЛИНЕЙНЫХ СОБСТВЕННЫХ МОД (по V_eig)
# -------------------------------------------------------------
def reconstruct_linear_modes(Phi: np.ndarray, V_eig: np.ndarray, m: int = 5) -> np.ndarray:
    """
    Возвращает массив форм ψ_i(x) первых m линейных мод (в физическом пространстве),
    нормированных на max|ψ_i| = 1 для сравнимости.
      Phi   : (N_modes, Nx)  — базисные формы
      V_eig : (N_modes, N_modes) — собственные векторы (массо-ортонормированные)
      m     : сколько мод вывести
    Выход:
      Psi   : (m, Nx) — формы ψ_i(x)
    """
    m = min(m, V_eig.shape[1])
    Psi = np.empty((m, Phi.shape[1]))
    for i in range(m):
        psi = V_eig[:, i] @ Phi    # (1 x N_modes) @ (N_modes x Nx) -> (Nx,)
        # нормировка для наглядности
        amp = np.max(np.abs(psi))
        Psi[i] = psi / (amp if amp > 0 else 1.0)
    return Psi

def plot_first5_linear_modes(x: np.ndarray, Phi: np.ndarray, V_eig: np.ndarray,
                             omega_signed: np.ndarray, title: str = None):
    """
    Рисует первые 5 линейных мод ψ_i(x) и печатает их частоты (со знаком).
    Отрицательный знак указывает на линейную неустойчивость (ω^2<0).
    """
    Psi = reconstruct_linear_modes(Phi, V_eig, m=5)
    plt.figure(figsize=(9.6, 5.2))
    for i in range(Psi.shape[0]):
        lbl = f"mode {i+1}: ω = {omega_signed[i]:.3g} rad/s"
        plt.plot(x, Psi[i], lw=2, label=lbl)
    plt.axhline(0, color='k', lw=0.8, ls=':')
    plt.xlim(0, L)
    plt.xlabel("x [m]")
    plt.ylabel("ψ_i(x) (normalized)")
    if title is None:
        title = "First 5 linear eigenmodes in physical space (mass-normalized, max|ψ|=1)"
    plt.title(title)
    plt.grid(True, ls='--', alpha=0.3)
    plt.legend(loc="best", fontsize=9, ncols=1)
    plt.tight_layout()
    plt.show()



# =============================================================
# Запуск всех тестов
# =============================================================
if __name__ == "__main__":
    print(f"P/Pcr={P/Pcr:.3f}; N_modes={N_modes}; zeta_default={zeta_default}")

    # --- показать первые 5 линейных мод, по которым строится нижний график ---
    plot_first5_linear_modes(x, Phi, V_eig, omega_signed,
                             title="Первые 5 линейных мод (по V_eig), max|ψ|=1")

    print('----- Recommendation --------')
    rec = recommend_quasistatic_alpha(eps=0.03)  # 3% инерционности
    alpha = 0.6 * rec["alpha_max"]  # небольшой запас
    print(f"[QS] alpha<= {rec['alpha_max']:.3e} N/s; using alpha={alpha:.3e} N/s")
    print('-----------------------------')


    for x_s in TESTS:
        print(f"Test: x_s/L={x_s/L:.3f}")
        animate_test(x_s)
    print("Done.")
