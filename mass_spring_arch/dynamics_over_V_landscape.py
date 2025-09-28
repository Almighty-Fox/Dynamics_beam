import numpy as np
from numpy import sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import random

# ------------------------------------------------------------
# 0. User settings
# ------------------------------------------------------------
# T_FINAL = 0.005  # total integration time, s
T_FINAL = 0.035
MAX_STEP = 1e-5  # max solver step
RTOL, ATOL = 1e-7, 1e-10
sample_every = 1  # record every Nth solver step

# Define 10 initial velocity sets (v1, v2) in m/s
# initial_velocities = [
#     (-10000, -1000), (-800, -7000), (-3400, -6000),
#     (-2000, -4000), (-20300, -2000), (200, 0),
#     (1000, 2000)
# ]

# random.seed(552)
# initial_velocities = [
#     (random.randint(4000, 15000), random.randint(15000, 25000))
#     for _ in range(13)
# ]


def finding_vel_init(alpha):
    # alpha = 0.03
    # K_init = 3.104e+07
    K_init = 3.20277e+05
    v1_init = np.sqrt(K_init * (1 + alpha))
    v2_init = np.sqrt(K_init * (1 - alpha))
    return (v1_init, v2_init)

initial_velocities = [finding_vel_init(2e-11)]

print("initial_velocities = [")
for v1, v2 in initial_velocities:
    print(f"    ({v1}, {v2}),")
print("]")

# ------------------------------------------------------------
# 1. System parameters
# ------------------------------------------------------------
a1, a2, a3 = 0.5, 0.5, 0.5  # geometry, m
l01, l02, l03 = 0.62, 0.57, 0.62  # natural lengths, m
k1, k2, k3 = 4.0e7, 3.34e7, 4.0e7  # spring stiffness, N/m
k_theta = 2.0e3  # torsional stiffness, N·m/rad
c = 10.0  # damping, N·s/m
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


# --- событие: все (y1, y2, v1, v2) < 0 ---
def all_negative_event(t, y, eps=0.0):
    y1, y2, v1, v2 = y
    # Возвращаем максимум — как только максимум опустится ниже 0, все четыре < 0
    # Можно поставить небольшой зазор eps = 1e-12, чтобы избежать погрешностей
    return max(y1 - eps, y2 - eps, v1 - eps, v2 - eps)


all_negative_event.terminal = False   # True — если хотите останавливать интегрирование
all_negative_event.direction = -1     # ловим переход сверху вниз через 0
# ------------------------------------------------------------
# 4. Simulate trajectories
# ------------------------------------------------------------
y1_trajs = []
y2_trajs = []
# approximate "up" equilibrium
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

# ---------- критические точки: решение Q1=Q2=0 и классификация ----------

def hessian_V(y1, y2, h=1e-6):
    """Численный гессиан V по центральным разностям."""
    f = potential
    f00   = f(y1,     y2)
    fpp10 = f(y1 + h, y2)
    fmm10 = f(y1 - h, y2)
    f0p01 = f(y1,     y2 + h)
    f0m01 = f(y1,     y2 - h)
    fpp11 = f(y1 + h, y2 + h)
    fpm1m = f(y1 + h, y2 - h)
    fmp1m = f(y1 - h, y2 + h)
    fmm11 = f(y1 - h, y2 - h)

    V_y1y1 = (fpp10 - 2.0*f00 + fmm10) / (h**2)
    V_y2y2 = (f0p01 - 2.0*f00 + f0m01) / (h**2)
    V_y1y2 = (fpp11 - fpm1m - fmp1m + fmm11) / (4.0*h**2)

    return np.array([[V_y1y1, V_y1y2],
                     [V_y1y2, V_y2y2]])

def classify_point(y1, y2, eps=1e-6):
    """min / max / saddle по собственным значениям гессиана."""
    w = np.linalg.eigvalsh(hessian_V(y1, y2))
    if (w >  eps).all():
        return "min"
    if (w < -eps).all():
        return "max"
    return "saddle"

def find_critical_points(xmin, xmax, ymin, ymax, n_seeds=21, extra_seeds=()):
    """
    Мультиядерный поиск решений Q1=Q2=0:
    - сетка стартов по прямоугольнику;
    - + дополнительные старты из уже найденных минимумов/максимумов;
    - отбрасывание дублей по близости.
    """
    y1_lin = np.linspace(xmin, xmax, n_seeds)
    y2_lin = np.linspace(ymin, ymax, n_seeds)
    seeds = [(x, y) for x in y1_lin for y in y2_lin]
    seeds += list(extra_seeds)

    roots = []
    for s in seeds:
        try:
            r = fsolve(static_residuals, s, xtol=1e-12, maxfev=500)
            if np.all(np.isfinite(r)):
                roots.append(tuple(r))
        except Exception:
            pass

    # удаляем дубликаты (по радиусу tol)
    uniq = []
    tol = 1e-3
    for p in roots:
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) > tol for q in uniq):
            uniq.append(p)

    # классифицируем и добавляем значение потенциала
    result = []
    for y1c, y2c in uniq:
        t = classify_point(y1c, y2c)
        result.append({"y1": y1c, "y2": y2c, "type": t, "V": potential(y1c, y2c)})
    return result

# ---------- потенциальная энергия --------------------------------------
def V(y1, y2):
    dL1, dL2, dL3, th1, th2, *_ = geometry(y1, y2)
    return 0.5 * (k1 * dL1 ** 2 + k2 * dL2 ** 2 + k3 * dL3 ** 2 + k_theta * (th1 ** 2 + th2 ** 2))

V_up = V(*y_eq_up)
V_origin = V(0, 0)
print(f'Potential energy init = {V_up}')
print(f'Potential energy origin = {V_origin}')

y_eq = (y_eq_up[0], y_eq_up[1])

y_hit_lst = []  # запоминаем точки, в которых наступило событие
for v1_init, v2_init in initial_velocities:
    sol = solve_ivp(eom, (0, T_FINAL), [y_eq[0], y_eq[1], -v1_init, -v2_init],
                    max_step=MAX_STEP, rtol=RTOL, atol=ATOL, events=[all_negative_event])
    indices = np.arange(0, sol.y.shape[1], sample_every)
    y1_trajs.append(sol.y[0, indices])
    y2_trajs.append(sol.y[1, indices])

    # # Вывести True/False для данного прогона
    # hit = sol.t_events[0].size > 0
    # print("Все < 0 одновременно:", hit)
    #
    # # Если нужно знать момент времени и состояние:
    # if hit:
    #     t_hit = sol.t_events[0][0]
    #     # состояние в момент события (интерполяция):
    #     y_hit = sol.sol(t_hit) if sol.sol is not None else sol.y[:, np.searchsorted(sol.t, t_hit)]
    #     print(f"Момент: t = {t_hit:.6e}, состояние = {y_hit}")
    #     y_hit_lst.append([y_hit[0], y_hit[1]])

    # все времена события
    t_hits = sol.t_events[0]

    for t_hit in t_hits:
        # состояние в момент события
        y_hit = sol.sol(t_hit) if sol.sol is not None else sol.y[:, np.searchsorted(sol.t, t_hit)]
        y_hit_lst.append([y_hit[0], y_hit[1]])
        print(f"Событие: t = {t_hit:.6e}, y = {y_hit}")
        y_hit_lst.append([y_hit[0], y_hit[1]])

# ------------------------------------------------------------
# 5. Plot potential & trajectories in original style
# ------------------------------------------------------------
# Grid for V
y_vals = np.linspace(-0.75, 0.75, 600)
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

# ---------------------------------------------------------------------------------------------
# находим значения минимумом и максимумов, поэтому берем большие диапазоны по осям
y_vals_log = np.linspace(-0.4, 0.4, 600)
Y1_log, Y2_log = np.meshgrid(y_vals_log, y_vals_log)
V_log = np.zeros_like(Y1_log)

for i in range(Y1_log.shape[0]):
    for j in range(Y1_log.shape[1]):
        V_log[i, j] = potential(Y1_log[i, j], Y2_log[i, j])

# Identify extrema
dV_max_log = maximum_filter(V_log, size=15, mode='nearest')
dV_min_log = minimum_filter(V_log, size=15, mode='nearest')
maxima_mask_log = (V_log == dV_max_log)
minima_mask_log = (V_log == dV_min_log)
maxima_coords_log = np.argwhere(maxima_mask_log)
minima_coords_log = np.argwhere(minima_mask_log)

print('Min potential energy:')
print(V_log[minima_coords_log[:, 0], minima_coords_log[:, 1]])
print('Max potential energy:')
print(V_log[maxima_coords_log[:, 0], maxima_coords_log[:, 1]])

# Дополнительные старты из детектора экстремумов (чтобы fsolve попадал в нужные области)
extra = []
if minima_coords_log.size > 0:
    xs_min = Y1_log[minima_coords_log[:, 0], minima_coords_log[:, 1]]
    ys_min = Y2_log[minima_coords_log[:, 0], minima_coords_log[:, 1]]
    extra += list(zip(xs_min, ys_min))
if maxima_coords_log.size > 0:
    xs_max = Y1_log[maxima_coords_log[:, 0], maxima_coords_log[:, 1]]
    ys_max = Y2_log[maxima_coords_log[:, 0], maxima_coords_log[:, 1]]
    extra += list(zip(xs_max, ys_max))

# можно добавить и вашу найденную "верхнюю" точку как старт
extra.append((y_eq_up[0], y_eq_up[1]))

# Находим особые точки в окне визуализации
crit = find_critical_points(
    xmin=y_vals_log[0], xmax=y_vals_log[-1], ymin=y_vals_log[0], ymax=y_vals_log[-1],
    n_seeds=19,  # 19–25 обычно достаточно; больше — медленнее, но надёжнее
    extra_seeds=extra
)

print("Найденные особые точки (type, y1, y2, V):")
for p in crit:
    print(p['type'], f"({p['y1']:.6f}, {p['y2']:.6f})", f"V={p['V']:.6e}")


# рисуем профиль потенциальной энергии, что бы понимать вообще с чем работаем
# plt.figure(figsize=(10, 7))
# # filled contours
# filled = plt.contourf(Y1_log, Y2_log, V_log, levels=70, alpha=0.75)
# # contour lines
# contours = plt.contour(Y1_log, Y2_log, V_log, levels=70, linewidths=0.8)
# plt.clabel(contours, inline=True, fontsize=8)
#
# # ---- седла: чёрные кресты, и подписи всех особых точек ----
# # сначала сами седла
# saddle_pts = [(p['y1'], p['y2']) for p in crit if p['type'] == 'saddle']
# if saddle_pts:
#     xs_sad, ys_sad = zip(*saddle_pts)
#     plt.scatter(xs_sad, ys_sad, marker='x', c='black', s=60, zorder=6, label='Saddles')
#
# # теперь максимумы и минимумы
# max_min_pts = [(p['y1'], p['y2']) for p in crit if p['type'] in ('min', 'max')]
# if max_min_pts:
#     xs_sad, ys_sad = zip(*max_min_pts)
#     plt.scatter(xs_sad, ys_sad, marker='o', facecolors='none', edgecolors='red', s=60, zorder=6, label='Max / min')
#
# # подписи координат для всех особых точек (min/max/saddle)
# for p in crit:
#     plt.annotate(
#         f"({p['y1']:.2f}, {p['y2']:.2f})",  # можно добавить:  + f"\nV={p['V']:.2e}"
#         (p['y1'], p['y2']),
#         textcoords="offset points",
#         xytext=(6, 4),  # сдвиг подписи от маркера
#         fontsize=8,
#         color='k'
#     )
#
# plt.xlabel('y1 (m)')
# plt.ylabel('y2 (m)')
# plt.title('Trajectories over 2D potential energy landscape')
# plt.legend(bbox_to_anchor=(0.02, 1), loc='upper left', framealpha=0.2)
# plt.colorbar(filled, label='V value')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# --- левый subplot: исходный 2D-контурный рисунок ---
filled = ax1.contourf(Y1_log, Y2_log, V_log, levels=70, alpha=0.75)
contours = ax1.contour(Y1_log, Y2_log, V_log, levels=70, linewidths=0.8)
ax1.clabel(contours, inline=True, fontsize=8)

# седла: чёрные кресты
saddle_pts = [(p['y1'], p['y2']) for p in crit if p['type'] == 'saddle']
if saddle_pts:
    xs_sad, ys_sad = zip(*saddle_pts)
    ax1.scatter(xs_sad, ys_sad, marker='x', c='black', s=60, zorder=6, label='Saddles')

# максимумы и минимумы (исправил условие отбора)
max_min_pts = [(p['y1'], p['y2']) for p in crit if p['type'] in ('min', 'max')]
if max_min_pts:
    xs_mm, ys_mm = zip(*max_min_pts)
    ax1.scatter(xs_mm, ys_mm, marker='o', facecolors='none', edgecolors='red', s=60,
                zorder=6, label='Max / min')

# подписи координат всех особых точек
for p in crit:
    ax1.annotate(f"({p['y1']:.2f}, {p['y2']:.2f})\nV={p['V']:.2e}",
                 (p['y1'], p['y2']),
                 textcoords="offset points", xytext=(6, 4),
                 fontsize=8, color='k')

ax1.set_xlabel('y1 (m)')
ax1.set_ylabel('y2 (m)')
ax1.set_title('2D potential: contours + critical points')
ax1.legend(bbox_to_anchor=(0.02, 1), loc='upper left', framealpha=0.2)
cbar1 = fig.colorbar(filled, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('V value')
ax1.grid(True)

# --- правый subplot: 3D поверхность потенциала ---
# чтобы не было слишком тяжело, чуть проредим сетку
skip = 12  # подбери по скорости/детализации
X3 = Y1_log[::skip, ::skip]
Y3 = Y2_log[::skip, ::skip]
Z3 = V_log[::skip, ::skip]

surf = ax2.plot_surface(X3, Y3, Z3, rstride=1, cstride=1, linewidth=0,
                        antialiased=True, cmap='viridis', alpha=0.95)
# изолинии на "дне" для визуального соответствия левому графику
ax2.contour(X3, Y3, Z3, levels=20, offset=Z3.min(), linewidths=0.6, cmap='viridis')

ax2.set_xlabel('y1 (m)')
ax2.set_ylabel('y2 (m)')
ax2.set_zlabel('V')
ax2.set_title('3D potential energy surface')
ax2.view_init(elev=35, azim=-60)
cbar2 = fig.colorbar(surf, ax=ax2, fraction=0.046, pad=0.08)
cbar2.set_label('V value')

plt.show()
# ---------------------------------------------------------------------------------------------




# рисунок вычисления траектории масс поверх потенциального рельефа
plt.figure(figsize=(10, 7))
# filled contours
filled = plt.contourf(Y1, Y2, V, levels=70, alpha=0.75)
# contour lines
contours = plt.contour(Y1, Y2, V, levels=70, linewidths=0.8)
plt.clabel(contours, inline=True, fontsize=8)

# # mark minima and maxima
# if minima_coords.size > 0:
#     xs_min = Y1[minima_coords[:, 0], minima_coords[:, 1]]
#     ys_min = Y2[minima_coords[:, 0], minima_coords[:, 1]]
#     plt.scatter(xs_min, ys_min, marker='o', facecolors='none', edgecolors='red', label='Local minima')
# if maxima_coords.size > 0:
#     xs_max = Y1[maxima_coords[:, 0], maxima_coords[:, 1]]
#     ys_max = Y2[maxima_coords[:, 0], maxima_coords[:, 1]]
#     plt.scatter(xs_max, ys_max, marker='X', facecolors='none', edgecolors='black', label='Local maxima')

# седла: чёрные кресты
saddle_pts = [(p['y1'], p['y2']) for p in crit if p['type'] == 'saddle']
if saddle_pts:
    xs_sad, ys_sad = zip(*saddle_pts)
    plt.scatter(xs_sad, ys_sad, marker='x', c='black', s=60, zorder=6, label='Saddles')

# максимумы и минимумы (исправил условие отбора)
max_min_pts = [(p['y1'], p['y2']) for p in crit if p['type'] in ('min', 'max')]
if max_min_pts:
    xs_mm, ys_mm = zip(*max_min_pts)
    plt.scatter(xs_mm, ys_mm, marker='o', facecolors='none', edgecolors='red', s=60,
                zorder=6, label='Max / min')

# подписи координат всех особых точек
for p in crit:
    plt.annotate(f"({p['y1']:.2f}, {p['y2']:.2f})\nV={p['V']:.2e}",
                 (p['y1'], p['y2']),
                 textcoords="offset points", xytext=(6, 4),
                 fontsize=8, color='k')

# overlay trajectories
colors = plt.cm.tab10(np.linspace(0, 1, len(y1_trajs)))
for i, (y1_t, y2_t) in enumerate(zip(y1_trajs, y2_trajs)):
    plt.plot(y1_t, y2_t, color=colors[i], label=f"set {i + 1}")

for ii in range(len(y_hit_lst)):
    plt.plot([y_hit_lst[ii][0]],[y_hit_lst[ii][1]], 'bo', lw=0.7)

plt.xlabel('y1 (m)')
plt.ylabel('y2 (m)')
plt.title('Trajectories over 2D potential energy landscape')
plt.legend(bbox_to_anchor=(0.02, 1), loc='upper left', framealpha=0.2)
plt.colorbar(filled, label='V value')
plt.grid(True)
plt.tight_layout()
plt.show()
