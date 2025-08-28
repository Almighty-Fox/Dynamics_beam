import numpy as np
from numpy import sqrt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import maximum_filter, minimum_filter

# ----------------------------------------------------------------------
# 1. Parameters
# ----------------------------------------------------------------------
a1 = 0.5
a2 = 0.5
a3 = 0.5

l01 = 1.5
l02 = 0.5
l03 = 1.5

k1 = 3.0e7
k2 = 3.0e7
k3 = 3.0e7
k_theta = 2.0e3

m = 1.0

RTOL = 1e-7
ATOL = 1e-10
dt_visual_initial = 1e-3
dt_visual = 1e-5
max_step_solver = 1e-5
T_final = 0.07

# ----------------------------------------------------------------------
# 2. Geometry helper
# ----------------------------------------------------------------------
def geometry(y1, y2):
    dL1 = sqrt(a1**2 + y1**2) - l01
    dL2 = sqrt(a2**2 + (y2 - y1)**2) - l02
    dL3 = sqrt(a3**2 + y2**2) - l03

    theta1 = np.arctan2(y1, a1)
    theta2 = np.arctan2(y2, a3)

    dL1_dy1 = y1 / sqrt(a1**2 + y1**2)
    dL2_dy1 = (y1 - y2) / sqrt(a2**2 + (y2 - y1)**2)
    dtheta1_dy1 = a1 / (a1**2 + y1**2)

    dL3_dy2 = y2 / sqrt(a3**2 + y2**2)
    dL2_dy2 = (y2 - y1) / sqrt(a2**2 + (y2 - y1)**2)
    dtheta2_dy2 = a3 / (a3**2 + y2**2)

    return (dL1, dL2, dL3, theta1, theta2,
            dL1_dy1, dL2_dy1, dtheta1_dy1,
            dL3_dy2, dL2_dy2, dtheta2_dy2)

# ----------------------------------------------------------------------
# 3. Potential energy surface
# ----------------------------------------------------------------------
y_vals = np.linspace(-0.3, 0.3, 500)
Y1, Y2 = np.meshgrid(y_vals, y_vals)

V = np.zeros_like(Y1)
for i in range(Y1.shape[0]):
    for j in range(Y1.shape[1]):
        dL1, dL2, dL3, th1, th2, *_ = geometry(Y1[i, j], Y2[i, j])
        V[i, j] = 0.5*(k1*dL1**2 + k2*dL2**2 + k3*dL3**2 + k_theta*(th1**2 + th2**2))

# Find extrema
V_max = maximum_filter(V, size=15, mode='nearest')
V_min = minimum_filter(V, size=15, mode='nearest')
maxima_mask = (V == V_max)
minima_mask = (V == V_min)
maxima_coords = np.argwhere(maxima_mask)
minima_coords = np.argwhere(minima_mask)

# Plot 3D surface and contours
fig = plt.figure(figsize=(12, 6))
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.plot_surface(Y1, Y2, V)
ax3d.set_xlabel('y1 (m)')
ax3d.set_ylabel('y2 (m)')
ax3d.set_zlabel('V(y1, y2)')
ax3d.set_title('Surface of Potential Energy V(y1, y2)')

ax2d = fig.add_subplot(1, 2, 2)
filled = ax2d.contourf(Y1, Y2, V, levels=20, alpha=0.75)
contours = ax2d.contour(Y1, Y2, V, levels=20, linewidths=0.8)
ax2d.clabel(contours, inline=True, fontsize=8)
if minima_coords.size > 0:
    ys_min = Y2[minima_coords[:,0], minima_coords[:,1]]
    xs_min = Y1[minima_coords[:,0], minima_coords[:,1]]
    ax2d.scatter(xs_min, ys_min, marker='o', facecolors='none', edgecolors='red', label='Minima')
if maxima_coords.size > 0:
    ys_max = Y2[maxima_coords[:,0], maxima_coords[:,1]]
    xs_max = Y1[maxima_coords[:,0], maxima_coords[:,1]]
    ax2d.scatter(xs_max, ys_max, marker='X', facecolors='none', edgecolors='black', label='Maxima')
ax2d.set_xlabel('y1 (m)')
ax2d.set_ylabel('y2 (m)')
ax2d.set_title('Contour of V(y1, y2)')
ax2d.legend()
fig.colorbar(filled, ax=ax2d, label='V value')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 4. External forces
# ----------------------------------------------------------------------
def f1(t):
    return 0

def f2(t):
    return 0

# ----------------------------------------------------------------------
# 5. Equations of motion
# ----------------------------------------------------------------------
def eom(t, y):
    y1, y2, v1, v2 = y
    (dL1, dL2, dL3, theta1, theta2,
     dL1_dy1, dL2_dy1, dtheta1_dy1,
     dL3_dy2, dL2_dy2, dtheta2_dy2) = geometry(y1, y2)
    Q1 = k1*dL1*dL1_dy1 + k2*dL2*dL2_dy1 + k_theta*theta1*dtheta1_dy1
    Q2 = k3*dL3*dL3_dy2 + k2*dL2*dL2_dy2 + k_theta*theta2*dtheta2_dy2
    c = 100  # damping
    a1_dd = (f1(t) - Q1 - c*v1)/m
    a2_dd = (f2(t) - Q2 - c*v2)/m
    return [v1, v2, a1_dd, a2_dd]

# ----------------------------------------------------------------------
# 6. Static equilibria
# ----------------------------------------------------------------------
def static_residuals(Y):
    y1, y2 = Y
    dL1, dL2, dL3, th1, th2, dL1_dy1, dL2_dy1, dth1_dy1, dL3_dy2, dL2_dy2, dth2_dy2 = geometry(y1, y2)
    Q1 = k1*dL1*dL1_dy1 + k2*dL2*dL2_dy1 + k_theta*th1*dth1_dy1
    Q2 = k3*dL3*dL3_dy2 + k2*dL2*dL2_dy2 + k_theta*th2*dth2_dy2
    return [Q1, Q2]

y_eq_up = fsolve(static_residuals, [1.1, 1.1])
print(y_eq_up)
y_eq_down = fsolve(static_residuals, [-1.1, -1.1])

# ----------------------------------------------------------------------
# 7. Dynamic simulation and energy plot
# ----------------------------------------------------------------------
# Initialize storage
times, KE_list, PE_list, TE_list = [], [], [], []

# Prepare figure and axes
x0, x1, x2, x3 = 0.0, a1, a1+a2, a1+a2+a3
plt.style.use('seaborn-notebook')
fig2, (ax_sim, ax_en) = plt.subplots(1, 2, figsize=(12, 5))

# Simulation axis
ax_sim.set_aspect('equal')
ax_sim.set_xlim(-0.1, x3+0.1)
ax_sim.set_ylim(-1.2*max(l01,l03), 1.2*max(l01,l03))
ax_sim.set_xlabel('x (m)')
ax_sim.set_ylabel('y (m)')
ax_sim.set_title('Two-mass Arch – Real-time Simulation')
anchor_style = dict(marker='s', markersize=8, color='black', linestyle='None')
mass_style   = dict(marker='o', markersize=10, color='tab:red', linestyle='None')
anchor_left, = ax_sim.plot(x0, 0.0, **anchor_style)
anchor_right,= ax_sim.plot(x3, 0.0, **anchor_style)
mass1, = ax_sim.plot([], [], **mass_style)
mass2, = ax_sim.plot([], [], **mass_style)
spring1,= ax_sim.plot([], [], '-', lw=1.8)
spring2,= ax_sim.plot([], [], '-', lw=1.8)
spring3,= ax_sim.plot([], [], '-', lw=1.8)
text_time = ax_sim.text(0.02, 0.95, '', transform=ax_sim.transAxes)

# Energy axis
line_ke, = ax_en.plot([], [], label='Kinetic E')
line_pe, = ax_en.plot([], [], label='Potential E')
line_te, = ax_en.plot([], [], label='Total E', linewidth=2)
ax_en.set_xlim(0, T_final)
ax_en.set_xlabel('Time (s)')
ax_en.set_ylabel('Energy (J)')
ax_en.set_title('Energy vs Time')
ax_en.legend(loc='upper right')
ax_en.grid(True)

# Initial conditions
current_time = 0.0
K = 1.452e+08
alpha = 0.92
v1_init = -np.sqrt(K * (1 + alpha))
v2_init = -np.sqrt(K * (1 - alpha))
state = np.array([y_eq_up[0], y_eq_up[1], v1_init, v2_init])
print(y_eq_up[0])

# Integration step function
def integrate_step(t0, y0, dt):
    sol = solve_ivp(eom, (t0, t0+dt), y0, max_step=max_step_solver,
                     rtol=RTOL, atol=ATOL)
    return sol.y[:, -1]

# Animation function
def animate(frame):
    global state, current_time, dt_visual
    state = integrate_step(current_time, state, dt_visual)
    current_time += dt_visual

    y1, y2, v1, v2 = state
    # Update simulation
    mass1.set_data([x1], [y1])
    mass2.set_data([x2], [y2])
    spring1.set_data([x0, x1], [0.0, y1])
    spring2.set_data([x1, x2], [y1, y2])
    spring3.set_data([x2, x3], [y2, 0.0])
    text_time.set_text(f"t = {current_time:5.3f} s")

    # Compute energies
    dL1, dL2, dL3, th1, th2, *_ = geometry(y1, y2)
    PE = 0.5*(k1*dL1**2 + k2*dL2**2 + k3*dL3**2 + k_theta*(th1**2 + th2**2))
    KE = 0.5*m*(v1**2 + v2**2)
    TE = PE + KE

    # Store and update energy plot
    times.append(current_time)
    PE_list.append(PE)
    KE_list.append(KE)
    TE_list.append(TE)
    line_ke.set_data(times, KE_list)
    line_pe.set_data(times, PE_list)
    line_te.set_data(times, TE_list)
    ax_en.relim()
    ax_en.autoscale_view()

    # Adjust dt_visual
    v_mag = np.hypot(v1, v2)
    if v_mag > 5.0:
        dt_visual = max(dt_visual/2, dt_visual_initial)
    elif v_mag < 0.5:
        dt_visual = min(dt_visual*1.2, dt_visual_initial)

    return (mass1, mass2, spring1, spring2, spring3,
            text_time, line_ke, line_pe, line_te)

ani = FuncAnimation(fig2, animate, blit=True, interval=20,
                    cache_frame_data=False)
plt.tight_layout()
plt.show()
