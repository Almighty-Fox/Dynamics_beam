import numpy as np
from numpy import sqrt
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Parameters (unchanged)
# ------------------------------------------------------------
a1 = 0.5
a2 = 0.5
a3 = 0.5  # m
l01 = 1.0
l02 = 0.5
l03 = 1.0  # m
k1 = 3.0e7
k2 = 3.0e6
k3 = 3.0e7  # N/m
k_theta = 2.0e3  # N·m/rad
m = 1.0  # kg

# ------------------------------------------------------------
# 2. Geometry helpers (unchanged)
# ------------------------------------------------------------
def geometry(y1, y2):
    dL1 = sqrt(a1**2 + y1**2) - l01
    dL2 = sqrt(a2**2 + (y2 - y1)**2) - l02
    dL3 = sqrt(a3**2 + y2**2) - l03

    th1 = np.arctan2(y1, a1)
    th2 = np.arctan2(y2, a3)

    # first derivatives
    dL1_y1 = y1 / sqrt(a1**2 + y1**2)
    dL2_y1 = (y1 - y2) / sqrt(a2**2 + (y2 - y1)**2)
    dTh1_y1 = a1 / (a1**2 + y1**2)

    dL3_y2 = y2 / sqrt(a3**2 + y2**2)
    dL2_y2 = (y2 - y1) / sqrt(a2**2 + (y2 - y1)**2)
    dTh2_y2 = a3 / (a3**2 + y2**2)

    # second derivatives
    d2L1_y1 = (a1**2) / (a1**2 + y1**2)**1.5
    d2L2_y1y1 = (a2**2) / (a2**2 + (y2 - y1)**2)**1.5
    d2Th1_y1 = -2 * a1 * y1 / (a1**2 + y1**2)**2

    d2L3_y2 = (a3**2) / (a3**2 + y2**2)**1.5
    d2L2_y2y2 = (a2**2) / (a2**2 + (y2 - y1)**2)**1.5
    d2Th2_y2 = -2 * a3 * y2 / (a3**2 + y2**2)**2

    # mixed
    d2L2_y1y2 = -a2**2 / (a2**2 + (y2 - y1)**2)**1.5

    return (dL1, dL2, dL3, th1, th2,
            dL1_y1, dL2_y1, dTh1_y1,
            dL3_y2, dL2_y2, dTh2_y2,
            d2L1_y1, d2L2_y1y1, d2Th1_y1,
            d2L3_y2, d2L2_y2y2, d2Th2_y2,
            d2L2_y1y2)

# ------------------------------------------------------------
# 3. Residual for equilibrium (takes F as argument)
# ------------------------------------------------------------
def residual(Y, F):
    y1, y2 = Y
    (dL1, dL2, dL3, th1, th2,
     dL1_y1, dL2_y1, dTh1_y1,
     dL3_y2, dL2_y2, dTh2_y2,
     *_rest) = geometry(y1, y2)

    dP_dy1 = (k1 * dL1 * dL1_y1 +
              k2 * dL2 * dL2_y1 +
              k_theta * th1 * dTh1_y1) - F  # F1 = F
    dP_dy2 = (k3 * dL3 * dL3_y2 +
              k2 * dL2 * dL2_y2 +
              k_theta * th2 * dTh2_y2) - F  # F2 = F
    return [dP_dy1, dP_dy2]

# ------------------------------------------------------------
# 4. Stiffness matrix in equilibrium
# ------------------------------------------------------------
def stiffness_matrix(y1_eq, y2_eq):
    (dL1, dL2, dL3, th1, th2,
     dL1_y1, dL2_y1, dTh1_y1,
     dL3_y2, dL2_y2, dTh2_y2,
     d2L1_y1, d2L2_y1y1, d2Th1_y1,
     d2L3_y2, d2L2_y2y2, d2Th2_y2,
     d2L2_y1y2) = geometry(y1_eq, y2_eq)

    K11 = (k1 * (dL1_y1**2 + dL1 * d2L1_y1) +
           k2 * (dL2_y1**2 + dL2 * d2L2_y1y1) +
           k_theta * (dTh1_y1**2 + th1 * d2Th1_y1))

    K22 = (k3 * (dL3_y2**2 + dL3 * d2L3_y2) +
           k2 * (dL2_y2**2 + dL2 * d2L2_y2y2) +
           k_theta * (dTh2_y2**2 + th2 * d2Th2_y2))

    K12 = k2 * (dL2_y1 * dL2_y2 + dL2 * d2L2_y1y2)
    return np.array([[K11, K12],
                     [K12, K22]])

# ------------------------------------------------------------
# 5. Sweep F = F1 = F2
# ------------------------------------------------------------
F_values = np.linspace(0, -6e6, 60)  # 60 points from 0 to 6e6
omega_values = []
c_cr_values = []
y_eq_values = []

# initial guess near symmetric configuration
Y0 = np.array([0.866, 0.866])

for F in F_values:
    sol, info, ier, _ = fsolve(residual, Y0, args=(F,), full_output=True)
    if ier != 1:
        # try switching sign (use magnitude of force)
        sol, info, ier, _ = fsolve(residual, Y0, args=(F,), full_output=True)
    y1_eq, y2_eq = sol
    y_eq_values.append(y1_eq)
    Y0 = sol  # warm start for next F

    K = stiffness_matrix(y1_eq, y2_eq)
    eigvals = np.linalg.eigvals(K / m)
    omega = np.sqrt(np.abs(eigvals))[0]  # both eigenvalues equal; take one
    omega_values.append(omega)
    c_cr_values.append(2 * m * omega)

omega_values = np.array(omega_values)
c_cr_values = np.array(c_cr_values)
y_eq_values = np.array(y_eq_values)

# ------------------------------------------------------------
# 6. Plotting
# ------------------------------------------------------------

F_values = -F_values  # поскольку математически отрицательную силу хотим визуализировать положительной

fig, axs = plt.subplots(1, 3, figsize=(16, 5))

axs[0].plot(F_values, y_eq_values, marker='o', color='black', markersize=4)
axs[0].set_xlabel("F1 = F2 (N)")
axs[0].set_ylabel("Equilibrium position y_eq (m)")
axs[0].set_title("y_eq vs symmetric axial force")
axs[0].grid(True)

axs[1].plot(F_values, omega_values, marker='o', color='blue', markersize=4)
axs[1].set_xlabel("F1 = F2 (N)")
axs[1].set_ylabel("Natural frequency ω (rad/s)")
axs[1].set_title("ω vs symmetric axial force")
axs[1].grid(True)

axs[2].plot(F_values, c_cr_values, marker='o', color='green', markersize=4)
axs[2].set_xlabel("F1 = F2 (N)")
axs[2].set_ylabel("Critical damping c_cr (N·s/m)")
axs[2].set_title("c_cr vs symmetric axial force")
axs[2].grid(True)

plt.tight_layout()
plt.show()
