import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0          # Length of the string (meters)
c = 1.0          # Wave speed (m/s)
T = 2.0          # Total time (seconds)
dx = 0.01        # Spatial step (meters)
dt = 0.01        # Time step (seconds)
beta = 0.25      # Newmark parameter beta
gamma = 0.5      # Newmark parameter gamma

# Derived parameters
nx = int(L / dx) + 1  # Number of spatial points
nt = int(T / dt) + 1  # Number of time steps
x = np.linspace(0, L, nx)  # Spatial grid

# Initialize displacement and velocities
u = np.zeros(nx)  # Displacement at current time step
v = np.zeros(nx)  # Velocity at current time step
a = np.zeros(nx)  # Acceleration at current time step

# Initial condition (pluck in middle)
u[nx//2] = 0.1

# Precompute constants
k = c**2 * dt**2 / dx**2
beta_dt2 = beta * dt**2

# Helper arrays for Newmark method
u_new = np.zeros(nx)
v_new = np.zeros(nx)
a_new = np.zeros(nx)

# Simulation
for t in range(1, nt):
    # Calculate new accelerations
    a_new[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) * k / beta_dt2

    # Update displacements
    u_new[1:-1] = u[1:-1] + dt * v[1:-1] + 0.5 * dt**2 * ((1 - 2*beta) * a[1:-1] + 2*beta * a_new[1:-1])

    # Update velocities
    v_new[1:-1] = v[1:-1] + dt * ((1 - gamma) * a[1:-1] + gamma * a_new[1:-1])

    # Apply boundary conditions
    u_new[0], u_new[-1] = 0, 0

    # Swap arrays
    u, u_new = u_new, u
    v, v_new = v_new, v
    a, a_new = a_new, a

    # Visualization at certain steps
    if t % 100 == 0:
        plt.plot(x, u, label=f't = {t*dt:.2f}s')

plt.title('String Displacement over Time')
plt.xlabel('Position along string (m)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()
