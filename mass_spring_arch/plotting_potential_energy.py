import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 1. Parameters
a1 = a3 = 0.5
a10 = a30 = 1.0
a12 = 0.5
a120 = 0.5
k1 = k3 = 3.0e7
k12 = 3.0e6
k_theta = 2.0e3

# 2. Geometric functions
def delta_L1(y):
    return np.sqrt(a1**2 + y**2) - a10

def delta_L3(y):
    return np.sqrt(a3**2 + y**2) - a30

def delta_L12(y1, y2):
    return np.sqrt(a12**2 + (y2 - y1)**2) - a120

def theta1(y):
    return np.arctan(y/a1)

def theta2(y):
    return np.arctan(y/a3)

# 3. Potential energy
def V_numeric(y1, y2):
    return (0.5*(k1*delta_L1(y1)**2 + k3*delta_L3(y2)**2 + k12*delta_L12(y1, y2)**2)
            + 0.5*k_theta*(theta1(y1)**2 + theta2(y2)**2))



# 6. 2D contour plot + markers
N = 400
y1_vals = np.linspace(-1.2, 1.2, N)
y2_vals = np.linspace(-1.2, 1.2, N)
Y1, Y2 = np.meshgrid(y1_vals, y2_vals)
V_vals = np.zeros_like(Y1)
for i in tqdm(range(N), desc='Computing V grid'):  # progress bar
    for j in range(N):
        V_vals[i, j] = V_numeric(Y1[i, j], Y2[i, j])

plt.figure(figsize=(8, 6))
cs = plt.contour(Y1, Y2, V_vals, levels=25)
plt.clabel(cs, inline=True, fontsize=8)
plt.xlabel('y1')
plt.ylabel('y2')
plt.title('2D Contour of V')
# plt.show()

# 7. 3D surface plot + saddle markers
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Y1, Y2, V_vals, cmap=cm.Spectral, edgecolor='none')
# mark saddles
ax.set_xlabel('y1')
ax.set_ylabel('y2')
ax.set_zlabel('V')
plt.title('3D Surface of V')
plt.show()
