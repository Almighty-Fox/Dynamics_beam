import numpy as np
from scipy.optimize import fsolve
import math


def equation(x):
    return math.sinh(x) * math.cos(x) + 1

# Initial guess for the root
# initial_guess = 1

lst_fr = []
for initial_guess in [2, 4, 8, 11, 14, 17, 21, 24, 26, 30]:
    # Using fsolve to find the root of the equation
    root = fsolve(equation, initial_guess)

    print(f"The root of the equation is approximately: {root[0]}")
    lst_fr.append(root[0])


with open(r'./initial_disp/Analytical_natural_frequencies.txt', 'w') as cur_file:
    for ID, item1 in enumerate(lst_fr):
        E = 2e11
        a = 10e-3
        J = a ** 4 / 12
        F = a ** 2
        ro = 7850
        l = 1
        omega = item1 ** 2 / l ** 2 * (E * J / ro / F) ** (0.5)
        item2 = omega ** 2
        # cur_file.write(str(ID+1) + ") " + "%s\n" % item)
        cur_file.write(f"{ID + 1}) {item2:e}\n")