from fenics import *
import numpy as np

# Параметры модели
L = 1.0  # Длина балки
W = 0.1  # Ширина балки
H = 0.01  # Толщина балки
E = 210e9  # Модуль Юнга
nu = 0.3  # Коэффициент Пуассона
rho = 7850  # Плотность материала
P = 1000.0  # Максимальная сила

# Сетка
nx, ny = 50, 5
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, H), nx, ny, 1)

# Пространство функций
V = VectorFunctionSpace(mesh, 'P', 2)


# Граничные условия
def clamped_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0)


bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Начальная конфигурация
u = Function(V)
v = TestFunction(V)
du = TrialFunction(V)


# Нелинейные деформации и напряжения
def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T + nabla_grad(u).T * nabla_grad(u))


def sigma(u):
    return lambda_ * tr(epsilon(u)) * Identity(d) + 2 * mu * epsilon(u)


# Модуль сдвига и коэффициент Пуассона
mu = E / (2 * (1 + nu))
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))

# Вариационная формулировка
F = inner(sigma(u), epsilon(v)) * dx - dot(Constant((0, 0, -P)), v) * ds

# Якобиан
J = derivative(F, u, du)

# Решение нелинейной задачи
problem = NonlinearVariationalProblem(F, u, bc, J)
solver = NonlinearVariationalSolver(problem)

# Параметры решения
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0

# Инкрементальная нагрузка
num_steps = 10
load_steps = np.linspace(0, P, num_steps)

for step in load_steps:
    F = inner(sigma(u), epsilon(v)) * dx - dot(Constant((0, 0, -step)), v) * ds
    problem = NonlinearVariationalProblem(F, u, bc, J)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()

    # Постпроцессинг
    print(f"Step {step} - Max displacement: {u.vector().norm('linf')}")
