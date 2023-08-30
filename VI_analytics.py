from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy import integrate


# ---------определяем параметры материала образца----------
a = 10e-3  # сторона квадратного сечения
S = a * a  # площадь сечения балки
I_inertia = a * a ** 3 / 12  # момент инерции сечения
ro = 7850 * S  # погонная плотность
E = 2e11  # модуль Юнга
nu = 0.3  # коэффициент Пуассона
# ---------определяем параметры каппы, местоположения барьера и длины балки----------
k = np.arange(0.00001, 10, 0.0001)  # лист значений каппы - замененная частота
l1 = 0.5  # местоположение барьера
l2 = 1  # длина балки

# задаем константы, полученные с помощью аналитики
A1 = ((k*np.cos(k*l1)-k*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k*np.sin(k*l1)-k*np.sinh(k*l1)))
A2 = ((-k**2*np.sin(k*l1)-k**2*np.sinh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k**2*np.cos(k*l1)-k**2*np.cosh(k*l1)))

B1 = ((k*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)) -
      (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
      (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
      (-k*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)))


B2 = ((-k**2*np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)) -
      (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
      (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
      (-k**2*np.cos(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)))

A1_fun = lambda k: ((k*np.cos(k*l1)-k*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k*np.sin(k*l1)-k*np.sinh(k*l1)))
A2_fun = lambda k: ((-k**2*np.sin(k*l1)-k**2*np.sinh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k**2*np.cos(k*l1)-k**2*np.cosh(k*l1)))

B1_fun = lambda k: ((k*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)) -
      (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
      (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
      (-k*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)))


B2_fun = lambda k: ((-k**2*np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)) -
      (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
      (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
      (-k**2*np.cos(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)))


# характеристическое уравнение
fun = -A1 * B2 + A2 * B1


# характеристическое уравнение для применения встроенноего метода поиска корнец fsolve (настроить на нахождение всех корней не получилось)
def func(k):
    return -((k*np.cos(k*l1)-k*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k*np.sin(k*l1)-k*np.sinh(k*l1))) * \
           ((-k**2*np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)) -
      (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
      (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
      (-k**2*np.cos(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1))) + \
           ((-k**2*np.sin(k*l1)-k**2*np.sinh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k**2*np.cos(k*l1)-k**2*np.cosh(k*l1))) * \
           ((k*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)) -
      (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
      (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
      (-k*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)))

# решение х-го ур-ния встроенным методом fsolve
roots = fsolve(func, np.array([2.42, 3.14, 5.7, 7.84, 8.6, 9.3]))
print('Корни, найденные с помощью fsolve')
print(roots)

# решаем х-ое ур-ние ручным способом, так мы точно найдем более менее точно все корни с погрешностью в шаг изменения каппы
roots_my = []
for i in range(len(fun) - 1):
    if fun[i] * fun[i+1] < 0:
        roots_my.append(((k[i] + k[i+1]) / 2))
print('Корни, найденные вручную')
print(roots_my)

omega_lst = np.sqrt(E * I_inertia / ro / S) * np.array(roots_my) ** 2
omega_lst_2 = (E * I_inertia / ro / S) * np.array(roots_my) ** 4
nu_lst = omega_lst / 2 / np.pi
print('Собственные частоты балки с упором на барьер')
print(omega_lst_2)

# выводим график х-го уравнения
plt.figure(1)
plt.plot(k, fun)
plt.grid()
plt.xlabel('Kappa')
plt.ylabel('Х-ое ур-ние')
plt.ylim(-600, 600)
# plt.show()


def find_aplha1(k):
    # проводим R-нормализацию  находим значение коэффициента альфа1
    v1_2_fun = lambda x: ((np.sin(k*x) - np.sinh(k*x)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1)) * (np.cos(k*x) - np.cosh(k*x))) ** 2
    v2_2_fun = lambda x: ((np.sin(k*x) + (np.cos(k*l2)*np.cosh(k*l2) - np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*x) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*x))
                      - (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
              (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
              (np.cos(k*x)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*x)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*x))) ** 2
    v1_2_integral = integrate.quad(v1_2_fun, 0, l1)[0]
    v2_2_integral = integrate.quad(v2_2_fun, l1, l2)[0]
    alpha1_2 = E * I_inertia / ro / S / (v1_2_integral + (A1_fun(k) / B1_fun(k))**2 * v2_2_integral)
    alpha1 = alpha1_2 ** (1/2)

    return alpha1


for k in roots_my:
    # k = roots_my[2]  # переобозначаем каппу, вместо листа делаем скаляром
    # alpha1 = find_aplha1(k)  # ищем альфа по нормализации
    alpha1 = 1
    # print(alpha1)
    beta1 = alpha1 * A1_fun(k) / B1_fun(k)
    x = np.arange(0, l1, 0.0001)
    v1 = alpha1 * ((np.sin(k*x) - np.sinh(k*x)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1)) * (np.cos(k*x) - np.cosh(k*x)))
    x = np.arange(l1, l2, 0.0001)
    v2 = beta1 * ((np.sin(k*x) + (np.cos(k*l2)*np.cosh(k*l2) - np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*x) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*x))
                  - (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (np.cos(k*x)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*x)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*x)))

    plt.figure()
    plt.plot(list(np.arange(0, l1, 0.0001)) + list(np.arange(l1, l2, 0.0001)), list(v1) + list(v2))
    plt.grid()
    # plt.plot(np.arange(0, l1, 0.0001), v1)
    # plt.plot(np.arange(l1, l2, 0.0001), v2)
plt.show()