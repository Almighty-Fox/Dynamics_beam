from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy import integrate
import pandas as pd


# ---------определяем параметры материала образца----------
a = 10e-3  # сторона квадратного сечения
S = a * a  # площадь сечения балки
I_inertia = a * a ** 3 / 12  # момент инерции сечения
ro = 7850  # плотность
E = 2e11  # модуль Юнга
nu = 0.3  # коэффициент Пуассона
# ---------определяем параметры каппы, местоположения барьера и длины балки----------
k = np.arange(0.0001, 20, 0.0001)  # лист значений каппы - замененная частота
# k_step = 0.000001
# k = np.concatenate((np.arange(2.4, 2.43, k_step), np.arange(3.1, 3.2, k_step), np.arange(5.55, 5.6, k_step)))
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


# # характеристическое уравнение для применения встроенноего метода поиска корнец fsolve (настроить на нахождение всех корней не получилось)
# def func(k):
#     return -((k*np.cos(k*l1)-k*np.cosh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k*np.sin(k*l1)-k*np.sinh(k*l1))) * \
#            ((-k**2*np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1)) -
#       (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
#       (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
#       (-k**2*np.cos(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k**2*np.sinh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k**2*np.cosh(k*l1))) + \
#            ((-k**2*np.sin(k*l1)-k**2*np.sinh(k*l1)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1))*(-k**2*np.cos(k*l1)-k**2*np.cosh(k*l1))) * \
#            ((k*np.cos(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)) -
#       (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
#       (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
#       (-k*np.sin(k*l1)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*k*np.cosh(k*l1) + (np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*k*np.sinh(k*l1)))
#
# # решение х-го ур-ния встроенным методом fsolve
# roots = fsolve(func, np.array([2.42, 3.14, 5.7, 7.84, 8.6, 9.3]))
# print('Корни, найденные с помощью fsolve')
# print(roots)

# решаем х-ое ур-ние ручным способом, так мы точно найдем более менее точно все корни с погрешностью в шаг изменения каппы
roots_my = []
for i in range(len(fun) - 1):
    # print(i/len(fun))
    if (fun[i] * fun[i+1] < 0) and (abs(fun[i] * fun[i+1]) < 300):
        roots_my.append(((k[i] + k[i+1]) / 2))
print('Корни, найденные вручную')
print(roots_my)
# roots_my = [1, 4, 6]

# print('\nCheck root')
# print(-A1_fun(roots_my[0]) * B2_fun(roots_my[0]) + A2_fun(roots_my[0]) * B1_fun(roots_my[0]))

omega_lst = np.sqrt(E * I_inertia / ro / S) * np.array(roots_my) ** 2
omega_lst_2 = (E * I_inertia / ro / S) * np.array(roots_my) ** 4
nu_lst = omega_lst / 2 / np.pi
print('Собственные частоты балки с упором на барьер (квадраты)')
print(omega_lst_2)
omega_lst_2_format = [('%.3g' % ii) for ii in omega_lst_2]
for ii in omega_lst_2_format:
    print(ii + '\t', end='')
print()

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
    print('Integral v1 square= ', str(v1_2_integral))
    v2_2_integral = integrate.quad(v2_2_fun, l1, l2)[0]
    print('Integral v2 square = ', str(v2_2_integral))
    alpha1_2 = E * I_inertia / ro / S / (v1_2_integral + (A1_fun(k) / B1_fun(k))**2 * v2_2_integral)
    print('beta1 ', str((A1_fun(k) / B1_fun(k))**2 * v2_2_integral))
    alpha1 = alpha1_2 ** (1/2)

    return alpha1


v1 = []  # лист форм левой части балки при различных частотах
v2 = []  # тоже для правой части
v1v2 = []  # форма всей балки
x_step = 1e-5
for i in range(len(roots_my)):
    k = roots_my[i]
    # k = roots_my[2]  # переобозначаем каппу, вместо листа делаем скаляром
    # alpha1 = find_aplha1(k)  # ищем альфа по нормализации
    alpha1 = 1
    # print(alpha1)
    beta1 = alpha1 * A1_fun(k) / B1_fun(k)
    x = np.arange(0, l1, x_step)
    v1.append(alpha1 * ((np.sin(k*x) - np.sinh(k*x)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1)) * (np.cos(k*x) - np.cosh(k*x))))
    x = np.arange(l1, l2, x_step)
    v2.append(beta1 * ((np.sin(k*x) + (np.cos(k*l2)*np.cosh(k*l2) - np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*x) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*x))
                  - (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (np.cos(k*x)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*x)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*x))))

    v1v2.append(np.concatenate((v1[i], v2[i])))  # форма всей балки как сумма формы левой и правой частей

    # plt.figure()
    # # plt.plot(list(np.arange(0, l1, x_step)) + list(np.arange(l1, l2, x_step)), list(v1[i]) + list(v2[i]), linewidth=2)
    # plt.plot(np.arange(0, l2, x_step), v1v2[i], linewidth=2)
    # plt.plot([l1], [0], 'ro', linewidth=2)
    # plt.grid()
    # # plt.plot(np.arange(0, l1, 0.0001), v1)
    # # plt.plot(np.arange(l1, l2, 0.0001), v2)
# plt.show()

check_orthogonal = np.zeros((len(roots_my), len(roots_my)))
for i in range(len(roots_my)):
    for j in range(len(roots_my)):
        print('i = {}, j = {}'.format(i, j))
        vivj = v1v2[i] * v1v2[j]
        # integ_vivj = x_step * sum(vivj)
        integ_vivj = 0
        for ii in range(1, len(vivj)):
            integ_vivj += (vivj[ii] + vivj[ii-1]) / 2
        integ_vivj = integ_vivj * x_step
        # print('Integral = {}'.format(integ_vivj))
        check_orthogonal[i][j] = integ_vivj

# не красивый вывод матрицы ортогональности на экран
# for row in check_ortog:
#     for val in row:
#         print('%1.2g' % val, end='\t')
#     print()

# красивый вывод матрицы ортогональности на экран
# Determine the maximum width for each column
max_widths = [max(len(f'{x:.2e}') for x in col) for col in check_orthogonal.T]
# Print the formatted array with aligned columns
print()
for row in check_orthogonal:
    row_str = ' '.join([f'{x:.2e}'.rjust(max_width) for x, max_width in zip(row, max_widths)])
    print(row_str)

# нахождение поля скоростей консольно-закрепленной балки в момент до удара о барьер
x = np.arange(0, l2, x_step)
al_l_1 = 1.8864
omega_1 = al_l_1**2 / l2**2 * np.sqrt(E * I_inertia / ro / S)
initial_vel = -omega_1 * ((np.cos(al_l_1) + np.cosh(al_l_1)) / (np.sin(al_l_1) + np.sinh(al_l_1)) * (np.sin(al_l_1 / l2 * x) - np.sinh(al_l_1 / l2 * x)) +
                          (np.cosh(al_l_1 / l2 * x) - np.cos(al_l_1 / l2 * x))) / 50  # /50 для масштаба


# нахождение коэффициентов альфа в ряду собственных форм балки, опертой о барьер
alpha_lst = []
for i in range(len(omega_lst)):
    vel_form = initial_vel * v1v2[i]
    integ_vel_form = 0
    for ii in range(1, len(vel_form)):
        integ_vel_form += (vel_form[ii] + vel_form[ii-1]) / 2
    integ_vel_form = integ_vel_form * x_step

    alpha_lst.append(1 / omega_lst[i] * integ_vel_form)
print('\nAlpha list = {}'.format(alpha_lst))
# plt.figure()
# plt.bar(np.arange(1, len(alpha_lst)+1), alpha_lst)

# строим формы колебаний с учетом найденных коэффициентов альфа
v1 = []  # лист форм левой части балки при различных частотах
v2 = []  # тоже для правой части
v1v2 = []  # форма всей балки
for i in range(len(roots_my)):
    k = roots_my[i]
    alpha1 = alpha_lst[i]
    # print(alpha1)
    beta1 = alpha1 * A1_fun(k) / B1_fun(k)
    x = np.arange(0, l1, x_step)
    v1.append(alpha1 * ((np.sin(k*x) - np.sinh(k*x)) - (np.sin(k*l1) - np.sinh(k*l1))/(np.cos(k*l1)-np.cosh(k*l1)) * (np.cos(k*x) - np.cosh(k*x))))
    x = np.arange(l1, l2, x_step)
    v2.append(beta1 * ((np.sin(k*x) + (np.cos(k*l2)*np.cosh(k*l2) - np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*x) + (np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*x))
                  - (np.sin(k*l1)+(np.cos(k*l2)*np.cosh(k*l2)-np.sin(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.sin(k*l2)-np.cos(k*l2)*np.sinh(k*l2))*np.cosh(k*l1))/
          (np.cos(k*l1) - (np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*l1)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*l1)) *
          (np.cos(k*x)-(np.sin(k*l2)*np.cosh(k*l2)+np.cos(k*l2)*np.sinh(k*l2))*np.sinh(k*x)+(np.cosh(k*l2)*np.cos(k*l2)+np.sin(k*l2)*np.sinh(k*l2))*np.cosh(k*x))))

    v1v2.append(np.concatenate((v1[i], v2[i])))  # форма всей балки как сумма формы левой и правой частей

    plt.figure()
    # plt.plot(list(np.arange(0, l1, x_step)) + list(np.arange(l1, l2, x_step)), list(v1[i]) + list(v2[i]), linewidth=2)
    plt.title('{} natural form'.format(i+1))
    plt.plot(np.arange(0, l2, x_step), v1v2[i], linewidth=2)
    plt.plot([l1], [0], 'ro', linewidth=2)
    plt.grid()

# строим итоговый график формы, как сумма всех собственных форм, умноженных на соответствующие коэффициенты
plt.figure()
plt.title('The sum of all natural forms')
plt.plot(np.arange(0, l2, x_step), sum(v1v2), linewidth=2)
plt.plot([l1], [0], 'ro', linewidth=2)
plt.grid()

plt.show()