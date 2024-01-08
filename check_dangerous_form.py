from sympy import *
from create_global_matrix import *
import matplotlib.pyplot as plt


# определяем частоты, формы для балки без барьера
def beam_without_barrier():
    al_l = np.arange(0, 12, 0.00001)
    fun1 = np.sinh(al_l) * np.cos(al_l) + 1
    roots1 = []
    omega1 = []
    for i in range(len(fun1) - 1):
        if fun1[i] * fun1[i+1] < 0:
            roots1_i = (al_l[i] + al_l[i+1]) / 2
            roots1.append(roots1_i)
            omega1.append((roots1_i / l) ** 2 * np.power((E * J_inertia / ro / S), (1 / 2)))

    forms1 = []
    form1_second_dif = []
    for root in roots1:
        alpha_i = root / l

        u_i = lambda z: ((np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                np.sin(alpha_i * z) - np.sinh(alpha_i * z)) + (
                                 np.cosh(alpha_i * z) - np.cos(alpha_i * z)))
        form1_i = np.array([u_i(ii) for ii in np.linspace(0, l, point)])
        forms1.append(form1_i)

        u_i_second_dif = lambda z: (
                (np.cos(root) + np.cosh(root)) / (np.sin(root) + np.sinh(root)) * (
                -alpha_i ** 2 * np.sin(alpha_i * z) - alpha_i ** 2 * np.sinh(alpha_i * z)) + (
                        alpha_i ** 2 * np.cosh(alpha_i * z) + alpha_i ** 2 * np.cos(alpha_i * z)))
        form1_second_dif_i = np.array([u_i_second_dif(ii) for ii in np.linspace(0, l, point)])
        form1_second_dif.append(form1_second_dif_i)


    return roots1, forms1, omega1, form1_second_dif


def normalization_forms():  # нормализуем собственные формы, что бы их энергия равнялась бы заданному значению
    value_of_potential_energy = 1
    norm_coef_lst = np.array([(value_of_potential_energy / (0.5 * E * J_inertia * (dl * sum(form1_second_dif[id] ** 2)))) ** 0.5 for id in range(len(roots1))])
    print(norm_coef_lst)

    form1_norm = [norm_coef_lst[id] * forms1[id] for id in range(len(roots1))]

    form1_second_dif_norm = [norm_coef_lst[id] * form1_second_dif[id] for id in range(len(roots1))]
    stress_max_lst = [form1_second_dif_norm[id] * E * a / 2 for id in range(len(roots1))]

    return form1_norm, form1_second_dif_norm


def visualize(y_values_lst, ylabel, set_ylim=False):
    plt.figure(figsize=(9, 6))

    for ID, y_values_lst_item in enumerate(y_values_lst):
        plt.plot(np.linspace(0, l, point), y_values_lst_item, label='fr=' + str(ID+1))

    plt.xlabel('beam')
    plt.ylabel(ylabel)
    plt.title('Equality of potential energy')

    if set_ylim:
        max_val = max(y_values_lst[0])
        plt.ylim(-max_val * 1.2, max_val * 1.2)
        plt.xlim(-l / 25, l * 1.2)
        plt.plot([0, l], [max_val, max_val], 'k--')
        plt.plot([0, l], [-max_val, -max_val], 'k--')
        plt.plot([0, l], [0, 0], 'k--')

    plt.grid()
    plt.legend()
    # plt.show()

# -----------------------------------------------
ro = 7850
a = 10e-3
S = a * a  # площадь сечения балки
J_inertia = a * a ** 3 / 12  # момент инерции сечения
E = 2e11  # модуль Юнга
# ------------------------------------------------
point = 200 + 1  # количество элементов балки
l = 1  # длина балки
dl = l / (point - 1)
# ------------------------------------------------


[roots1, forms1, omega1, form1_second_dif] = beam_without_barrier()
form1_norm, form1_second_dif_norm = normalization_forms()
visualize(form1_norm, ylabel='Displacement')
visualize(form1_second_dif_norm, ylabel='Second diff', set_ylim=True)


plt.show()



