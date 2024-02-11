from class_beam_elements import Class_beam_elements
from create_global_matrix import *
from chart_script import *


def main_body_fun(loc_bar=0.8):
    # ---------определяем параметры материала образца----------
    a = 10e-3  # сторона квадратного сечения
    S = a * a  # площадь сечения балки
    I_inertia = a * a ** 3 / 12  # момент инерции сечения
    ro = 7850 * S  # погонная плотность
    E = 2e11  # модуль Юнга
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    # MaxNode = 20 + 1  # количество узлов
    MaxNode = 500 + 1  # количество узлов
    dl = L / (MaxNode - 1)
    dm = dl * ro
    # ------------------------------------------------------------

    elements = [Class_beam_elements(i, L, MaxNode, E, I_inertia, ro) for i in
                range(MaxNode - 1)]  # создаем массив балочных элементов

    # формируем матрицы для дифура
    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ (матрица жесткости)
    global_stiffness = global_stiffness_matrix_with_GU_barrier(global_stiffness, MaxNode,
                                                               loc_bar)  # вносим ГУ в МЖ с учетом барьера
    # print(global_stiffness)

    global_mass = build_global_mass_matrix(elements, MaxNode)  # собирает глобальную ММ (матрица масс)
    # print(global_mass)

    eigenvalues, eigenvectors_normalized = create_modal_matrix(global_stiffness,
                                                               global_mass)  # создаем модальную матрицу для перехода в модальные координаты для расчета возбуждаемый мод
    # print((ro / E / I_inertia * np.array(eigenvalues[:10])) ** (1/4))  # вывод корней аналитического уравнения

    roots2 = (ro / E / I_inertia * np.array(eigenvalues[:10])) ** (1 / 4)

    # maxN = 7
    print('First roots')
    print(roots2[:maxN])

    eigenvectors_normalized_tr = np.transpose(eigenvectors_normalized)
    natural_shapes = [eigenvectors_normalized_tr[i][::2] for i in range(len(eigenvectors_normalized_tr))]

    # i = 3
    # plt.plot(np.linspace(0, 1, MaxNode), natural_shapes[i])

    plt.figure()
    plt.grid()
    cnt = 0
    for i in range(shiftN, maxN + shiftN):
        cnt += 1
        plt.plot(np.linspace(0, 1, MaxNode), natural_shapes[i], label=f'fr={cnt}')

    return roots2, natural_shapes


def saving_beam_shapes(roots2, natural_shapes):
    with open('./plots/saving_beam_shapes.txt', 'w') as cur_file:
        # cur_file.write(', '.join(map(str, roots2[shiftN:maxN + shiftN])))

        cur_file.write(str(list(roots2[shiftN:maxN + shiftN])) + '\n')
        cur_file.write(str(list(map(list, natural_shapes[shiftN:maxN + shiftN]))))


if __name__ == '__main__':
    maxN = 7  # количество рассматриваемых первых мод
    shiftN = 2  # количество шумовых мод, стоящих вначале

    [roots2, natural_shapes] = main_body_fun(loc_bar=0.8)  # находим собственные частоты и собственные формы балки с барьером
    saving_beam_shapes(roots2, natural_shapes)  # сохраняем собственные частоты и собственные формы балки с барьером

    plt.legend()
    plt.show()
