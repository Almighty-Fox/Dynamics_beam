from class_beam_elements import Class_beam_elements
from create_global_matrix import *
from chart_script import *


def main_body_fun(VI_flag, loc_bar=0.8):
    # ---------определяем параметры материала образца----------
    a = 10e-3  # сторона квадратного сечения
    S = a * a  # площадь сечения балки
    I_inertia = a * a ** 3 / 12  # момент инерции сечения
    ro = 7850 * S  # погонная плотность
    E = 2e11  # модуль Юнга
    # ---------определяем параметры геометрии и КЭ образца----------
    L = 1
    # MaxNode = 20 + 1  # количество узлов
    MaxNode = 1000 + 1  # количество узлов
    dl = L / (MaxNode - 1)
    dm = dl * ro
    # ------------------------------------------------------------

    elements = [Class_beam_elements(i, L, MaxNode, E, I_inertia, ro) for i in
                range(MaxNode - 1)]  # создаем массив балочных элементов

    # формируем матрицы для дифура
    global_stiffness = build_global_stiffness_matrix(elements, MaxNode)  # собираем глобальную МЖ (матрица жесткости)
    if VI_flag:
        global_stiffness = global_stiffness_matrix_with_GU_barrier(global_stiffness, MaxNode, loc_bar)  # вносим ГУ в МЖ с учетом барьера
    else:
        global_stiffness = global_stiffness_matrix_with_GU(global_stiffness)  # вносим ГУ в МЖ
    # print(global_stiffness)

    global_mass = build_global_mass_matrix(elements, MaxNode)  # собирает глобальную ММ (матрица масс)
    # print(global_mass)

    eigenvalues, eigenvectors_normalized = create_modal_matrix(global_stiffness,
                                                               global_mass)  # создаем модальную матрицу для перехода в модальные координаты для расчета возбуждаемый мод
    # print((ro / E / I_inertia * np.array(eigenvalues[:10])) ** (1/4))  # вывод корней аналитического уравнения

    roots2 = (ro / E / I_inertia * np.array(eigenvalues)) ** (1 / 4)

    eigenvectors_normalized_tr = np.transpose(eigenvectors_normalized)
    natural_shapes = [eigenvectors_normalized_tr[i][::2] for i in range(len(eigenvectors_normalized_tr))]

    # фильтруем собственные формы, убираем шумовые
    noise_lst = []
    threshold = min(max(abs(natural_shapes[5])), max(abs(natural_shapes[6])), max(abs(natural_shapes[7])))
    for i in range(len(natural_shapes)):
        if max(abs(natural_shapes[i])) > 2 * threshold:  # критерий шума
            noise_lst.append(i)

    natural_shapes_pure = [natural_shapes[i] for i in range(len(natural_shapes)) if i not in noise_lst]
    roots2_pure = [roots2[i] for i in range(len(roots2)) if i not in noise_lst]

    print('First roots')
    print(roots2_pure[:maxN])

    # ----------------------------------------

    plt.figure()
    plt.grid()
    plt.title(f'{maxN} natural frequencies')
    cnt = 0
    # for i in range(shiftN, maxN + shiftN):
    for i in range(maxN):
        cnt += 1
        plt.plot(np.linspace(0, 1, MaxNode), natural_shapes_pure[i], label=f'fr={cnt}')

    return roots2_pure[:maxN], natural_shapes_pure[:maxN]


def saving_beam_shapes(roots2, natural_shapes, VI_flag):
    if VI_flag:
        path = './plots/saving_beam_shapes.txt'
    else:
        path = './plots/saving_beam_shapes_noVI.txt'

    with open(path, 'w') as cur_file:
        # cur_file.write(', '.join(map(str, roots2[shiftN:maxN + shiftN])))

        # cur_file.write(str(list(roots2[shiftN:maxN + shiftN])) + '\n')
        # cur_file.write(str(list(map(list, natural_shapes[shiftN:maxN + shiftN]))))

        cur_file.write(str(list(roots2)) + '\n')
        cur_file.write(str(list(map(list, natural_shapes))))


def reading_beam_shapes(path):
    with open(path, 'r') as cur_file:
        data = cur_file.readlines()

    roots = eval(data[0])
    natural_shapes = eval(data[1])

    print('Chech reading fuction')
    # print(roots)
    # print(natural_shapes)
    print(f'len(roots) = {len(roots)}')

    return roots, natural_shapes


if __name__ == '__main__':
    maxN = 20  # количество рассматриваемых первых мод
    # shiftN = 0  # количество шумовых мод, стоящих вначале

    VI_flag = True
    [roots2, natural_shapes] = main_body_fun(VI_flag, loc_bar=0.8)  # находим собственные частоты и собственные формы балки с барьером
    saving_beam_shapes(roots2, natural_shapes, VI_flag)  # сохраняем собственные частоты и собственные формы балки с барьером

    # Проверяем записанные в файл частоты и формы
    # path = r'./plots/saving_beam_shapes_noVI.txt'
    # reading_beam_shapes(path)

    plt.legend()
    plt.show()
