import numpy as np


class Class_beam_elements:
    def __init__(self, i, L, maxnode, E, I, ro):
        self.id = i
        self.E = E  # модуль Юнга
        self.I = I  # момент инерции
        self.ro = ro  # погонная плотность
        self.dl = L / (maxnode - 1)  # длина КЭ

        self.kel = 2 * self.E * self.I / self.dl ** 3 \
                   * np.array(  # создаем матрицу жесткости для каждого элемента (Stiffness_matrix_beam_elements.jpg)
            [[6, 3 * self.dl, -6, 3 * self.dl],
             [3 * self.dl, 2 * self.dl * self.dl, -3 * self.dl, self.dl * self.dl],
             [-6, -3 * self.dl, 6, -3 * self.dl],
             [3 * self.dl, self.dl * self.dl, -3 * self.dl, 2 * self.dl * self.dl]])

        self.mel = self.ro * self.dl / 420 \
                   * np.array(  # создаем матрицу масс для каждого элемента (Mass_matrix_beam_elements.jpg)
            [[156, 22 * self.dl, 54, -13 * self.dl],
             [22 * self.dl, 4 * self.dl * self.dl, 13 * self.dl, -3 * self.dl * self.dl],
             [54, 13 * self.dl, 156, -22 * self.dl],
             [-13 * self.dl, -3 * self.dl * self.dl, -22 * self.dl, 4 * self.dl * self.dl]])

        # self.mel = self.ro * self.dl / 2 / 105 \
        #            * np.array(  # создаем матрицу масс для каждого элемента (Mass_matrix_beam_elements.jpg)
        #     [[78, 22 * self.dl/2, 27, -13 * self.dl/2],
        #      [22 * self.dl/2, 8 * self.dl * self.dl / 4, 13 * self.dl / 2, -6 * self.dl * self.dl / 4],
        #      [27, 13 * self.dl / 2, 78, -22 * self.dl/2],
        #      [-13 * self.dl/2, -6 * self.dl * self.dl / 4, -22 * self.dl / 2, 8 * self.dl * self.dl / 4]])