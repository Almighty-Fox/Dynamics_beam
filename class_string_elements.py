import numpy as np


class Class_string_elements:
    def __init__(self, i, L, maxnode, E, S, ro, point_bar, dict_elements):
        self.id = i
        self.E = E  # модуль Юнга
        self.S = S  # площадь поперечного сечения
        self.ro = ro  # погонная плотность

        # self.dl = L / (maxnode - 1)  # длина КЭ
        # if (self.id >= 45) and (self.id <= 144):
        #     self.dl = L / 1000  # длина КЭ
        # else:
        #     self.dl = L / 100  # длина КЭ

        self.dl = dict_elements[self.id]


        # создаем матрицу жесткости для каждого элемента (Stiffness_matrix_beam_elements.jpg)
        self.kel = self.E * self.S / self.dl * np.array([[1, -1], [-1, 1]])

        # создаем матрицу масс для каждого элемента (Mass_matrix_beam_elements.jpg)
        # self.mel = self.ro * self.dl / 6 * np.array([[2, 1], [1, 2]])
        self.mel = self.ro * self.dl / 2 * np.array([[1, 0], [0, 1]])
