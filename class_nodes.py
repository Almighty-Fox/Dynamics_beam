class Class_nodes:
    def __init__(self, i):
        self.id = i  # id узла
        self.u = 0  # перемещение узла
        self.q = 0  # поворот узла

        self.u1 = 0  # перемещение узла на пред шаге по времени
        self.q1 = 0  # поворот узла на пред шаге по времени
        self.u2 = 0  # перемещение узла на 2 шага назад по времени
        self.q2 = 0  # поворот узла на 2 шага назад по времени

        self.du = 0  # скорость узла
        self.ddu = 0  # ускорение узла
        self.dq = 0  # угловая скорость узла
        self.ddq = 0  # угловое ускорение узла

    def calc_vel(self, dt):
        self.du = (self.u - self.u1) / dt
        self.dq = (self.q - self.q1) / dt

    def calc_acc(self, dt):
        self.ddu = (self.u - 2 * self.u1 + self.u2) / dt**2
        self.ddq = (self.q - 2 * self.q1 + self.q2) / dt**2
