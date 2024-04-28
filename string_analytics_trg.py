import numpy as np
from matplotlib import pyplot as plt


def trg_t_a(a, t):
    if abs(round(t / 2 / a) - (t / 2 / a)) < 1e-5:
        return 0
    else:
        return np.pi / 2 * (1 - 1/a * (t - 2*a * ((t/2/a) // 1)))


def S_a_t(a, t, N):
    return a**2 * sum([np.sin(np.pi * j * t / a) / j / (j**2 - a**2) for j in range(1, N + 1)])


def F_tr(a, t, N, f1):
    return -2 * f1 * np.sin(np.pi * a) * (trg_t_a(a, t) + trg_t_a(1-a, t) + S_a_t(a, t, N) + S_a_t(1-a, t, N))


def read_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        # Assuming each file contains a single line with data in the format "[x, y, z]"
        data = eval(content)
        return data


def plot_fit_an_fem():
    time_lst = read_file(r'./plots/VI_force_string_FEM_time.txt')
    y_values = read_file(r'./plots/VI_force_string_FEM_values.txt')

    plt.figure(1)
    plt.plot(time_lst, y_values, 'r', label='FEM')

    plt.legend()
    plt.xlabel('Time')
    # plt.ylabel('Functional')
    plt.ylabel('VI force')
    plt.grid()
    # plt.show()


if __name__ == "__main__":
    a = 0.5
    N = 100
    f1 = -1 / np.pi
    t_lst = np.linspace(0, 0.5, 1000)
    F_tr_ans = []
    for t in t_lst:
        F_tr_ans.append(F_tr(a, t, N, f1))

    plt.plot(t_lst, F_tr_ans, 'b', linewidth=1, label='Analytics')
    plt.grid()
    # plt.show()

    plot_fit_an_fem()

    plt.legend()
    plt.grid()
    plt.title('Comparison of analytics and FEM')
    plt.show()