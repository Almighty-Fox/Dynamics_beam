import numpy as np
from matplotlib import pyplot as plt


def read_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        # Assuming each file contains a single line with data in the format "[x, y, z]"
        data = eval(content)
        return data


def plot_fit_an_fem():
    time_lst = read_file(r'./plots/VI_delta_finding_mistake/FEM_time_lst_VI_5_mode.txt')
    y_values = read_file(r'./plots/VI_delta_finding_mistake/FEM_time_disp_end_VI_5_mode.txt')
    plt.figure(1)
    plt.plot(time_lst, y_values, 'r', label='FEM')

    time_lst = read_file(r'./plots/VI_delta_finding_mistake/time_global_maple_roots_5_mode.txt')
    y_values = read_file(r'./plots/VI_delta_finding_mistake/y_end_global_maple_roots_5_mode.txt')
    plt.figure(1)
    plt.plot(time_lst, y_values, 'k', label='Green function')

    plt.figure(1)
    scale = max(abs(np.array(y_values)))
    plt.plot([0.0002872000000000061]*2, [-scale, scale], 'r--', label='Contact break')

    time_lst = read_file(r'./plots/VI_delta_finding_mistake/FEM_time_lst_VI_5_mode_free.txt')
    y_values = read_file(r'./plots/VI_delta_finding_mistake/FEM_time_disp_end_VI_5_mode_free.txt')
    plt.figure(1)
    plt.plot(time_lst, y_values, 'g--', label='FEM free')

    plt.legend()
    plt.xlabel('Time')
    # plt.ylabel('Functional')
    plt.ylabel('Displacement')
    plt.title('Comparison of beam end displacements')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    plot_fit_an_fem()