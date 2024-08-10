import numpy as np
from matplotlib import pyplot as plt


def read_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        # Assuming each file contains a single line with data in the format "[x, y, z]"
        data = eval(content)
        return data


def read_file_2(file_name):
    with open(file_name, 'r') as cur_file:
        body_file = cur_file.readlines()

    # Step 1: Combine the list into a single string
    combined_str = ''.join(body_file)

    # Step 2: Remove any newline characters and extra spaces
    cleaned_str = combined_str.replace('\n', '').replace('[', '').replace(']', '')

    # Step 3: Split the string into individual numeric strings
    numeric_strs = cleaned_str.split()

    # Step 4: Convert each numeric string to a float
    float_list = [float(num) for num in numeric_strs]

    # Print the result
    print(float_list)

    return float_list


def plot_analytics():
    t = 0.5
    loc_bar = 0.7
    a = loc_bar
    sep1 = loc_bar - 0.5
    sep2 = loc_bar
    sep3 = 1.5 - loc_bar
    x_lst_sep1 = np.linspace(0, sep1, round(sep1 / 1e-3))
    x_lst_sep2 = np.linspace(sep1, sep2, round((sep2 - sep1) / 1e-3))
    x_lst_sep3 = np.linspace(sep2, sep3, round((sep3 - sep2) / 1e-3))
    x_lst_sep4 = np.linspace(sep3, 1, round((1 - sep3) / 1e-3))


    x = x_lst_sep1.copy()
    y_lst_sep1 = np.sin(np.pi * x) * np.sin(np.pi * t) - (4 / np.pi) * np.sin(np.pi * a) * (
        1/2 * np.pi * t * np.sin(np.pi * x) * np.sin(np.pi * a) * np.sin(np.pi * t) +
        np.cos(np.pi * t) * (1/2 * (-np.sin(np.pi * (a - x)) * np.pi * (1 - a + x) + 1 + 1/2 * np.cos(np.pi * (a - x))) -
         1/2 * (-np.sin(np.pi * (x + a)) * np.pi * (1 - x - a) + 1 + 1/2 * np.cos(np.pi * (x + a)))) -
        1/4 * (
            1/2 * (-np.sin(np.pi * (a + t - x)) * np.pi * (1 - a - t + x) + 1 + 1/2 * np.cos(np.pi * (a + t - x))) +
            1/2 * (-np.sin(np.pi * (a - x - t)) * np.pi * (1 - a + x + t) + 1 + 1/2 * np.cos(np.pi * (a - x - t))) -
            1/2 * (-np.sin(np.pi * (x + a - t)) * np.pi * (1 - x - a + t) + 1 + 1/2 * np.cos(np.pi * (x + a - t))) -
            1/2 * (-np.sin(np.pi * (x + a + t)) * np.pi * (1 - x - a - t) + 1 + 1/2 * np.cos(np.pi * (x + a + t)))
        )
    )


    x = x_lst_sep2.copy()
    y_lst_sep2 = np.sin(np.pi * x) * np.sin(np.pi * t) - (4 / np.pi) * np.sin(np.pi * a) * (
        1/2 * np.pi * t * np.sin(np.pi * x) * np.sin(np.pi * a) * np.sin(np.pi * t) +
        np.cos(np.pi * t) * (1/2 * (-np.sin(np.pi * (a - x)) * np.pi * (1 - a + x) + 1 + 1/2 * np.cos(np.pi * (a - x))) -
        1/2 * (-np.sin(np.pi * (x + a)) * np.pi * (1 - x - a) + 1 + 1/2 * np.cos(np.pi * (x + a)))) -
        1/4 * (
            1/2 * (-np.sin(np.pi * (a + t - x)) * np.pi * (1 - a - t + x) + 1 + 1/2 * np.cos(np.pi * (a + t - x))) +
            1/2 * (-np.sin(np.pi * (x - a + t)) * np.pi * (1 - x + a - t) + 1 + 1/2 * np.cos(np.pi * (x - a + t))) -
            1/2 * (-np.sin(np.pi * (x + a - t)) * np.pi * (1 - x - a + t) + 1 + 1/2 * np.cos(np.pi * (x + a - t))) -
            1/2 * (-np.sin(np.pi * (x + a + t)) * np.pi * (1 - x - a - t) + 1 + 1/2 * np.cos(np.pi * (x + a + t)))
        )
    )


    x = x_lst_sep3.copy()
    y_lst_sep3 = np.sin(np.pi * x) * np.sin(np.pi * t) - (4 / np.pi) * np.sin(np.pi * a) * (
        1/2 * np.pi * t * np.sin(np.pi * x) * np.sin(np.pi * a) * np.sin(np.pi * t) +
        np.cos(np.pi * t) * (1/2 * (-np.sin(np.pi * (x - a)) * np.pi * (1 - x + a) + 1 + 1/2 * np.cos(np.pi * (x - a))) -
        1/2 * (-np.sin(np.pi * (x + a)) * np.pi * (1 - x - a) + 1 + 1/2 * np.cos(np.pi * (x + a)))) -
        1/4 * (
            1/2 * (-np.sin(np.pi * (a + t - x)) * np.pi * (1 - a - t + x) + 1 + 1/2 * np.cos(np.pi * (a + t - x))) +
            1/2 * (-np.sin(np.pi * (x - a + t)) * np.pi * (1 - x + a - t) + 1 + 1/2 * np.cos(np.pi * (x - a + t))) -
            1/2 * (-np.sin(np.pi * (x + a - t)) * np.pi * (1 - x - a + t) + 1 + 1/2 * np.cos(np.pi * (x + a - t))) -
            1/2 * (-np.sin(np.pi * (x + a + t)) * np.pi * (1 - x - a - t) + 1 + 1/2 * np.cos(np.pi * (x + a + t)))
        )
    )


    x = x_lst_sep4.copy()
    y_lst_sep4 = np.sin(np.pi * x) * np.sin(np.pi * t) - (4 / np.pi) * np.sin(np.pi * a) * (
        1/2 * np.pi * t * np.sin(np.pi * x) * np.sin(np.pi * a) * np.sin(np.pi * t) +
        np.cos(np.pi * t) * (1/2 * (-np.sin(np.pi * (x - a)) * np.pi * (1 - x + a) + 1 + 1/2 * np.cos(np.pi * (x - a))) -
        1/2 * (-np.sin(np.pi * (x + a)) * np.pi * (1 - x - a) + 1 + 1/2 * np.cos(np.pi * (x + a)))) -
        1/4 * (
            1/2 * (-np.sin(np.pi * (a + t - x)) * np.pi * (1 - a - t + x) + 1 + 1/2 * np.cos(np.pi * (a + t - x))) +
            1/2 * (-np.sin(np.pi * (x - a + t)) * np.pi * (1 - x + a - t) + 1 + 1/2 * np.cos(np.pi * (x - a + t))) -
            1/2 * (-np.sin(np.pi * (x + a - t)) * np.pi * (1 - x - a + t) + 1 + 1/2 * np.cos(np.pi * (x + a - t))) -
            1/2 * (-np.sin(np.pi * (x + a + t - 2)) * np.pi * (1 - x - a - t + 2) + 1 + 1/2 * np.cos(np.pi * (x + a + t - 2)))
        )
    )

    x_lst = list(x_lst_sep1) + list(x_lst_sep2) + list(x_lst_sep3) + list(x_lst_sep4)
    y_lst = list(y_lst_sep1) + list(y_lst_sep2) + list(y_lst_sep3) + list(y_lst_sep4)

    return x_lst, y_lst


def plot_rude_way():
    a = 0.7
    t = 0.1
    x_lst = np.linspace(0, 1, 200)
    ss = np.zeros(len(x_lst))
    for k in range(1, 1000):
        ss += np.sin(np.pi * k * x_lst) * np.sin(np.pi * k * a) * (1 - np.cos(np.pi * k * t)) / k**2
    # return x_lst, np.sin(np.pi * x_lst) - 2 * np.sin(np.pi * a) * ss
    # return x_lst, np.sin(np.pi * x_lst) - 8 / np.pi ** 2 * np.sin(np.pi * a) * ss
    return x_lst, np.sin(np.pi * x_lst) - 4 / np.pi * np.sin(np.pi * a) * np.cos(np.pi * t) * ss


def plot_rude_way_test():
    a = 0.5
    x_lst = np.linspace(0, 1, 1000)
    ss = np.zeros(len(x_lst))
    for k in range(1, 1000):
        ss += np.sin(np.pi * k * x_lst) * np.sin(np.pi * k * a) * (1 - np.cos(np.pi * k * 0.5)) / k**2

    x_lst_2, y_lst_2 = x_lst, np.sin(np.pi * x_lst) - 8 / np.pi ** 2 * np.sin(np.pi * a) * ss
    y_lst_2 = -np.array(y_lst_2)
    plt.figure(1)
    plt.plot(x_lst_2, y_lst_2, '-g', label='rude')


    plt.legend()
    plt.xlabel('Time')
    # plt.ylabel('Functional')
    plt.ylabel('Displacement')
    plt.title('Comparison of beam end displacements')
    plt.grid()

    plt.show()



def plot_fit_an_fem():
    time_lst = read_file_2(r'./plots/string_break_contact/nodes_lst_k_1e4_bar_07.txt')
    y_values = read_file(r'./plots/string_break_contact/dis_lst_k_1e4_bar_07.txt')
    plt.figure(1)
    plt.plot(time_lst, np.array(y_values) * np.pi, 'r', label='FEM')

    time_lst = np.loadtxt(r'./plots/string_break_contact/x_green_07.txt')
    y_values = np.loadtxt(r'./plots/string_break_contact/y_green_07.txt')
    plt.figure(1)
    plt.plot(time_lst, -y_values, '--', label='Green', linewidth=4, alpha=0.8)

    # time_lst = read_file(r'./plots/VI_delta_finding_mistake/time_global_maple_roots_5_mode.txt')
    # y_values = read_file(r'./plots/VI_delta_finding_mistake/y_end_global_maple_roots_5_mode.txt')
    # plt.figure(1)
    # plt.plot(time_lst, y_values, 'k', label='Green function')

    x_lst, y_lst = plot_analytics()
    y_lst = -np.array(y_lst)
    plt.figure(1)
    plt.plot(x_lst, y_lst, '--k', label='analytics', linewidth=2)

    # x_lst_2, y_lst_2 = plot_rude_way()
    # y_lst_2 = -np.array(y_lst_2)
    # plt.figure(1)
    # plt.plot(x_lst_2, y_lst_2, '--g', label='rude')


    plt.legend()
    plt.xlabel('Time')
    # plt.ylabel('Functional')
    plt.ylabel('Displacement')
    plt.title('Comparison of beam end displacements')
    plt.grid()

    plt.show()


def plot_fit_an_fem_2():
    time_lst = read_file_2(r'./plots/string_break_contact/nodes_lst_k_1e4_bar_07__2.txt')
    y_values = read_file(r'./plots/string_break_contact/vel_lst_k_1e4_bar_07__2.txt')
    plt.figure(1)
    plt.plot(time_lst, np.array(y_values) * np.pi, 'r', label='FEM')

    time_lst = np.loadtxt(r'./plots/string_break_contact/x_green_07_test.txt')
    y_values = np.loadtxt(r'./plots/string_break_contact/vel_green_07_test.txt')
    plt.figure(1)
    plt.plot(time_lst, -y_values, '-', label='Green', linewidth=4, alpha=0.8)


    plt.legend()
    plt.xlabel('Time')
    # plt.ylabel('Functional')
    plt.ylabel('Velocity')
    plt.title('Comparison of beam end displacements')
    plt.grid()

    plt.show()


def plot_fit_an_fem_3():
    x_lst, y_lst = plot_analytics()
    y_lst = -np.array(y_lst)
    # plt.figure(1)
    plt.figure(figsize=(8, 4))
    plt.plot(x_lst, y_lst, '--r', label='Analytical', linewidth=3)

    time_lst = np.loadtxt(r'./plots/string_break_contact/x_green_07.txt')
    y_values = np.loadtxt(r'./plots/string_break_contact/y_green_07.txt')
    # plt.figure(1)
    plt.plot(time_lst, -y_values, label='Semi-analytical', linewidth=2, alpha=0.7)

    plt.plot([0.7], [-0.02], 'k^', markersize=10, markerfacecolor='none', markeredgecolor='black', label='Barrier')



    # x_lst_2, y_lst_2 = plot_rude_way()
    # y_lst_2 = -np.array(y_lst_2)
    # plt.figure(1)
    # plt.plot(x_lst_2, y_lst_2, '--g', label='rude')


    plt.legend()
    plt.xlabel('Time')
    # plt.ylabel('Functional')
    plt.ylabel('Displacement')
    plt.title('Displacement field at contact break')
    plt.grid()

    plt.show()


def mode_decomposition_form():
    x_lst, y_lst = plot_analytics()
    x_lst, y_lst = np.array(x_lst), np.array(y_lst)
    # Расчет коэффициентов для первых 10 мод
    num_modes = 10

    coefficients = []
    for k in range(1, num_modes + 1):
        mode = np.sin(np.pi * k * x_lst)
        coefficient = np.dot(y_lst, mode) * 2 / len(x_lst)  # Интеграл заменен на дискретную сумму
        coefficients.append(coefficient)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_modes + 1), coefficients, tick_label=[f'Mode {k}' for k in range(1, num_modes + 1)])
    plt.xlabel('Modes')
    plt.ylabel('Coefficients')
    plt.title('Contribution of the first 10 modes to the string shape')
    plt.show()


def mode_decomposition_energy():
    pass



if __name__ == '__main__':
    # plot_fit_an_fem()
    # plot_rude_way_test()

    # mode_decomposition_form()
    # plot_fit_an_fem_2()
    plot_fit_an_fem_3()