from matplotlib import pyplot as plt
import numpy as np


def read_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        # Assuming each file contains a single line with data in the format "[x, y, z]"
        data = eval(content)
        return data


def data_on_cut_files(path, main_file_name):
    float_list = []

    for i in range(1, 23):  # Assuming you have up to file_99.txt
        print(i)
        file_name = main_file_name + f'_{i}.txt'
        try:
            data = read_file(path + file_name)
            float_list.extend(data)
        except FileNotFoundError:
            break  # Stop if file not found

    return float_list


def plot_data_on_cut_files():
    plt.figure(1)

    # plt.title('Energy functional', pad=8)
    # plt.xlabel('time, sec')
    # plt.ylabel('functional')

    # plt.title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', pad=8)
    # plt.title('Full energy')
    # plt.xlabel('time, sec')
    # plt.ylabel('functional')

    # plt.grid()

    # loc_bar = 0.9
    # path = './plots/earthquake_damping/location_0.0_earthquake/'.format(round(loc_bar, 1))
    # # en_func = data_on_cut_files('en_func')
    # en_func = data_on_cut_files(path, 'full_en_lst')
    # print(len(en_func))
    # # time_en_func = data_on_cut_files('time_en_func')
    # time_en_func = data_on_cut_files(path, 'time_lst')
    # # plt.plot(time_en_func, en_func, color='r', label=f'loc = {loc_bar}', linewidth=2)
    # plt.plot(time_en_func, en_func, color='g', label='VI', linewidth=1)  # label='Beam end coordinate'

    # loc_bar = 0.9
    # path = './plots/location_0.0_right/'.format(round(loc_bar, 1))
    # # en_func = data_on_cut_files('en_func')
    # en_func_1 = data_on_cut_files(path, 'time_force')
    # # work_earth = data_on_cut_files(path, 'earthquake_en_lst')
    # print(len(en_func_1))
    # time_en_func_1 = data_on_cut_files(path, 'time_lst')
    # # plt.plot(time_en_func, en_func, color='k', label=f'loc = {loc_bar}', linestyle='--')
    # plt.figure(1)
    # plt.plot(time_en_func_1, np.array(work_earth), color='g', label='work_earth', linestyle='-',
    #          linewidth=2)
    # plt.plot(time_en_func_1, np.array(en_func_1), color='r', label='full', linestyle='-',
    #          linewidth=2)
    # plt.plot(time_en_func_1, np.array(work_earth) - np.array(en_func_1), color='k', label='diff', linestyle='-', linewidth=2)  # label='Point opposite the barrier'  # linestyle='--'
    #
    # plt.title('Energy')
    # plt.xlabel('Time, sec')
    # plt.grid()
    # plt.legend()
    # # plt.show()

    loc_bar = 0.8
    path = './plots/calc_stress/location_0.8_stress_kc_100/'.format(round(loc_bar, 1))
    x_axis = data_on_cut_files(path, 'time_lst')
    y_axis = data_on_cut_files(path, 'earthquake_en_lst')
    plt.plot(x_axis, y_axis, color='g', label='work earth', linestyle='-', linewidth=2)  # label=f'loc = {loc_bar}'

    y_axis_2 = data_on_cut_files(path, 'full_en_lst')
    plt.plot(x_axis, y_axis_2, color='r', label='full', linestyle='-', linewidth=2)  # label=f'loc = {loc_bar}'

    plt.plot(x_axis, np.array(y_axis) - np.array(y_axis_2), color='k', label='diff', linestyle='-', linewidth=2)  # label=f'loc = {loc_bar}'

    plt.title('Energy, J, kc=100')



    # plt.title('Full Energy')
    plt.xlabel('Time, sec')
    plt.grid()
    plt.legend()
    plt.show()

def plot_data_time_relaxation():
    path = r'./plots/data_before_time_relaxation/Impuls_loading/impulse_period_1e_2/damp_0.0015/'
    loc_bar_lst = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    for loc_bar in loc_bar_lst:
        path2 = 'location_{}/'.format(round(loc_bar, 1))
        # file_name = 'en_func_2.txt'
        file_name = 'full_en_lst.txt'
        y_values = read_file(path + path2 + file_name)
        file_name = 'time_lst.txt'
        time_lst = read_file(path + path2 + file_name)

        plt.figure(2)
        plt.plot(time_lst[1:], y_values[1:], label='loc={0}, tau={1}'.format(loc_bar, round(time_lst[-1], 3)))

    plt.legend()
    plt.xlabel('Time, sec')
    # plt.ylabel('Functional')
    plt.ylabel('Full energy')
    plt.grid()
    plt.show()


def plot_fit_an_fem():
    path = r'./plots/fit_an_fem/analytics_start_deformation_loc_0_8_dynamic_step/'
    # file_name = 'Analytics_write_disp_end.txt'
    file_name = 'Analytics_write_disp_barrier.txt'
    y_values = read_file(path + file_name)
    file_name = 'Analytics_write_time.txt'
    time_lst = read_file(path + file_name)

    plt.figure(1)
    plt.plot(time_lst, y_values, 'k', label='analytics')

    path = r'./plots/fit_an_fem/location_0.8_better/'
    # file_name = 'time_disp_end_last.txt'
    file_name = 'time_disp_1.txt'
    y_values = read_file(path + file_name)
    file_name = 'time_lst_1.txt'
    time_lst = read_file(path + file_name)

    plt.plot(time_lst, y_values, 'r--', label='FEM')

    plt.legend()
    plt.xlabel('Time, sec')
    # plt.ylabel('Functional')
    plt.ylabel('Beam end coordinate, m')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # plot_data_on_cut_files()
    plot_fit_an_fem()
    # plot_data_time_relaxation()