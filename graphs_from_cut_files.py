from matplotlib import pyplot as plt


def read_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        # Assuming each file contains a single line with data in the format "[x, y, z]"
        data = eval(content)
        return data


def data_on_cut_files(main_file_name):
    float_list = []

    for i in range(1, 20):  # Assuming you have up to file_99.txt
        print(i)
        file_name = main_file_name + f'_{i}.txt'
        try:
            data = read_file(path + file_name)
            float_list.extend(data)
        except FileNotFoundError:
            break  # Stop if file not found

    return float_list


if __name__ == "__main__":
    plt.figure(1)

    # plt.title('Energy functional', pad=8)
    # plt.xlabel('time, sec')
    # plt.ylabel('functional')

    plt.title('Black - Beam end coordinate,\nGreen - Point opposite the barrier.', pad=8)
    plt.xlabel('time, sec')
    # plt.ylabel('functional')

    plt.grid()

    # loc_bar = 0.7
    # path = './plots/location_{}_earth_split_en_func_faster/'.format(round(loc_bar, 1))
    loc_bar = 0.9
    path = './plots/location_{}/'.format(round(loc_bar, 1))
    # en_func = data_on_cut_files('en_func')
    en_func = data_on_cut_files('time_disp_end')
    print(len(en_func))
    # time_en_func = data_on_cut_files('time_en_func')
    time_en_func = data_on_cut_files('time_lst')
    # plt.plot(time_en_func, en_func, color='r', label=f'loc = {loc_bar}', linewidth=2)
    plt.plot(time_en_func, en_func, color='k', label='Beam end coordinate', linewidth=1)

    loc_bar = 0.9
    path = './plots/location_{}/'.format(round(loc_bar, 1))
    # en_func = data_on_cut_files('en_func')
    en_func = data_on_cut_files('time_disp')
    print(len(en_func))
    # time_en_func = data_on_cut_files('time_en_func')
    # plt.plot(time_en_func, en_func, color='k', label=f'loc = {loc_bar}', linestyle='--')
    plt.plot(time_en_func, en_func, color='g', label='Point opposite the barrier', linestyle='--')

    # plt.legend()
    plt.show()

