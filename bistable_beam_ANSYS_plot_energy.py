import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_set = []
        for line in lines:
            line = line.strip()
            if line.startswith("'"):  # Условие для начала нового набора данных
                if current_set:  # Добавляем предыдущий набор в общий список
                    data.append(current_set)
                current_set = []  # Начинаем новый набор
            else:
                current_set.append(float(line))
        if current_set:
            data.append(current_set)
    data_array = np.array(data)
    return data_array

def calculate_mode_contributions(data_array, num_modes=10):
    n_points = data_array.shape[1]
    x = np.linspace(0, 1, n_points)
    mode_contributions = np.zeros((data_array.shape[0], num_modes))

    for i in range(data_array.shape[0]):
        for k in range(1, num_modes + 1):
            mode = np.sin(k * np.pi * x)
            contribution = np.trapz(data_array[i] * mode, x)
            mode_contributions[i, k-1] = contribution
    return mode_contributions


# ---- reading file with timestamps
def extract_time_column(file_path):
    # Считываем данные из файла, пропуская первую строку (заголовок)
    data = np.loadtxt(file_path, skiprows=1, usecols=1)
    return data

file_path = r'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L\timesteps_sin_1e3damp.txt'
time_array = extract_time_column(file_path)
# time_array = time_array[::20]
def extract_time_steps(time_array, step_inc=10):
    steps1 = 3680
    steps2 = 44005
    # steps1 = 729
    # steps2 = 24005
    # step_inc = 10
    steps_extract = list(range(steps1))[::step_inc] + list(range(steps1, steps2+steps1))[::step_inc]
    return time_array[steps_extract]

time_array = extract_time_steps(time_array, step_inc=20)
print(f'time_array shape = {time_array.shape}')
print()
# ---------------------------------

# Чтение данных из файла и преобразование в numpy массив
# file_path = r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\plots\ansys_energy\test_10\potential_energy_apdl.txt'
file_path = r'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L\potential_energy_apdl_sin_1e3damp.txt'
data_array_potential = read_data(file_path)

file_path = r'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L\kinetic_energy_apdl_sin_1e3damp.txt'
data_array_kinetic = read_data(file_path)

print(data_array_kinetic.shape)  # Вывод формы массива
print(data_array_potential.shape)  # Вывод формы массива


data_array = data_array_kinetic + data_array_potential
# data_array = data_array_potential

# Число точек
n_points = data_array.shape[1]
print(f'n_points = {n_points}')
x = np.linspace(0, 1, n_points)

# Рассчитываем вклады по первым 10 собственным формам
num_modes = 20
mode_contributions = calculate_mode_contributions(data_array, num_modes)


def plot_dinamic_graphs():
    # Настройка фигуры для анимации
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    # Увеличиваем расстояние между подграфиками
    plt.subplots_adjust(hspace=0.3)

    line, = ax1.plot(x, data_array[0], 'b-')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(np.min(data_array), np.max(data_array))
    # ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    bar = ax2.bar(np.arange(1, num_modes + 1), mode_contributions[0])
    # Устанавливаем нумерацию всех целых значений на оси x
    ax2.set_xticks(np.arange(1, num_modes + 1))

    ax2.set_ylim(0, np.max(mode_contributions))
    # ax2.set_ylim(0, 10)
    ax2.set_xlabel('Mode number')
    ax2.set_ylabel('Energy Contribution')

    # Full energy
    full_en_lst = []
    line_en, = ax3.plot([], full_en_lst, 'b-')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Full energy')
    # ax3.set_title("Full energy")

    # plt.pause(15.0)

    # Цикл для обновления графика
    for i in range(len(data_array)):
        line.set_ydata(data_array[i])

        full_en_lst.append(np.sum(data_array[i]))
        line_en.set_xdata(time_array[:i+1])
        line_en.set_ydata(full_en_lst)
        ax3.set_xlim(0, time_array[i])
        ax3.set_ylim(np.min(full_en_lst), np.max(full_en_lst) * 1.1)

        for j, b in enumerate(bar):
            b.set_height(mode_contributions[i, j])
        ax1.set_title(f"Time step: {i + 1}, Time = {time_array[i]}")
        ax2.set_title(f"Time = {time_array[i]}")
        ax2.set_ylim(0, np.max(mode_contributions[i] * 1.2))

        plt.draw()
        plt.pause(0.1)

    plt.show()


def plot_static_graph():
    plt.figure()
    lim_t = 0.397 # -------------------------------------------------------

    ii_lim = 0  # after this index, free vibration beam
    for ii in range(len(data_array)):
        if time_array[ii] >= lim_t:
            ii_lim = ii
            break

    ii_E0 = 0  # full energy
    for ii in range(len(data_array)):
        if time_array[ii] >= 0.205:
            ii_E0 = ii
            break

    ii_tau = 0  # tau
    value_E0 = np.sum(data_array[ii_E0])
    print(f'value_E0 = {value_E0}')
    for ii in range(ii_lim, len(data_array)):
        if ((np.sum(data_array[ii_lim]) - value_E0) / (np.sum(data_array[ii]) - value_E0)) >= np.exp(1):
            ii_tau = ii
            break

    # ii_lim_plot = ii_lim
    ii_lim_plot = 0
    plt.plot(time_array[ii_lim_plot:], [np.sum(data_array[i]) for i in range(ii_lim_plot, len(data_array))], linewidth=2)
    plt.plot([time_array[ii_lim_plot], time_array[-1]], [np.sum(data_array[ii_tau]), np.sum(data_array[ii_tau])],
             'r--', linewidth=1.5, label=f'tau = {round((time_array[ii_tau] - time_array[ii_lim]), 6)}')
    plt.plot([time_array[ii_lim], time_array[ii_lim]], [np.sum(data_array, axis=1).min(), np.sum(data_array, axis=1).max()], '--r',
             linewidth=1.0)
    plt.plot([time_array[ii_lim_plot], time_array[ii_lim_plot]], [np.sum(data_array, axis=1).min(), np.sum(data_array, axis=1).max()], '--k',
             linewidth=1.5)
    plt.plot([time_array[ii_lim_plot], time_array[-1]], [np.sum(data_array, axis=1).min(), np.sum(data_array, axis=1).min()], '--k',
             linewidth=1.5)
    plt.legend()
    plt.title('∆L = 0.14,    ξ=0.0001')


    # # ----------------------------------------------------------
    # file_path = r'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L\timesteps_10.txt'
    # time_array_2 = extract_time_column(file_path)
    # time_array_2 = extract_time_steps(time_array_2, step_inc=50)
    #
    # file_path = r'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L\potential_energy_apdl_10_2.txt'
    # data_array_potential = read_data(file_path)
    #
    # file_path = r'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L\kinetic_energy_apdl_10_2.txt'
    # data_array_kinetic = read_data(file_path)
    #
    # print(data_array_kinetic.shape)  # Вывод формы массива
    # print(data_array_potential.shape)  # Вывод формы массива
    #
    # data_array_2 = data_array_kinetic + data_array_potential
    # print(time_array_2.shape)
    # print(data_array_2.shape)
    # plt.plot(time_array_2[:], [np.sum(data_array_2[i]) for i in range(len(data_array_2))], '--r',
    #          linewidth=2)
    # # ----------------------------------------------------------


    plt.show()


# plot_static_graph()
plot_dinamic_graphs()
