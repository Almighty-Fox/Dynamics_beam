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


# ---- reading file with timestamps
def extract_time_column(file_path):
    # Считываем данные из файла, пропуская первую строку (заголовок)
    data = np.loadtxt(file_path, skiprows=1, usecols=1)
    return data

file_path = r'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L\timesteps_sin_1e3damp.txt'
time_array = extract_time_column(file_path)
# time_array = time_array[::20]


def extract_time_steps(time_array, steps1, steps2, step_inc):
    steps_extract = list(range(steps1))[::step_inc] + list(range(steps1, steps2+steps1))[::step_inc]

    return time_array[steps_extract]


# time_array = extract_time_steps(time_array, step_inc=20)
# print(f'time_array shape = {time_array.shape}')
# print()
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

def extract_data_from_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Извлечение значений
    deltaL = float(lines[0].split('=')[1].strip())  # Первая строчка: deltaL
    t_max = float(lines[2].split('(')[1].split(',')[1].split(')')[0].strip())  # Третья строчка: t_max
    t_free = float(lines[10].split('=')[0].split('->')[1].strip())  # Одиннадцатая строчка: t_free
    f_max = float(lines[11].split('=')[1].strip())  # Двенадцатая строчка: f_max
    step1 = int(lines[13].split(',')[-2].strip())  # Четырнадцатая строчка: step1
    step2 = int(lines[14].split(',')[-2].strip())  # Пятнадцатая строчка: step2
    f_snap = float(lines[9].split('=')[-1].strip())  # Десятая строчка: f_snap

    return deltaL, t_max, t_free, f_max, step1, step2, f_snap


def loop_all_files_plot_tau_deltaL():
    t_tau_lst = []
    deltaL_lst = []
    f_snap_lst = []
    for i in range(1, 22):
        deltaL, t_max, t_free, f_max, steps1, steps2, f_snap = extract_data_from_log(rf'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L_new\log_{i}.txt')

        print(f"deltaL = {deltaL}")
        print(f"t_max = {t_max}")
        print(f"t_free = {t_free}")
        print(f"f_max = {f_max}")
        print(f"step1 = {steps1}")
        print(f"step2 = {steps2}")
        print(f"f_snap = {f_snap}")

        f_snap_lst.append(f_snap)

        time_array = extract_time_column(rf'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L_new\timesteps_{i}.txt')
        time_array = extract_time_steps(time_array, steps1, steps2, step_inc=5)

        data_array_potential = read_data(rf'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L_new\potential_energy_apdl_{i}.txt')
        data_array_kinetic = read_data(rf'C:\Technion\ANSYS_projects\ansys_python_energy\var_delta_L_new\kinetic_energy_apdl_{i}.txt')

        data_array = data_array_kinetic + data_array_potential

        ii_lim = 0  # after this index, free vibration beam
        for ii in range(len(data_array)):
            if time_array[ii] >= t_free:
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

        t_tau = time_array[ii_tau] - time_array[ii_lim]
        t_tau_lst.append(t_tau)
        deltaL_lst.append(deltaL)

    print(deltaL_lst)
    print(t_tau_lst)
    print(f_snap_lst)
    plt.figure()
    plt.plot(deltaL_lst, t_tau_lst, marker='o', linestyle='-', color='b')
    plt.grid()
    plt.legend()

    plt.show()


loop_all_files_plot_tau_deltaL()
