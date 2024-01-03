from matplotlib import pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------
# находим в папке все файлы с расширением ".AT2" и записываем все имена в лист
def find_files_with_extension(folder_path, extension):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

folder_path = r'C:\Technion\Beam\PEERNGARecords_Unscaled'
extension = '.AT2'

file_list = find_files_with_extension(folder_path, extension)
print(file_list)

# Extracting only file names without the path and extension
main_file_names = [os.path.splitext(os.path.basename(file))[0] for file in file_list]

print(main_file_names)
# --------------------------------------------------------


def open_file_and_plot(file_name, need_plot=True):
    # читаем файл в одну переменную
    # path = r'F:\Evgenii\Technion\PEERNGARecords_Unscaled'
    # file_name = 'RSN1101_KOBE_AMA-UP.AT2'
    # with open(path + '\\' + file_name, 'r') as cur_file:
    with open(file_name, 'r') as cur_file:
        body = cur_file.readlines()

    # выделяем из файла значения количества точек и шага по времени
    print(body[:4])
    param = body[3].split(', ')
    points = int(param[0].split('=')[1])
    time_step = float((param[1].split('=')[1]).strip().split(' ')[0])

    # отбрасываем шапку тела файла
    body_points = list(map(lambda x: x.strip(), body[4:]))
    all_data = []
    for line in body_points:
        line_sep = list(map(float, line.split('  ')))
        all_data += line_sep

    print('Количетво точек совпадает с указанным в файле:', end=' ')
    print(len(all_data) == points)
    print('Количество значений: {}'.format(points))
    print('Time step: {}'.format(time_step))

    max_value = max(abs(np.array(all_data)))

    time_lst = np.linspace(0, (points - 1) * time_step, len(all_data))

    if need_plot:
        plt.plot(time_lst, all_data)
        plt.grid()
        plt.xlabel('Time, sec')
        plt.ylabel('Acceleration, g')
        plt.title('Kobe earthquake')
        # plt.show()

    return max_value, all_data, time_lst, time_step

dict = {}
for file_name in file_list:
    max_value, all_data, time_lst, time_step = open_file_and_plot(file_name, need_plot=False)
    dict[file_name] = max_value

file_name_max_value = max(dict, key=dict.get)
print('\nFile with max values')
print(file_name_max_value)
max_value, all_data, time_lst, time_step = open_file_and_plot(file_name_max_value)

# ----------------------------------------------------------------
# определяем несущую частоту землетрясения
time_start = 7.38
time_end = 13

def cut_list(lst, value):
    id_cut = 0
    for i in range(len(lst)):
        if lst[i] > value:
            id_cut = i
            break
    return id_cut

i_start = cut_list(time_lst, time_start)
i_end = cut_list(time_lst, time_end)

counter = 0
for i in range(i_start, i_end):
    if all_data[i+1] * all_data[i] < 0:
        counter += 1

print('counter = ', counter)
basic_freq = (counter / 2) / (time_lst[i_end] - time_lst[i_start])
print('nu = ', basic_freq)
basic_omega_2 = (2 * np.pi * basic_freq) ** 2
print('Omega 2 = ', basic_omega_2)
print('Omega = ', np.sqrt(basic_omega_2))

# ----------------------------------------------------------------
# строим нормализированную акселерограмму. Строим интересующий нас интервал.
# normal_fr = np.sqrt(2688.5 / 244.98707749566427)
normal_fr = 1
time_lst_norm = time_lst / normal_fr

time_start = 4.36
time_end = 23.71783

def cut_list(lst, value):
    id_cut = 0
    for i in range(len(lst)):
        if lst[i] > value:
            id_cut = i
            break
    return id_cut


i_start = cut_list(time_lst_norm, time_start)
i_end = cut_list(time_lst_norm, time_end)

# scale_acc = max(all_data[i_start:i_end])
scale_acc_pos = max(all_data[:])
scale_acc_neg = min(all_data[:])

plt.figure()
plt.plot(time_lst_norm, all_data)
plt.plot([time_lst_norm[i_start], time_lst_norm[i_start]], [scale_acc_neg, scale_acc_pos], 'r--')
plt.plot([time_lst_norm[i_end], time_lst_norm[i_end]], [scale_acc_neg, scale_acc_pos], 'r--')
plt.grid()
plt.xlabel('Time, sec')
plt.ylabel('Acceleration, g')
plt.title('Kobe earthquake')

plt.show()



