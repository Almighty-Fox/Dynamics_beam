import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_file(file_name):
    with open(r'./plots/' + file_name, 'r') as cur_file:
        body_file = cur_file.readlines()

    str_body_file = body_file[0].split(', ')
    str_body_file[0] = str_body_file[0][1:]
    str_body_file[-1] = str_body_file[-1][:-1]

    lst_body_file = list(map(float, str_body_file))

    return lst_body_file


lst_VI_force = read_file('write_VI_force.txt')
lst_time = read_file('write_time.txt')
lst_disp = read_file('write_disp.txt')
an_force = read_file('Analytics_write_VI_force.txt')
an_time = read_file('Analytics_write_time.txt')
an_disp = read_file('Analytics_write_disp.txt')

# # ------------------------------------
# # отбрасываем конец, если он мешает
# fem_id_cut = len(lst_time)
# for i in range(len(lst_time)):
#     if lst_time[i] > 0.1:
#         fem_id_cut = i
#         break
# lst_time = lst_time[:fem_id_cut]
# lst_disp = lst_disp[:fem_id_cut]
# lst_VI_force = lst_VI_force[:fem_id_cut]
#
# an_id_cut = len(an_time)
# for i in range(len(an_time)):
#     if an_time[i] > 0.1:
#         an_id_cut = i
#         break
# an_time = an_time[:an_id_cut]
# an_disp = an_disp[:an_id_cut]
# an_force = an_force[:an_id_cut]
# # ------------------------------------

df_disp = pd.DataFrame({'time': lst_time, 'VI_force': lst_VI_force})
window_size = round(df_disp.shape[0] / 100)
# df_disp['Rolling_Avg'] = df_disp['VI_force'].rolling(window_size, min_periods=1, center=True).mean()
df_disp['Rolling_Avg'] = df_disp['VI_force'].rolling(window_size, center=True).mean()

# print(df_disp)
plt.figure(1)
plt.title('VI force')
plt.plot(df_disp.time, df_disp.VI_force, color='b', label='FEM')
plt.plot(df_disp.time, df_disp.Rolling_Avg, color='r', label='Rolling FEM')
# plt.plot(an_time, an_force, color='k', label='Analytics')
plt.grid()
plt.xlabel('Time')
plt.ylabel('VI force')
plt.legend()

plt.figure(2)
plt.title('Beam end displacement')
plt.plot(lst_time, lst_disp, color='r', label='FEM')
plt.plot(an_time, an_disp, color='k', label='Analytics')
plt.grid()
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()

plt.show()