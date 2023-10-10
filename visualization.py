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
an_force = read_file('Analytics_write_VI_force.txt')
an_time = read_file('Analytics_write_time.txt')

df_disp = pd.DataFrame({'time': lst_time, 'VI_force': lst_VI_force})
df_disp['Rolling_Avg'] = df_disp['VI_force'].rolling(10000, min_periods=1, center=True).mean()

# print(df_disp)
# plt.plot(df_disp.time, df_disp.VI_force, label='DataFrame 1')
plt.plot(df_disp.time, df_disp.Rolling_Avg, color='r', label='DataFrame 2')

plt.plot(an_time, an_force, color='k', label='DataFrame 3')

plt.grid()
plt.show()