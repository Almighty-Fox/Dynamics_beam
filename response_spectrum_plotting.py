import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from create_global_matrix import open_file_earthquake_data


def difur(yd, t):
    global w
    F = [yd[1], -2*0.0*yd[1] - w**2*yd[0] - acceleration(t)]
    return F

def acceleration(tt):
    i_start = cut_list(earthquake_time_lst, tt)

    return earthquake_all_data[i_start - 1]

def cut_list(lst, value):
    id_cut = 0
    for i in range(len(lst)):
        if lst[i] > value:
            id_cut = i
            break

    return id_cut

def proizvodnay(x, y):
    proiz_y = np.zeros_like(x)
    for i in range(len(x)-1):
        proiz_y[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
    proiz_y[-1] = proiz_y[-2]
    return proiz_y


def spot():
    # It should return x and y as arrays.
    earthquake_time_step, earthquake_all_data = open_file_earthquake_data()
    earthquake_time_lst = np.linspace(0, (len(earthquake_all_data) - 1) * earthquake_time_step, len(earthquake_all_data))

    return earthquake_time_lst, earthquake_all_data


earthquake_time_lst, earthquake_all_data = spot()

max_acc = []
# w_lst_start, w_lst_end, w_lst_step = 0, 301, 1
w_lst_start, w_lst_end, w_lst_step = 6, 41, 0.5
# w_lst = range(w_lst_start, w_lst_end, w_lst_step)
w_lst = np.arange(w_lst_start, w_lst_end, w_lst_step)
for w in w_lst:
    print(w)
    # time_lst = np.linspace(0, earthquake_time_lst[-1], 1000)
    Y = odeint(difur, [0, 0], earthquake_time_lst)
    acc = proizvodnay(earthquake_time_lst, Y[:, 1])
    aksel = np.array([acceleration(t) for t in earthquake_time_lst])
    max_acc.append(max(abs(acc + aksel)))

file_name = 'write_response_spectrum.txt'
with open(r'./plots/' + file_name, 'w') as cur_file:
    cur_file.write(' '.join(list(map(str, [w_lst_start, w_lst_end, w_lst_step]))))
    cur_file.write('\n')
    cur_file.write(str(max_acc))


plt.figure()
plt.grid(True)
plt.plot(w_lst, max_acc, '-ok')
plt.title('Отклик')
plt.xlabel('Угловая частота, 1/с')
plt.ylabel('Ускорение, м/c^2')
plt.show()
