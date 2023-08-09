from matplotlib import pyplot as plt
import numpy as np


def main_chart_first_step(L, MaxNode, dis_i, start_def, time_lst, time_disp, time_disp_end, time_force):
    fig, axs = plt.subplots(3)
    plt.subplots_adjust(wspace=0.4, hspace=0.7)
    fig.suptitle('ГРАФИКИ')
    axs[0].set_title('Форма балки')
    axs[1].set_title('Временная з-ть узла балки. \n Черная - конец, зеленая - середина.', fontsize=10)
    axs[2].set_title('VI force')
    axs[0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r', linewidth=1)
    scale = start_def[-2][0]  # Масштаб графика формы балки
    axs[0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей
    axs[1].plot(time_lst, time_disp, 'g', linewidth=1)
    axs[1].plot(time_lst, time_disp_end, 'k', linewidth=1)
    axs[2].plot(time_lst, time_force, 'k', linewidth=1)
    plt.pause(2)
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()


def main_chart(L, MaxNode, dis_i, start_def, time_lst, time_disp, time_disp_end, time_force):
    fig, axs = plt.subplots(3)
    plt.subplots_adjust(wspace=0.4, hspace=0.7)
    axs[0].set_title('Форма балки')
    axs[1].set_title('Временная з-ть узла балки. \n Черная - конец, зеленая - середина.', fontsize=10)
    axs[2].set_title('VI force')
    fig.suptitle('Время = ' + str('%.2f' % t)
                 + ' c = ' + str('%.2f' % (t * 1e3)) + ' мс = ' + str('%.2f' % (t * 1e6)) + ' мкс')
    axs[0].plot(np.linspace(0, L, num=MaxNode), [dis_i[i * 2, 0] for i in range(MaxNode)], 'r',
                linewidth=1)  # Положение балки
    axs[0].plot([L * loc_bar], [dis_i1[point_bar, 0]], 'go', markersize=4)  # Жирная точка середина балки
    axs[0].plot([L], [dis_i1[-2, 0]], 'ko', markersize=4)  # Жирная точка конца балки
    axs[0].plot([L * loc_bar], [-delta], 'b^', markersize=7)  # Местоположение барьера
    # scale = max(abs(min(time_disp_end)), abs(max(time_disp_end)), delta * 2)  # Масштаб графика формы балки
    scale = start_def[-2][0]  # Масштаб графика формы балки
    axs[0].axis([0, L * 1.1, -scale * 1.2, scale * 1.2])  # устанавливаем диапозон осей

    axs[1].plot(time_lst, time_disp, color='g', linewidth=1)  # временная з-ть середины балки
    axs[1].plot(time_lst, time_disp_end, color='k', linewidth=1)  # временная з-ть конца балки
    axs[1].plot(time_lst, [-delta] * len(time_lst), 'r--', linewidth=1)  # барьер

    axs[2].plot(time_lst, time_force, 'k', linewidth=1)  # временная з-ть силы VI
    plt.pause(0.0001)
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()