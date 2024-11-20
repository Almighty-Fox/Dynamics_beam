import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# Чтение данных из файла
data = pd.read_csv(r'C:\Users\evgenii\PycharmProjects\Dynamics_beam\plots\ansys_energy\test_7\export_energy_ansys.txt', delimiter='\t')

# Извлечение данных
time = data['Time [s]']
kinetic_energy = data['Energy Probe 3 (Kinetic) [J]']
strain_energy = data['Energy Probe 3 (Strain) [J]']
total_energy = kinetic_energy + strain_energy  # Сумма кинетической и деформационной энергии

# Инициализация графика
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Настройки первого графика
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Energy [J]')
line1, = ax1.plot([], [], label='Kinetic Energy', color='blue')
line2, = ax1.plot([], [], label='Strain Energy', color='red')
ax1.legend()

# Настройки второго графика
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Total Energy [J]')
line3, = ax2.plot([], [], label='Total Energy', color='green')
ax2.legend()


# Функция инициализации
def init():
    ax1.set_xlim(min(time), max(time))
    ax1.set_ylim(min(min(kinetic_energy), min(strain_energy)), max(max(kinetic_energy), max(strain_energy)))
    line1.set_data([], [])
    line2.set_data([], [])

    ax2.set_xlim(min(time), max(time))
    ax2.set_ylim(min(total_energy), max(total_energy))
    line3.set_data([], [])

    return line1, line2, line3


# Функция анимации
def update(frame):
    # Обновление данных для первого графика
    line1.set_data(time[:frame], kinetic_energy[:frame])
    line2.set_data(time[:frame], strain_energy[:frame])
    ax1.set_title(f'Dynamic Energy Plot: Time = {time[frame]:.4f} s')

    # Обновление данных для второго графика
    line3.set_data(time[:frame], total_energy[:frame])
    ax2.set_title(f'Total Energy Plot: Time = {time[frame]:.4f} s')

    return line1, line2, line3


# Настройка анимации с уменьшением интервала
ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=False, repeat=False, interval=10)

plt.tight_layout()
plt.show()
