import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Загрузка CSV с разделителем ';'
file_path = 'final_positions_4_c_100.csv'  # Убедитесь, что путь корректен
df = pd.read_csv(file_path, sep=';')

# Функция для назначения цвета в зависимости от y1 и y2
def assign_color(y1, y2):
    if y1 > 0 and y2 > 0:
        return 'blue'
    elif y1 < 0 and y2 > 0:
        return 'yellow'
    elif y1 > 0 and y2 < 0:
        return 'green'
    elif y1 < 0 and y2 < 0:
        return 'red'
    else:
        return 'black'  # если один из них точно 0

# Применение функции к каждой строке
df['color'] = df.apply(lambda row: assign_color(row['y1_final [m]'], row['y2_final [m]']), axis=1)

# Построение графика
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['v1_max [N]'], df['v2_max [N]'], c=df['color'])

# Настройки осей и заголовка
plt.xlabel('v1_init')
plt.ylabel('v2_init')
plt.title('Final states: classification by y1_final and y2_final')
plt.grid(True)

# Добавление легенды вручную
legend_elements = [
    Patch(facecolor='blue', label='y1 > 0, y2 > 0'),
    Patch(facecolor='yellow', label='y1 < 0, y2 > 0'),
    Patch(facecolor='green', label='y1 > 0, y2 < 0'),
    Patch(facecolor='red', label='y1 < 0, y2 < 0'),
    Patch(facecolor='black', label='one of the values = 0'),
]
plt.legend(handles=legend_elements, title='Final state domains', loc='upper right', framealpha=0.9)

plt.show()
