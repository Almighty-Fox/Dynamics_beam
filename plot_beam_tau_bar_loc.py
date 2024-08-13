import matplotlib.pyplot as plt
import numpy as np
import os

# Строит график х-го времени затухания для балки, объединяя множество текстовых файлов в зависимости от положения барьера

# Initialize arrays to hold the location parameters and the last values from each file
location_params = []
last_values = []

# Loop through the location parameters from 0.99 down to 0.41
for i in np.arange(0.99, 0.43, -0.01):
    # folder_name = f"location_{i:.2f}"
    folder_name = f"./plots/data_before_time_relaxation_small_step_damp_0_0015/location_{round(i,2)}/"
    file_path = folder_name + "time_lst_last.txt"
    print(folder_name)

    # Read the file
    with open(file_path, 'r') as file:
        contents = file.read()

        # Convert the string from the file into a list
        time_list = eval(contents)
        # Append the last value of the list to the last_values array
        last_values.append(time_list[-1])
        # Append the location parameter to the location_params array
        location_params.append(i)


for i in range(len(location_params)):
    print(round(location_params[i], 2), end=' = ')
    print(last_values[i])
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(location_params, last_values, marker='o', linestyle='-')
plt.title('Damping = 0.0015')
plt.xlabel('Barrier location')
plt.ylabel('Characteristic damping time')
plt.grid(True)
plt.show()
