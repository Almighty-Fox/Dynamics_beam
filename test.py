import numpy as np

# Create a matrix
matrix = np.array([[1, 2],
                   [3, 4]])

# Create a column vector
column_vector = np.array([[2],
                          [3]])

# Scalar multiply the matrix by the column vector
result = matrix * column_vector

result2 = np.matmul(matrix, column_vector)

# Print the result
print(result)
print(result2)
