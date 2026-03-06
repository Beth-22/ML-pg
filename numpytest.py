import numpy as np


# 1. Creating arrays
print("\n1. Creating Arrays")
arr1 = np.array([10, 20, 30, 40, 50])
print("1D Array:", arr1)

arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("2D Array:\n", arr2)

# 2. Special arrayss
print("\n2. Special Arrays")

zeros = np.zeros((3,3))
print("Zeros array:\n", zeros)

ones = np.ones((2,4))
print("Ones array:\n", ones)

range_arr = np.arange(0,10)
print("Range array:", range_arr)

lin = np.linspace(0,1,5)
print("Linspace array:", lin)

# 3. Reshaping
print("\n3. Reshaping")

reshaped = np.arange(12).reshape(3,4)
print("Reshaped array:\n", reshaped)

# 4. Indexing and slicing
print("\n4. Indexing & Slicing")

print("First element:", arr1[0])
print("Slice 1:4:", arr1[1:4])

print("Element from 2D array:", arr2[1,2])

# 5. Mathematical operations
print("\n5. Mathematical Operations")

a = np.array([1,2,3])
b = np.array([4,5,6])

print("Addition:", a + b)
print("Multiplication:", a * b)
print("Square root:", np.sqrt(a))

# 6. Stat
print("\n6. Statistics")

data = np.array([10,20,30,40,50])

print("Mean:", np.mean(data))
print("Sum:", np.sum(data))
print("Max:", np.max(data))
print("Min:", np.min(data))
print("Standard deviation:", np.std(data))

# 7. Random numbers
print("\n7. Random Numbers")

random_arr = np.random.rand(3,3)
print("Random array:\n", random_arr)

rand_int = np.random.randint(1,100,(3,3))
print("Random integers:\n", rand_int)

# 8. Boolean filtering
print("\n8. Boolean Filtering")

nums = np.array([5,10,15,20,25,30])
filtered = nums[nums > 15]
print("Numbers greater than 15:", filtered)

# 9. Matrix multiplication
print("\n9. Matrix Multiplication")

m1 = np.array([
    [1,2],
    [3,4]
])

m2 = np.array([
    [5,6],
    [7,8]
])

result = np.dot(m1,m2)
print("Matrix multiplication result:\n", result)

print("\nProgram Finished")
