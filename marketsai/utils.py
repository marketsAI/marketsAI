import numpy as np

dims = [3, 4, 7]
print(dims[len(dims) - 1 :])


def index(array=[1, 2, 2], dims=[3, 8, 7]):
    index = array[-1]
    for i in range(len(dims) - 1):
        index += array[-(i + 2)] * np.product(dims[len(dims) - 1 - i :])

    return index


print(index(array=[1, 2, 3], dims=[3, 3, 4]))
