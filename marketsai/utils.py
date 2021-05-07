import numpy as np

dims = [3, 4, 7]
print(dims[len(dims) - 1 :])


def encode(array=[1, 2, 2], dims=[3, 8, 7]):
    assert len(array) == len(dims)
    code = array[-1]
    for i in range(len(dims) - 1):
        code += array[-(i + 2)] * np.product(dims[len(dims) - 1 - i :])

    return code


def decode(code=1, dims=[3, 8, 7]):
    max_code = np.product(dims) - 1
    array = []
    array[2] = code % dims[2]
    array[0] = code // dims[1]
    array = []
    for i in range(len(dims) - 1):
        array += code

    return array


print(encode(array=[1, 2, 3], dims=[3, 3, 4]))
