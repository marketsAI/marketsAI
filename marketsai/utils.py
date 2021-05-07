import numpy as np


def encode(array, dims):
    assert len(array) == len(dims)
    for i in range(len(dims)):
        assert array[i] < dims[i]
        assert type(dims[i]) == int and dims[i] > 0

    code = array[-1]
    for i in range(len(dims) - 1):
        code += array[-(i + 2)] * np.product(dims[len(dims) - 1 - i :])

    return code


def decode(code, dims):
    dims_total = np.product(dims)
    assert code < dims_total
    array = [0 for i in range(len(dims))]

    array[-1] = code % dims[-1]
    array[0] = code // np.product(dims[1:])
    for i in range(1, len(dims) - 1):
        array[i] = (
            code - np.dot(array[:i], [np.product(dims[j + 1 :]) for j in range(i)])
        ) // np.product(dims[i + 1 :])

    return array
