import numba as nb
import numpy as np


@nb.njit(nopython=True)
def dtw_edge(arr, x, y):
    n = len(x)
    m = len(y)
    arr = np.ones((n, m), dtype=np.float64)
    arr[0, 0] = np.abs(x[0] - y[0])
    for i in range(1, n):
        arr[i, 0] = arr[i - 1, 0] + np.abs(x[i] - y[0])
    for j in range(1, m):
        arr[0, j] = arr[0, j - 1] + np.abs(x[0] - y[j])
    return arr, n, m


def cost(arr, i, j):
    cost = np.abs(x[i] - y[j])
    return cost + min(arr[i - 1, j], arr[i, j - 1], arr[i - 1, j - 1])


@nb.njit("i8[:,:](f8[:,:])", nopython=True)
def backtracking(arr):
    n, m = arr.shape
    i = n - 1
    j = m - 1
    step = 0
    path = np.zeros((n + m, 2), dtype=np.int64)
    path[step, 0] = i
    path[step, 1] = i
    while i > 0 and j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            if arr[i - 1, j] < arr[i, j - 1] and arr[i - 1, j] < arr[i - 1, j - 1]:
                i = i - 1
            elif arr[i, j - 1] < arr[i - 1, j] and arr[i, j - 1] < arr[i - 1, j - 1]:
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        step += 1
        path[step, 0] = i
        path[step, 1] = j
    return path[: (step + 1)]
