import numpy as np
import numba as nb
import os
import sys
import matplotlib.pyplot as plt
import timeit

library = os.path.abspath("../realtime_operator")
if os.path.exists(library) and library not in sys.path:
    sys.path.insert(0, library)

from realtime_operator.single_operator import (
    get_inhomogeneous_time_seris,
    create_random_index,
    get_sinusoid,
    get_white_noise,
    ema,
    outlier,
    limit,
)

SAMPLE_SIZE = 1000


def plot_timeseries(t, x, y=None, title=""):
    _, ax1 = plt.subplots()

    ax1.plot(t, x, "b-")
    ax1.set_xlabel("utc second")
    ax1.set_ylabel("raw", color="b")
    ax1.tick_params("y", colors="b")

    if y is not None:
        ax2 = ax1.twinx()
        ax2.plot(t, y, "r-")
        ax2.set_ylabel("calculated", color="r")
        ax2.tick_params("y", colors="r")

    plt.title(title)
    plt.show()
    plt.pause(0.001)


if __name__ == "__main__":

    t = get_inhomogeneous_time_seris(SAMPLE_SIZE)
    t, z = get_sinusoid(t, 100, 0, 2)
    plot_timeseries(t, z, title="sinusoid")

    z += get_white_noise(SAMPLE_SIZE, 0, 10)
    plot_timeseries(t, z, title="sinusoid + white noise")

    # create vectorized function
    # @nb.jit(nopython=True)
    def ema_vector(t, z, tau=30):
        zn = np.zeros(t.size, dtype=float)
        buffer = np.zeros(3, dtype=float)
        inter = 0
        for i in range(1000):
            _t, _z = ema(tau, inter, buffer, t[i], z[i])
            zn[i] = _z
        return t, zn

    t, zn = ema_vector(t, z, 30)
    plot_timeseries(t, z, zn, title="ema tau=30 seconds")

    zo = z.copy()
    index = create_random_index(10, SAMPLE_SIZE)
    zo[index] = zo[index] + 1000
    plot_timeseries(t, zo, title="outlier")

    t, zn = ema_vector(t, zo, 30)
    plot_timeseries(t, zo, zn, title="outlier: ema tau=30 seconds")

    # create vectorized function
    @nb.jit(nopython=True)
    def outlier_vector(t, z, tau=30):
        zn = np.zeros(t.size, dtype=float)
        buffer = np.zeros(3, dtype=float)
        for i in range(t.size):
            _, _z = outlier(buffer, t[i], z[i], 100, 2)
            zn[i] = _z
        return t, zn

    t, zz = outlier_vector(t, zo)
    plot_timeseries(t, zo, zz, title="outlier: ema tau=30 seconds")

    # chain operators
    @nb.jit(nopython=True)
    def chain_vector(t, z, tau=30):
        zn = np.zeros(t.size, dtype=float)
        buffer = np.zeros(100, dtype=float)
        n = 10
        inter = 0
        zp = z[0]
        for i in range(t.size):
            _, _z = outlier(buffer, t[i], z[i], 100, 2)
            _, _z = ema(tau, inter, buffer[3:], t[i], _z)
            zn[i] = _z
        return t, zn

    t, zc = chain_vector(t, zo)
    plot_timeseries(t, zo, zc, title="outlier: ema tau=30 seconds")

    # chain operators, denoising + limits
    @nb.jit(nopython=True)
    def chain_vector(t, z, tau=30):
        zn = np.zeros(t.size, dtype=float)
        zl = np.zeros(t.size, dtype=float)
        buffer = np.zeros(100, dtype=float)
        n = 10
        inter = 0
        zp = z[0]
        for i in range(t.size):
            _, _z = outlier(buffer, t[i], z[i], 100, 2)
            _, _z = ema(tau, inter, buffer[3:], t[i], _z)
            zn[i] = _z
            _, _z = limit(t[i], _z, -50, 50, None, None, None, None)
            zl[i] = _z
        return t, zn, zl

    t, zn, zl = chain_vector(t, zo)
    plot_timeseries(t, zn, zl, title="denoise + control limits")

    timer = timeit.Timer(lambda: chain_vector(t, zo))
    print(f"time per operation:{timer.timeit(100) / 100.0}")
