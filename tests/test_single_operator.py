import numpy as np
import pytest
import math
from realtime_operator.single_operator import (
    BLOCK_SIZE,
    to_utc_seconds,
    from_utc_seconds,
    identity,
    update_buffer,
    tick,
    diff,
    diff_slope,
    log_return,
    ema,
    nema,
    delta,
    delta2nd,
    xy_slope,
    ma,
    _msd,
    msd,
    zscore,
    _cor,
    cor,
    cor2,
    linear_regression,
    linear_regression2,
    downsample,
    outlier,
    outlier_slope,
)

SAMPLE_SIZE = 10
BUFFER_SIZE = 100


def create_test_data(n=SAMPLE_SIZE):
    t = np.arange(1, n, 1).astype(float)
    z = np.arange(1, n, 1).astype(float)
    buffer = np.zeros(BUFFER_SIZE, dtype=float)
    return t, z, buffer


def test_to_utc_seconds():
    dt64 = np.datetime64("2022-01-01T00:00:00")
    assert to_utc_seconds(dt64) == 1640995200


def test_from_utc_seconds():
    seconds_since_epoch = 1640995200.0
    assert from_utc_seconds(seconds_since_epoch) == np.datetime64("2022-01-01T00:00:00")


def test_identity():
    assert identity(1, 2, 3) == (1, 2, 3)


def test_update_buffer():
    buffer = np.zeros(100, dtype=float)
    ptr = update_buffer(buffer, 2)
    assert ptr == 1
    assert buffer[0] == 1 + 2
    ptr = update_buffer(buffer, 2)
    assert ptr == 3
    assert buffer[0] == 3 + 2


def test_exceed_buffer_size():
    buffer = np.zeros(100, dtype=float)
    try:
        ptr = update_buffer(buffer, 101)
        assert False
    except Exception as e:
        assert "Buffer too small"


def test_benchmark_udapte_buffer(benchmark):
    buffer = np.zeros(500, dtype=float)

    def setup():
        return (buffer, 2), {}

    benchmark.pedantic(update_buffer, setup=setup, rounds=100, warmup_rounds=1)


def test_tick():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = tick(buffer, t[i], z[i])

    assert tn == t[-1]
    assert zn == 1
    assert buffer[0] == t[-1]
    assert buffer[1] == z[-1]


def test_diff():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = diff(buffer, t[i], z[i])

    assert tn == t[-1]
    assert zn == 1


def test_diff_slope():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = diff_slope(buffer, t[i], z[i])

    assert tn == t[-1]
    assert zn == 1


def test_log_return():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = log_return(buffer, t[i], z[i])

    assert tn == t[-1]
    assert abs(math.log(9 / 8) - zn) < 1e-9


def test_ema():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = ema(1, 0, buffer, t[i], z[i])

    assert tn == t[-1]
    assert zn > z[-2]
    assert zn < z[-1]


def test_nema():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        buffer[0] = 0
        tn, zn = nema(1, 0, 5, buffer, t[i], z[i])

    assert tn == t[-1]
    assert zn < z[-1]


def test_delta():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        buffer[0] = 0
        tn, zn = delta(1, buffer, t[i], z[i])

    assert tn == t[-1]
    assert zn > 0


def test_delta2nd():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        buffer[0] = 0
        tn, zn = delta2nd(1, buffer, t[i], z[i])

    assert tn == t[-1]
    assert abs(zn) < 1e-2


def test_xy_slope():
    t, x, buffer = create_test_data()
    y = np.arange(2, 20, 1).astype(float)
    zn = 0
    for i in range(9):
        tn, zn = xy_slope(1, buffer, t[i], x[i], y[i])

    assert tn == t[-1]
    assert abs(zn) < 3


def test_ma():
    t, z, buffer = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = ma(1, 0, 5, buffer, t[i], z[i])

    assert tn == t[-1]
    assert zn < z[-1]


def test_benchmark_ma(benchmark):
    buffer = np.zeros(500, dtype=float)

    def setup():
        return (1, 0, 10, buffer, 1, 2), {}

    benchmark.pedantic(ma, setup=setup, rounds=100, warmup_rounds=1)


def test__msd():
    t, z, buffer = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = _msd(1, 0, 5, buffer, t[i], z[i])

    assert tn == t[-1]
    assert zn.size == 3


def test_msd():
    t, z, buffer = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, _ = msd(1, 0, 5, buffer, t[i], z[i])

    assert tn == t[-1]


"""    zscore,
    _cor,
    cor,
    cor2,
    linear_regression,
    linear_regression2,
    downsample,
    outlier,
    outlier_slope
    """


def test_zscore():
    t, z, buffer = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, _ = zscore(1, 0, 5, buffer, t[i], z[i])

    assert tn == t[-1]


def test__cor():
    t, z, buffer = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = _cor(1, 0, 5, buffer, t[i], z[i], z[i])

    assert tn == t[-1]
    assert abs(zn[6] - 1) < 1e6


def test_cor():
    t, z, buffer = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = cor(1, 0, 5, buffer, t[i], z[i], z[i])

    assert tn == t[-1]
    assert abs(zn - 1) < 1e6


def test_linear_regression():
    t, z, buffer = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = linear_regression(1, 0, 5, buffer, t[i], z[i], z[i])

    assert tn == t[-1]
    assert abs(zn - 1) < 1e6


def test_linear_regression2():
    t, z, buffer = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = linear_regression2(1, 0, 5, buffer, t[i], z[i], z[i])

    assert tn == t[-1]
    assert abs(zn - 1) < 1e6


def test_outlier():
    t, z, buffer = create_test_data(100)
    z[-1] = 5
    zn = 0
    for i in range(99):
        tn, zn = outlier(buffer, t[i], z[i], 2, 1)

    assert tn == t[-1]
    assert zn == z[-2]


def test_outlier_slope():
    t, z, buffer = create_test_data(100)
    z[-1] = 5
    zn = 0
    for i in range(99):
        tn, zn = outlier_slope(buffer, t[i], z[i], 2, 1)

    assert tn == t[-1]
    assert zn == z[-2]
