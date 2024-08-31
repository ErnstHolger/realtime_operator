import numpy as np
import pytest
import math
from realtime_operator.single_operator import (
    BLOCK_SIZE,
    to_utc_seconds,
    from_utc_seconds,
    identity,
    update_state,
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
    median
)

SAMPLE_SIZE = 10
state_SIZE = 100


def create_test_data(n=SAMPLE_SIZE):
    t = np.arange(1, n, 1).astype(float)
    z = np.arange(1, n, 1).astype(float)
    state = np.zeros(state_SIZE, dtype=float)
    return t, z, state

def test_median():
    t, z, state = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = median(state, t[i], z[i],5)
    assert zn==97

def test_to_utc_seconds():
    dt64 = np.datetime64("2022-01-01T00:00:00")
    assert to_utc_seconds(dt64) == 1640995200


def test_from_utc_seconds():
    seconds_since_epoch = 1640995200.0
    assert from_utc_seconds(seconds_since_epoch) == np.datetime64("2022-01-01T00:00:00")


def test_identity():
    assert identity(1, 2, 3) == (1, 2, 3)


def test_update_state():
    state = np.zeros(100, dtype=float)
    ptr = update_state(state, 2)
    assert ptr == 1
    assert state[0] == 1 + 2
    ptr = update_state(state, 2)
    assert ptr == 3
    assert state[0] == 3 + 2


def test_exceed_state_size():
    state = np.zeros(100, dtype=float)
    try:
        ptr = update_state(state, 101)
        assert False
    except Exception as e:
        assert "state too small"


def test_tick():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = tick(state, t[i], z[i])

    assert tn == t[-1]
    assert zn == 1
    assert state[0] == t[-1]
    assert state[1] == z[-1]


def test_diff():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = diff(state, t[i], z[i])

    assert tn == t[-1]
    assert zn == 1


def test_diff_slope():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = diff_slope(state, t[i], z[i])

    assert tn == t[-1]
    assert zn == 1


def test_log_return():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = log_return(state, t[i], z[i])

    assert tn == t[-1]
    assert abs(math.log(9 / 8) - zn) < 1e-9


def test_ema():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = ema(1, 0, state, t[i], z[i])

    assert tn == t[-1]
    assert zn > z[-2]
    assert zn < z[-1]

def test_ema_ooo():
    t, z, state = create_test_data()

    # add ooo event
    t[6]=1
    z[6]=100_000
    zn = 0
    for i in range(9):
        tn, zn = ema(1, 0, state, t[i], z[i])

    assert tn == t[-1]
    assert zn > z[-2]
    assert zn < z[-1]


def test_nema():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        state[0] = 0
        tn, zn = nema(1, 0, 5, state, t[i], z[i])

    assert tn == t[-1]
    assert zn < z[-1]

def test_nema_ooo():
    t, z, state = create_test_data()
    t[6]=1
    z[6]=100_000
    zn = 0
    for i in range(9):
        state[0] = 0
        tn, zn = nema(1, 0, 5, state, t[i], z[i])

    assert tn == t[-1]
    assert zn < z[-1]


def test_delta():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        state[0] = 0
        tn, zn = delta(1, state, t[i], z[i])

    assert tn == t[-1]
    assert zn > 0


def test_delta2nd():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        state[0] = 0
        tn, zn = delta2nd(1, state, t[i], z[i])

    assert tn == t[-1]
    assert abs(zn) < 1e-2


def test_xy_slope():
    t, x, state = create_test_data()
    y = np.arange(2, 20, 1).astype(float)
    zn = 0
    for i in range(9):
        tn, zn = xy_slope(1, state, t[i], x[i], y[i])

    assert tn == t[-1]
    assert abs(zn) < 3


def test_ma():
    t, z, state = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = ma(1, 0, 5, state, t[i], z[i])

    assert tn == t[-1]
    assert zn < z[-1]


def test__msd():
    t, z, state = create_test_data()
    zn = 0
    for i in range(9):
        tn, zn = _msd(1, 0, 5, state, t[i], z[i])

    assert tn == t[-1]
    assert zn.size == 3


def test_msd():
    t, z, state = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, _ = msd(1, 0, 5, state, t[i], z[i])

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
    t, z, state = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, _ = zscore(1, 0, 5, state, t[i], z[i])

    assert tn == t[-1]


def test__cor():
    t, z, state = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = _cor(1, 0, 5, state, t[i], z[i], z[i])

    assert tn == t[-1]
    assert abs(zn[6] - 1) < 1e6


def test_cor():
    t, z, state = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = cor(1, 0, 5, state, t[i], z[i], z[i])

    assert tn == t[-1]
    assert abs(zn - 1) < 1e6


def test_linear_regression():
    t, z, state = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = linear_regression(1, 0, 5, state, t[i], z[i], z[i])

    assert tn == t[-1]
    assert abs(zn - 1) < 1e6


def test_linear_regression2():
    t, z, state = create_test_data(100)
    zn = 0
    for i in range(99):
        tn, zn = linear_regression2(1, 0, 5, state, t[i], z[i], z[i])

    assert tn == t[-1]
    assert abs(zn - 1) < 1e6


def test_outlier():
    t, z, state = create_test_data(100)
    z[-1] = 5
    zn = 0
    for i in range(99):
        tn, zn = outlier(state, t[i], z[i], 2, 1)

    assert tn == t[-1]
    assert zn == z[-2]


def test_outlier_slope():
    t, z, state = create_test_data(100)
    z[-1] = 5
    zn = 0
    for i in range(99):
        tn, zn = outlier_slope(state, t[i], z[i], 2, 1)

    assert tn == t[-1]
    assert zn == z[-2]
