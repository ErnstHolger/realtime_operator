import numpy as np
import time

from realtime_operator.compression import (
    COMPRESSION_TYPE,
    interpolate,    
    interpolate_fast,
    any_compression,
    deduplicate,
    minimum_timedelta,
    exception_deviation,
    exception_deviation_previous,
    swinging_door,
)

t = np.arange(1, 20, dtype=float) + 1
z = np.arange(1, 20, dtype=float)
state = np.zeros(4, dtype=float)


def test_interpolate_fast():
    t = np.arange(1, 20,1, dtype=float)
    tn= np.arange(0, 20,0.001, dtype=float) 

    zn=interpolate(tn,t,z)
    zm=interpolate_fast(tn,t,z)

    zn = np.where(np.isnan(zn), 0, zn)
    zm = np.where(np.isnan(zm), 0, zm)
    assert np.all(zn == zm)


def test_deduplicate():
    state = np.zeros(3, dtype=float)
    zp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0]
    n = len(zp)
    result = []
    for i in range(n):
        _ , zn, _ = deduplicate(state, t[i], zp[i],0.0, 1e6)
        for i in zn:
            result.append(i)

    desired_array = np.array([1.0, 2.0, 3.0])
    assert np.all(np.array(result) == desired_array)


def test_minimum_timedelta():
    n = len(z)
    state = np.zeros(3, dtype=float)
    result = []
    for i in range(n):
        _, zn, _ = minimum_timedelta(3, state, t[i], z[i])
        for i in zn:
            result.append(i)

    desired_array = np.array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0])
    assert np.all(np.array(result) == desired_array)


def test_exception_deviation():
    n = len(z)
    state = np.zeros(3, dtype=float)
    result = []
    for i in range(n):
        _, zn, _ = exception_deviation(3, state, t[i], z[i], 0, 1e6)
        for i in zn:
            result.append(i)

    desired_array = np.array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0])
    assert np.all(np.array(result) == desired_array)


def test_exception_deviation_previous():
    n = len(z)
    state = np.zeros(5, dtype=float)
    result = []
    for i in range(n):
        _, zn, _ = exception_deviation_previous(4, state, t[i], z[i], 0, 1e6)
        for i in zn:
            result.append(i)

    desired_array = np.array([1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0, 16.0, 17.0])
    assert np.all(np.array(result) == desired_array)


def test_swinging_door():
    zp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    n = len(zp)
    state = np.zeros(7, dtype=float)
    result = []
    for i in range(n):
        _, zn, _ = swinging_door(4, state, t[i], zp[i], 0, 1e6)
        for i in zn:
            result.append(i)

    desired_array = np.array([1.0, 10.0])
    assert np.all(np.array(result) == desired_array)
