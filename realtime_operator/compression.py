import math
import numba as nb
import numpy as np
import sys


# @nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8[:], f8, f8, f8, f8)", nopython=True)
def deduplicate(state, t, z, min_duration_seconds=0, max_duration_seconds=1e9):
    epsilon = 1e-15
    if state[0] == 0:
        state[0] = t
        state[1] = z
        state[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )
    deviation_condition = abs(z - state[1]) >= epsilon
    min_duration_condition = (t - state[0]) >= min_duration_seconds
    max_duration_condition = (t - state[0]) >= max_duration_seconds

    state[2] += 1
    if (deviation_condition and min_duration_condition) or max_duration_condition:
        state[0] = t
        state[1] = z
        count = state[2]
        state[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([count], dtype=np.float64),
        )
    else:
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )


@nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8)", nopython=True)
def minimum_timedelta(duration_seconds, state, t, z):
    if state[0] == 0:
        state[0] = t
        state[1] = z
        state[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

    state[2] += 1
    if t - state[0] >= duration_seconds:
        state[0] = t
        state[1] = z
        count = state[2]
        state[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([count], dtype=np.float64),
        )
    else:
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )


@nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8, f8, f8)", nopython=True)
def exception_deviation(
    deviation, state, t, z, min_duration_seconds=0, max_duration_seconds=1e9
):
    if state[0] == 0:
        state[0] = t
        state[1] = z
        state[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

    state[2] += 1
    deviation_condition = abs(z - state[1]) >= deviation
    min_duration_condition = (t - state[0]) >= min_duration_seconds
    max_duration_condition = (t - state[0]) >= max_duration_seconds

    if (deviation_condition and min_duration_condition) or max_duration_condition:
        state[0] = t
        state[1] = z
        count = state[2]
        state[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([count], dtype=np.float64),
        )
    else:
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )


@nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8, f8, f8)", nopython=True)
def exception_deviation_previous(
    deviation, state, t, z, min_duration_seconds=0, max_duration_seconds=1e9
):
    if state[0] == 0:
        state[0] = t
        state[1] = z
        state[2] = t  # previous point
        state[3] = z  # previous point
        state[4] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

    state[4] += 1
    deviation_condition = abs(z - state[1]) >= deviation
    min_duration_condition = (t - state[0]) >= min_duration_seconds
    max_duration_condition = (t - state[0]) >= max_duration_seconds

    if (deviation_condition and min_duration_condition) or max_duration_condition:
        # case 1: last hold point and previous point are the same
        count = state[4]
        if abs(state[0] - state[2]) < 1.0e-6:
            t_array = [t]
            z_array = [z]
            c_array = [count]
        else:
            # case 2: last hold point and previous point are different
            t_array = [state[2], t]
            z_array = [state[3], z]
            c_array = [count, 0]
        state[0] = t
        state[1] = z
        state[2] = t  # previous point
        state[3] = z  # previous point
        state[4] = 0
        # send previous point if exists
        return (
            np.array(t_array, dtype=np.float64),
            np.array(z_array, dtype=np.float64),
            np.array(c_array, dtype=np.float64),
        )
    else:
        state[2] = t
        state[3] = z
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )


# @nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8, f8, f8)", nopython=True)
def swinging_door(
    deviation, state, t, z, min_duration_seconds=0, max_duration_seconds=1e9
):
    if state[0] == 0:
        state[0] = t
        state[1] = z
        state[2] = -np.inf  # min slope
        state[3] = np.inf  # max slope
        state[4] = t
        state[5] = z
        state[6] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

    state[6] += 1
    # calculate slopes
    min_slope = (z - deviation - state[1]) / (t - state[0])
    max_slope = (z + deviation - state[1]) / (t - state[0])

    slope_condition = min_slope <= state[2] or max_slope >= state[3]
    min_duration_condition = (t - state[0]) >= min_duration_seconds
    max_duration_condition = (t - state[0]) >= max_duration_seconds

    if (slope_condition and min_duration_condition) or max_duration_condition:
        tp = state[4]
        zp = state[5]
        state[0] = tp
        state[1] = zp
        state[2] = -np.inf  # min slope
        state[3] = np.inf  # max slope
        state[4] = t
        state[5] = z
        count = state[6]
        state[6] = 0
        return (
            np.array([tp], dtype=np.float64),
            np.array([zp], dtype=np.float64),
            np.array([count], dtype=np.float64),
        )
    else:
        state[2] = min_slope
        state[3] = max_slope
        state[4] = t
        state[5] = z
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )
