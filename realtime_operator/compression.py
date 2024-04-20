import math
import numba as nb
import numpy as np


@nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8)", nopython=True)
def timedelta_min(duration_seconds, buffer, t, z):
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        buffer[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

    buffer[2] += 1
    if t - buffer[0] >= duration_seconds:
        buffer[0] = t
        buffer[1] = z
        count = buffer[2]
        buffer[2] = 0
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
    deviation, buffer, t, z, min_duration_seconds=0, max_duration_seconds=1e9
):
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        buffer[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

    buffer[2] += 1
    deviation_condition = abs(z - buffer[1]) >= deviation
    min_duration_condition = (t - buffer[0]) >= min_duration_seconds
    max_duration_condition = (t - buffer[0]) >= max_duration_seconds

    if (deviation_condition and min_duration_condition) or max_duration_condition:
        buffer[0] = t
        buffer[1] = z
        count = buffer[2]
        buffer[2] = 0
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
def pi_exception_deviation(
    deviation, buffer, t, z, min_duration_seconds=0, max_duration_seconds=1e9
):
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        buffer[2] = t  # previous point
        buffer[3] = z  # previous point
        buffer[4] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

    buffer[4] += 1
    deviation_condition = abs(z - buffer[1]) >= deviation
    min_duration_condition = (t - buffer[0]) >= min_duration_seconds
    max_duration_condition = (t - buffer[0]) >= max_duration_seconds

    if (deviation_condition and min_duration_condition) or max_duration_condition:
        # case 1: last hold point and previous point are the same
        count = buffer[4]
        if abs(buffer[0] - buffer[2]) < 1.0e-6:
            t_array = [t]
            z_array = [z]
            c_array = [count]
        else:
            # case 2: last hold point and previous point are different
            t_array = [buffer[2], t]
            z_array = [buffer[3], z]
            c_array = [count, 0]
        buffer[0] = t
        buffer[1] = z
        buffer[2] = t  # previous point
        buffer[3] = z  # previous point
        buffer[4] = 0
        # send previous point if exists
        return (
            np.array(t_array, dtype=np.float64),
            np.array(z_array, dtype=np.float64),
            np.array(c_array, dtype=np.float64),
        )
    else:
        buffer[2] = t
        buffer[3] = z
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )


# @nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8, f8, f8)", nopython=True)
def swinging_door(
    deviation, buffer, t, z, min_duration_seconds=0, max_duration_seconds=1e9
):
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        buffer[2] = -np.inf  # min slope
        buffer[3] = np.inf  # max slope
        buffer[4] = t
        buffer[5] = z
        buffer[6] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

    buffer[6] += 1
    # calculate slopes
    min_slope = (z - deviation - buffer[1]) / (t - buffer[0])
    max_slope = (z + deviation - buffer[1]) / (t - buffer[0])

    slope_condition = min_slope <= buffer[2] or max_slope >= buffer[3]
    min_duration_condition = (t - buffer[0]) >= min_duration_seconds
    max_duration_condition = (t - buffer[0]) >= max_duration_seconds

    if (slope_condition and min_duration_condition) or max_duration_condition:
        tp = buffer[4]
        zp = buffer[5]
        buffer[0] = tp
        buffer[1] = zp
        buffer[2] = -np.inf  # min slope
        buffer[3] = np.inf  # max slope
        buffer[4] = t
        buffer[5] = z
        count = buffer[6]
        buffer[6] = 0
        return (
            np.array([tp], dtype=np.float64),
            np.array([zp], dtype=np.float64),
            np.array([count], dtype=np.float64),
        )
    else:
        buffer[2] = min_slope
        buffer[3] = max_slope
        buffer[4] = t
        buffer[5] = z
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )
