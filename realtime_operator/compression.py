import math
import numba as nb
import numpy as np
from numba.typed import List
import sys
from enum import Enum

COMPRESSION_TYPE={"DEDUPLICATE" : 0,
    "MINIMUM_TIMEDELTA" : 1,
    "EXCEPTION_DEVIATION" : 2,
    "EXCEPTION_DEVIATION_PREVIOUS" : 3,
    "SWINGING_DOOR": 4
}
EPSILON = 1e-15

def interpolate(t,tn,zn):
    return np.interp(t, tn, zn, left=np.nan, right=np.nan)

@nb.jit(nopython=True)
def interpolate_fast(t,tn,zn):
    npointer=0
    len_t = len(t)
    len_tn = len(tn)
    z = np.full(len_t, np.nan, dtype=float)
    # start condition: t[i]>=tn[0]
    left_t = 0
    left_tn = 0

    # no interpolation left of the first point
    while left_t < len_t and t[left_t] < tn[0]:
        left_t += 1

    # llop through all t values
    for i in range(left_t, len_t):
        while left_tn < len_tn-1 and t[i] > tn[left_tn+1]:
            left_tn += 1
        if left_tn == len_tn - 1:
            break
        if t[i] >= tn[left_tn] and t[i] <= tn[left_tn+1]:
            m=(zn[left_tn+1]-zn[left_tn])/(tn[left_tn+1]-tn[left_tn])
            z[i] = zn[left_tn] + m * (t[i] - tn[left_tn])
    return z



@nb.jit(nopython=True)
def any_compression(t,z,delta, ftype):
    tn=[]
    zn=[]
    cn=[]
    n = len(z)
    if ftype == 0:
        state = np.zeros(3, dtype=float)
        for i in range(n):
            tn_ , zn_, cn_ = deduplicate(state, t[i], z[i],0.0, 1.0e6)
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    elif ftype == 1:
        state = np.zeros(3, dtype=float)
        for i in range(n):
            tn_ , zn_, cn_ = minimum_timedelta(delta,state, t[i], z[i])
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    elif ftype == 2:
        state = np.zeros(3, dtype=float)
        for i in range(n):
            tn_ , zn_, cn_ = exception_deviation(delta,state, t[i], z[i],0.0, 1.0e6)
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    elif ftype == 3:
        state = np.zeros(5, dtype=float)
        for i in range(n):
            tn_ , zn_, cn_ = exception_deviation_previous(delta,state, t[i], z[i],0.0, 1.0e6)
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    elif ftype == 4:
        state = np.zeros(7, dtype=float)
        for i in range(n):
            tn_ , zn_, cn_ = swinging_door(delta,state, t[i], z[i],0.0, 1.0e6)
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    # always capture the last point to allow for interpolation:
    if abs(tn[-1] - t[-1]) >= EPSILON:
        tn.append(t[-1])
        zn.append(z[-1])
        cn.append(0)
    return np.array(tn,np.float64),np.array(zn,np.float64),np.array(cn,np.float64)

@nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8[:], f8, f8, f8, f8)", nopython=True)
def deduplicate(state, t, z, min_duration_seconds=0, max_duration_seconds=1e9):
    """
    Deduplicates data points based on time and value.

    Args:
        state (numpy.ndarray): State array to store previous time, value, and count.
        t (float): Current time.
        z (float): Current value.
        min_duration_seconds (float, optional): Minimum duration between data points in seconds. Defaults to 0.
        max_duration_seconds (float, optional): Maximum duration between data points in seconds. Defaults to 1e9.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple containing arrays of deduplicated time, value, and count.
    """
    
    if state[0] == 0:
        state[0] = t
        state[1] = z
        state[2] = 0
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )
    deviation_condition = abs(z - state[1]) >= EPSILON
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
    """
    Filters data points based on minimum time duration.

    Args:
        duration_seconds (float): Minimum duration between data points in seconds.
        state (numpy.ndarray): State array to store previous time, value, and count.
        t (float): Current time.
        z (float): Current value.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple containing arrays of filtered time, value, and count.
    """
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
    """
    Filters data points based on deviation from previous value.

    Args:
        deviation (float): Maximum allowed deviation from previous value.
        state (numpy.ndarray): State array to store previous time, value, and count.
        t (float): Current time.
        z (float): Current value.
        min_duration_seconds (float, optional): Minimum duration between data points in seconds. Defaults to 0.
        max_duration_seconds (float, optional): Maximum duration between data points in seconds. Defaults to 1e9.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple containing arrays of filtered time, value, and count.
    """
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
    """
    Filters data points based on deviation from previous value and previous point.

    Args:
        deviation (float): Maximum allowed deviation from previous value.
        state (numpy.ndarray): State array to store previous time, value, previous point, and count.
        t (float): Current time.
        z (float): Current value.
        min_duration_seconds (float, optional): Minimum duration between data points in seconds. Defaults to 0.
        max_duration_seconds (float, optional): Maximum duration between data points in seconds. Defaults to 1e9.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple containing arrays of filtered time, value, and count.
    """
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


@nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8, f8, f8)", nopython=True)
def swinging_door(
    deviation, state, t, z, min_duration_seconds=0, max_duration_seconds=1e9
):
    """
    Filters data points based on deviation from previous value and slope.

    Args:
        deviation (float): Maximum allowed deviation from previous value.
        state (numpy.ndarray): State array to store previous time, value, minimum slope, maximum slope, previous point, and count.
        t (float): Current time.
        z (float): Current value.
        min_duration_seconds (float, optional): Minimum duration between data points in seconds. Defaults to 0.
        max_duration_seconds (float, optional): Maximum duration between data points in seconds. Defaults to 1e9.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple containing arrays of filtered time, value, and count.
    """
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
    # GE = deviation/2
    # PI = deviation
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
        # only update if min_slope>prev_min_slopw or max_slope<prev_max_slope
        if min_slope > state[2]:
            state[2] = min_slope
        if max_slope < state[3]:
            state[3] = max_slope
        state[4] = t
        state[5] = z
        return (
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )

