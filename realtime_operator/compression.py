import numba as nb
import numpy as np

from realtime_operator.single_operator import msd

COMPRESSION_TYPE = {
    "DEDUPLICATE": 0,
    "MINIMUM_TIMEDELTA": 1,
    "EXCEPTION_DEVIATION": 2,
    "EXCEPTION_DEVIATION_PREVIOUS": 3,
    "SWINGING_DOOR": 4,
}
EPSILON = 1e-15


def interpolate(t, tn, zn):
    return np.interp(t, tn, zn, left=np.nan, right=np.nan)


def segment_resample(t, z, lb, ub, count, ratio=0.6):
    n = len(t)
    main_count = int(round(ratio * n / count, 0))
    side_count = int(round((1 - ratio) * n / (count * 2), 0))
    lb_index = int(round(lb * n, 0))
    lb_step = int(round(lb * n / side_count, 0))
    ub_index = int(ub * n)
    main_step = int(round((ub - lb) * n / main_count, 0))
    ub_step = int((1 - ub) * n / side_count)
    ti = []
    zi = []
    index = []
    if lb_step > 0:
        for i in range(0, lb_index, lb_step):
            ti.append(float(t[i]))
            zi.append(float(z[i]))
            index.append(int(i))
    if main_step > 0:
        for i in range(lb_index, ub_index, main_step):
            ti.append(float(t[i]))
            zi.append(float(z[i]))
            index.append(int(i))
    if ub_step > 0:
        for i in range(ub_index, n, ub_step):
            ti.append(float(t[i]))
            zi.append(float(z[i]))
            index.append(int(i))
    return ti, zi, index


@nb.jit(nopython=True)
def interpolate_fast(t, tn, zn):
    len_t = len(t)
    len_tn = len(tn)
    z = np.full(len_t, np.nan, dtype=float)
    # start condition: t[i]>=tn[0]
    left_t = 0
    left_tn = 0

    # no interpolation left of the first point
    while left_t < len_t and t[left_t] < tn[0]:
        left_t += 1

    # loop through all t values
    for i in range(left_t, len_t):
        while left_tn < len_tn - 1 and t[i] > tn[left_tn + 1]:
            left_tn += 1
        if left_tn == len_tn - 1:
            break
        if t[i] >= tn[left_tn] and t[i] <= tn[left_tn + 1]:
            m = (zn[left_tn + 1] - zn[left_tn]) / (tn[left_tn + 1] - tn[left_tn])
            z[i] = zn[left_tn] + m * (t[i] - tn[left_tn])
    return t, z


@nb.jit(nopython=True)
def minimum_timedelta_interp(t, z, delta):
    n = len(z)
    if n == 0:
        return t, z
    t_new = np.arange(t[0], t[-1] + delta, delta)
    z_new = interpolate_fast(t_new, t, z)
    return t_new, z_new


@nb.jit(nopython=True)
def any_compression(t, z, delta, ftype):
    """
    Applies one of the following compression algorithms to the data:

    0: deduplicate
    1: minimum_timedelta
    2: exception_deviation
    3: exception_deviation_previous
    4: swinging_door

    Args:
        t (numpy.ndarray): Array of timestamps.
        z (numpy.ndarray): Array of values.
        delta (float): Parameter for the selected compression algorithm.
        ftype (int): Type of compression algorithm to use.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple containing arrays of compressed time, value, and count.
    """
    tn = []
    zn = []
    cn = []
    n = len(z)
    if ftype == 0:
        state = np.zeros(3, dtype=float)
        for i in range(n):
            tn_, zn_, cn_ = deduplicate(state, t[i], z[i], 0.0, 1.0e6)
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    elif ftype == 1:
        state = np.zeros(3, dtype=float)
        for i in range(n):
            tn_, zn_, cn_ = minimum_timedelta(delta, state, t[i], z[i])
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    elif ftype == 2:
        state = np.zeros(3, dtype=float)
        for i in range(n):
            tn_, zn_, cn_ = exception_deviation(delta, state, t[i], z[i], 0.0, 1.0e6)
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    elif ftype == 3:
        state = np.zeros(5, dtype=float)
        for i in range(n):
            tn_, zn_, cn_ = exception_deviation_previous(
                delta, state, t[i], z[i], 0.0, 1.0e6
            )
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    elif ftype == 4:
        state = np.zeros(7, dtype=float)
        for i in range(n):
            tn_, zn_, cn_ = swinging_door(delta, state, t[i], z[i], 0.0, 1.0e6)
            for i in range(len(tn_)):
                tn.append(tn_[i])
                zn.append(zn_[i])
                cn.append(cn_[i])
    # always capture the last point to allow for interpolation:
    if abs(tn[-1] - t[-1]) >= EPSILON:
        tn.append(t[-1])
        zn.append(z[-1])
        cn.append(0)
    return np.array(tn, np.float64), np.array(zn, np.float64), np.array(cn, np.float64)


@nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8[:], f8, f8, f8, f8)", nopython=True)
def deduplicate(state, t, z, min_duration_seconds, max_duration_seconds):
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

    # ooo condition
    out_of_order = t <= state[0]
    if out_of_order:
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

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
            np.empty(0),
            np.empty(0),
            np.empty(0),
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

    # ooo condition
    out_of_order = t <= state[0]
    if out_of_order:
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
            np.empty(0),
            np.empty(0),
            np.empty(0),
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

    # ooo condition
    out_of_order = t <= state[0]
    if out_of_order:
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

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
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )


@nb.jit("Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8, f8, f8)", nopython=True)
def exception_deviation_previous(
    deviation, state, t, z, min_duration_seconds=0, max_duration_seconds=1e9
):
    """
    Filters data points based on deviation from previous value and previous point.

    This function is similar to exception_deviation, but it also checks the deviation from the previous point.

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
        # initialization
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

    # ooo condition
    out_of_order = t <= state[0]
    if out_of_order:
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

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
            np.empty(0),
            np.empty(0),
            np.empty(0),
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
        state[2] = np.finfo(dtype=np.float64).min  # min slope
        state[3] = np.finfo(dtype=np.float64).max  # max slope
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

    # ooo condition
    if t <= state[0]:
        # return event
        return (
            np.array([t], dtype=np.float64),
            np.array([z], dtype=np.float64),
            np.array([0], dtype=np.float64),
        )

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
        state[2] = np.finfo(dtype=np.float64).min  # min slope
        state[3] = np.finfo(dtype=np.float64).max  # max slope
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
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )


@nb.jit(
    "Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8, f8, f8, i8, i8, f8, f8)", nopython=True
)
def swinging_door_auto(
    deviation,
    state,
    t,
    z,
    tau,
    inter,
    n,
    min_duration_seconds=0,
    max_duration_seconds=1e9,
):
    # estimate the stadard deviation
    _, deviation = msd(tau, inter, n, state, t, z)
    tn, zn, count = swinging_door(
        deviation, state, t, z, min_duration_seconds, max_duration_seconds
    )
    return tn, zn, count


@nb.jit(
    "Tuple((f8[:], f8[:], f8[:]))(f8, f8[:], f8[:], f8[:], f8, i8, i8, f8, f8)",
    nopython=True,
)
def swinging_door_timeseries(
    deviation,
    state,
    t,
    z,
    tau,
    inter,
    n,
    min_duration_seconds=0,
    max_duration_seconds=1e9,
):
    # estimate the stadard deviation
    n = len(t)
    for i in range(n):
        _, deviation = msd(tau, inter, n, state, t[i], z[i])
        tn, zn, count = swinging_door(
            deviation, state, t[i], z[i], min_duration_seconds, max_duration_seconds
        )
    return tn, zn, count
