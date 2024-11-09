import math

import numba as nb
import numpy as np

BLOCK_SIZE = 512
START_TIME = "2024-01-01 08:00:00"


def to_utc_seconds(dt64):
    """
    Converts a numpy datetime64 object to the number of seconds since the Unix epoch (UTC).

    Parameters:
    dt64 (numpy.datetime64): The datetime object to be converted.

    Returns:
    float: The number of seconds since the Unix epoch (UTC).
    """
    # Create a datetime object at UTC
    unix_epoch = np.datetime64(0, "s")
    seconds_since_epoch = (dt64 - unix_epoch) / np.timedelta64(1, "s")
    return seconds_since_epoch


def from_utc_seconds(seconds_since_epoch):
    """
    Converts seconds since the Unix epoch to a numpy.datetime64 object.

    Parameters:
    seconds_since_epoch (int): The number of seconds since the Unix epoch.

    Returns:
    numpy.datetime64: The corresponding datetime object.

    """
    # The Unix epoch as a numpy.datetime64 object
    unix_epoch = np.datetime64(0, "s")
    # Convert seconds since epoch to a timedelta64, and add it to the epoch
    dt64 = unix_epoch + np.timedelta64(int(seconds_since_epoch), "s")
    return dt64


@nb.jit("Tuple((f8, f8, f8))(f8, f8, f8)", nopython=True)
def identity(n, t, z):
    """
    Returns the given parameters n, t, and z.

    Parameters:
        n (int): The value of n.
        t (float): The value of t.
        z (str): The value of z.

    Returns:
        tuple: A tuple containing the values of n, t, and z.
    """
    return n, t, z


def inhomogeneous_time_seris(n, min_step=0.1, max_step=2.5, offset=1_704_096_000):
    """
    Generate an inhomogeneous time series.

    Parameters:
    - n (int): Number of time steps to generate.
    - min_step (float): Minimum time step.
    - max_step (float): Maximum time step.
    - offset (int): Offset to add to the generated time series.

    Returns:
    - t (ndarray): Array of generated time steps.
    """

    time_delta = np.random.uniform(min_step, max_step, n)
    t = np.cumsum(time_delta) + offset
    return t


@nb.njit()
def sinusoid(t, amplitude, shift, frequency):
    """
    Generate a sinusoidal waveform.

    Parameters:
    t (array-like): Time values.
    amplitude (float): Amplitude of the waveform.
    shift (float): Phase shift of the waveform.
    frequency (float): Frequency of the waveform.

    Returns:
    tuple: A tuple containing the time values and the generated waveform.
    """
    z = amplitude * np.sin(2 * np.pi * frequency * (t + shift))
    return t, z


@nb.njit()
def square(t, amplitude, frequency, shift):
    # Modified to go from 0 to amplitude instead of -amplitude to +amplitude
    z = (amplitude / 2) * (np.sign(np.sin(2 * np.pi * frequency * (t + shift))) + 1)
    return t, z


@nb.njit()
def sawtooth(t, amplitude, frequency, shift):
    # Modified to go from 0 to amplitude
    z = amplitude * ((t + shift) * frequency - np.floor((t + shift) * frequency))
    return t, z


@nb.njit()
def gaussian(t, amplitude, frequency, shift, std):
    # frequency determines how often the pulse repeats
    # std controls the width of each pulse
    # shift moves the pattern left/right

    # Calculate position within each period
    period = 1 / frequency
    wrapped_t = (t + shift) % period

    # Center the pulse in each period
    center = period / 2

    # Calculate the Gaussian pulse
    z = amplitude * np.exp(-((wrapped_t - center) ** 2) / (2 * std**2))
    return t, z


@nb.njit()
def white_noise(length, mean=0, std_dev=1):
    """
    Generate white noise samples.

    Parameters:
    - length (int): The number of samples to generate.
    - mean (float, optional): The mean of the normal distribution. Default is 0.
    - std_dev (float, optional): The standard deviation of the normal distribution. Default is 1.

    Returns:
    - numpy.ndarray: An array of white noise samples.
    """
    return np.random.normal(mean, std_dev, length)


@nb.njit()
def random_index(n, length):
    """
    Generate an array of random indices.

    Parameters:
    - n (int): The number of indices to generate.
    - length (int): The maximum value for the indices.

    Returns:
    - index (ndarray): An array of random indices.

    """
    index = np.zeros(n, dtype=np.int64)
    for i in range(n):
        index[i] = np.random.randint(0, length)
    return index


@nb.jit("i8(f8[:],i8)", nopython=True)
def update_state(state, increment):
    """
    Update the state by incrementing the value at index 0.

    Args:
        state (list): The state to be updated.
        increment (int): The value to increment the state by.

    Returns:
        int: The original value at index 0 of the state.

    Raises:
        ValueError: If the state size is smaller than the increment value.

    """
    idx = int(state[0])
    if idx == 0:
        idx = 1
    if state.size < +increment:
        raise ValueError("state too small")
        # state = state.resize(state.size + BLOCK_SIZE, refcheck=False)
    state[0] = idx + increment
    return idx


@nb.njit()
def median(state, t, z, tau_int, inter=0, time_weighted=0):
    delta = np.zeros(tau_int, dtype=float)
    if time_weighted:
        _tau_int = int(tau_int + 1)
    else:
        _tau_int = int(tau_int)

    t_slice = state[0:_tau_int]
    z_slice = state[_tau_int : (2 * _tau_int)]
    for j in range(_tau_int - 1, 0, -1):
        t_slice[j] = t_slice[j - 1]
        z_slice[j] = z_slice[j - 1]
        delta[j] = t_slice[j] - t_slice[j - 1]
    t_slice[0] = t
    z_slice[0] = z
    if t_slice[_tau_int - 1] == 0:
        return t, z
    else:
        return t, np.median(z_slice)


@nb.jit("Tuple((f8, f8))(f8[:], f8, f8)", nopython=True)
def tick(state, t, z):
    """
    Update the state with the given time and value, and calculate the difference in time between the current and previous tick.

    Args:
        state (list): A list containing the previous time and value.
        t (float): The current time.
        z (float): The current value.

    Returns:
        tuple: A tuple containing the current time and the difference in time between the current and previous tick.
    """
    if state[0] == 0:
        state[0] = t
        state[1] = z
        return t, 0

    value = t - state[0]
    state[0] = t
    state[1] = z
    return t, value


@nb.jit("Tuple((f8, f8))(f8[:], f8, f8)", nopython=True)
def diff(state, t, z):
    """
    Calculate the difference between the current value 'z' and the previous value stored in 'state'.

    Args:
        state (list): A list containing the previous time and value.
        t (float): The current time.
        z (float): The current value.

    Returns:
        tuple: A tuple containing the current time and the difference between the current value and the previous value.
    """
    if state[0] == 0:
        state[0] = t
        state[1] = z
        return t, 0

    value = z - state[1]
    state[0] = t
    state[1] = z
    return t, value


@nb.jit("Tuple((f8, f8))(f8[:], f8, f8)", nopython=True)
def diff_slope(state, t, z):
    """
    Calculate the difference in slope between the current point and the previous point.

    Args:
        state (list): A list containing the previous point's time and value.
        t (float): The current point's time.
        z (float): The current point's value.

    Returns:
        tuple: A tuple containing the current point's time and the difference in slope.

    """
    if state[0] == 0:
        state[0] = t
        state[1] = z
        return t, 0

    value = (z - state[+1]) / (t - state[+0])
    state[0] = t
    state[1] = z
    return t, value


@nb.jit("Tuple((f8, f8))(f8[:], f8, f8)", nopython=True)
def log_return(state, t, z):
    """
    Calculate the logarithmic return based on the given state, time, and value.

    Args:
        state (list): A list containing the previous time and value.
        t (float): The current time.
        z (float): The current value.

    Returns:
        tuple: A tuple containing the current time and the calculated logarithmic return.
    """
    if state[0] == 0:
        state[0] = t
        state[1] = z
        return t, 0

    value = math.log(z / state[+1])
    state[0] = t
    state[1] = z
    return t, value


@nb.jit("Tuple((f8, f8))(f8,i8,f8[:], f8, f8)", nopython=True)
def ema(tau, inter, state, t, z):
    """
    Calculates the exponential moving average (EMA) based on the given parameters.

    Parameters:
    - tau (float): The time constant for the EMA calculation.
    - inter (int): The interpolation method to use (-1, 0, or 1).
    - state (list): A list containing the state values for the EMA calculation.
    - t (float): The current time value.
    - z (float): The current input value.

    Returns:
    - tuple: A tuple containing the updated time value and the calculated EMA value.
    """

    if state[1] == 0:
        state[0] = z
        state[1] = t
        state[2] = z
        return t, state[0]

    delta = t - state[1]
    # ooo event
    if delta < 0:
        # return previous state
        return state[1], state[0]
    if delta == 0:
        # update previous value
        state[2] = (state[2] + z) / 2
        return state[1], state[0]
    alpha = delta / tau
    mu = math.exp(-alpha)
    nu = 0
    if inter == -1:
        nu = 1
    elif inter == 0:
        nu = (1 - mu) / alpha
    elif inter == 1:
        nu = mu
    state[0] = mu * state[0] + (nu - mu) * state[2] + (1 - nu) * z
    state[1] = t
    state[2] = z
    return t, state[0]


@nb.jit("Tuple((f8, f8))(f8, i8, i8, f8[:], f8, f8)", nopython=True)
def nema(tau, inter, n, state, t, z):
    """
    Calculates the NEMA (Normalized Exponential Moving Average) value.

    Parameters:
    - tau (float): The time constant for the exponential moving average.
    - inter (int): The interval for the exponential moving average.
    - n (int): The number of iterations.
    - state (numpy.ndarray): The state containing the data.
    - t (float): The current time.
    - z (float): The previous value.

    Returns:
    - tuple: A tuple containing the current time and the NEMA value.
    """

    v = np.zeros(n, dtype=np.float64)
    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    _tau = tau / float(n)
    _, v[0] = ema(_tau, inter, state, t, z)
    for i in range(1, n):
        _, v[i] = ema(_tau, 0, state[3 * i :], t, v[i - 1])
    return t, v[-1]


@nb.jit("Tuple((f8, f8))(f8, f8[:], f8, f8)", nopython=True)
def delta(tau, state, t, z):
    """
    Calculate the delta value based on the given parameters.

    Args:
        tau (float): The tau value.
        state (list): The state containing data points.
        t (float): The t value.
        z (float): The z value.

    Returns:
        tuple: A tuple containing the calculated t value and the delta value.
    """
    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    gamma = 1.22208
    beta = 0.65
    alpha = 1 / (gamma * (8 * beta - 3))
    _, n0 = ema(alpha * tau, 0, state, t, z)
    _, n1 = ema(alpha * tau, 0, state[1 * 3 :], t, z)
    _, n1 = ema(alpha * tau, 0, state[2 * 3 :], t, n1)
    _, n2 = ema(alpha * beta * tau, 0, state[3 * 3 :], t, z)
    for i in range(4, 7):
        _, n2 = ema(alpha * beta * tau, 0, state[i * 3 :], t, n2)
    return t, gamma * (n0 + n1 - 2 * n2)


@nb.jit("Tuple((f8, f8))(f8, f8[:], f8, f8)", nopython=True)
def delta2nd(tau, state, t, z):
    """
    Calculates the second-order delta value based on the given parameters.

    Args:
        tau (float): The tau value.
        state (list): The state list.
        t (float): The t value.
        z (float): The z value.

    Returns:
        tuple: A tuple containing the updated t and z values.
    """
    _tau = tau / 2  # according to the book, shifted twice
    t, z1 = delta(_tau, state, t, z)
    t, z2 = delta(_tau, state[7 * 3 :], t, z1)
    return t, z2


@nb.jit("Tuple((f8, f8))(f8, f8[:], f8, f8, f8)", nopython=True)
def xy_slope(tau, state, t, x, y):
    """
    Calculate the slope of the y-coordinate with respect to the x-coordinate.

    Parameters:
    - tau (float): The time constant.
    - state (list): The state containing data points.
    - t (float): The current time.
    - x (float): The x-coordinate.
    - y (float): The y-coordinate.

    Returns:
    - tuple: A tuple containing the current time and the slope of the y-coordinate with respect to the x-coordinate.
    """
    _, dx = delta(tau, state, t, x)
    _, dy = delta(tau, state[7 * 3 :], t, y)
    if dx == 0:
        return t, 0
    return t, dy / dx


@nb.jit("Tuple((f8, f8))(f8, i8, i8, f8[:], f8, f8)", nopython=True)
def ma(tau, inter, n, state, t, z):
    """
    Calculates the moving average of a given time series.

    Parameters:
    - tau (float): Time constant for the exponential moving average.
    - inter (int): Interpolation method.
    - n (int): Number of data points to consider for the moving average.
    - state (numpy.ndarray): Array containing the time series data.
    - t (float): Current time.
    - z (float): Previous value of the moving average.

    Returns:
    - tuple: A tuple containing the current time and the moving average value.
    """
    v = np.zeros(n, dtype=np.float64)
    rsum = 0
    _tau = 2 * tau / (n + 1.0)
    _, v[0] = ema(_tau, inter, state, t, z)
    for i in range(1, n):
        _, v[i] = ema(_tau, 0, state[3 * i :], t, v[i - 1])
    for i in range(0, n):
        rsum += v[i]
    return t, rsum / n


@nb.jit("Tuple((f8, f8[:]))(f8, i8, i8, f8[:], f8, f8)", nopython=True)
def _msd(tau, inter, n, state, t, z):
    """
    Calculate the mean squared displacement (MSD) using the given parameters.

    Parameters:
    - tau: The tau parameter.
    - inter: The inter parameter.
    - n: The n parameter.
    - state: The state parameter.
    - t: The t parameter.
    - z: The z parameter.

    Returns:
    - t: The t parameter.
    - v: The calculated mean squared displacement (MSD) values.
    """
    v = np.zeros(3, dtype=np.float64)
    _, v[0] = ma(tau, inter, n, state, t, z)
    tmp = (v[0] - z) * (v[0] - z)
    _, v[1] = ma(tau, inter, n, state[3 * n :], t, tmp)
    v[2] = math.sqrt(v[1])
    return t, v


@nb.jit("Tuple((f8, f8))(f8, i8, i8, f8[:], f8, f8)", nopython=True)
def msd(tau, inter, n, state, t, z):
    """
    Calculate the mean standard deviation (MSD) for a given set of parameters.

    Parameters:
    - tau: The tau parameter.
    - inter: The inter parameter.
    - n: The n parameter.
    - state: The state parameter.
    - t: The t parameter.
    - z: The z parameter.

    Returns:
    - tuple: A tuple containing the time values and the MSD values.

    """
    _, v = _msd(tau, inter, n, state, t, z)
    return t, v[2]


@nb.jit(nopython=True)
def zscore(tau, inter, n, state, t, z):
    """
    Calculate the z-score of a given value.

    Parameters:
    - tau (float): Time constant for the moving average calculation.
    - inter (float): Interval between data points.
    - n (int): Number of data points to consider for the moving average.
    - state (list): List of states containing previous data points.
    - t (float): Current time.
    - z (float): Value for which to calculate the z-score.

    Returns:
    - tuple: A tuple containing the current time and the calculated z-score.
    """

    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    v = np.zeros(3, dtype=np.float64)
    _, v[0] = ma(tau, inter, n, state, t, z)
    tmp = (v[0] - z) * (v[0] - z)
    _, v[1] = ma(tau, inter, n, state[n * 3 :], t, tmp)
    v[2] = math.sqrt(v[1])
    if v[2] == 0:
        return t, 0
    return t, (z - v[0]) / v[2]


@nb.jit(nopython=True)
def skewness(tau, inter, n, state, t, z):
    """
    Calculate the skewness of a given time series.

    Parameters:
    - tau (float): Time constant for the moving average calculation.
    - inter (float): Interval between data points.
    - n (int): Number of data points to consider for the moving average.
    - state (list): List of states containing previous data points.
    - t (float): Current time.
    - z (float): Value for which to calculate the skewness.

    Returns:
    - tuple: A tuple containing the current time and the calculated skewness value.
    """
    _, zm = zscore(tau, inter, n, state, t, z)
    tmp = math.pow(zm, 3)
    tn, zn = ma(tau, inter, n, state[n * 6 :], t, tmp)
    return tn, zn


def kurtosis(tau, inter, n, state, t, z):
    """
    Calculate the kurtosis of a given time series.

    Parameters:
    - tau (float): Time constant for the moving average calculation.
    - inter (float): Interval between data points.
    - n (int): Number of data points to consider for the moving average.
    - state (list): List of states containing previous data points.
    - t (float): Current time.
    - z (float): Value for which to calculate the kurtosis.

    Returns:
    - tuple: A tuple containing the current time and the calculated kurtosis value.
    """
    _, zm = zscore(tau, inter, n, state, t, z)
    tmp = math.pow(zm, 4)
    tn, zn = ma(tau, inter, n, state[n * 6 :], t, tmp)
    return tn, zn


@nb.jit(nopython=True)
def _cor(tau, inter, n, state, t, x, y):
    """
    Calculate the correlation between two time series.

    Parameters:
    - tau (float): Time constant for the moving average.
    - inter (float): Interval between data points.
    - n (int): Number of data points.
    - state (numpy.ndarray): state containing the data.
    - t (float): Current time.
    - x (numpy.ndarray): First time series.
    - y (numpy.ndarray): Second time series.

    Returns:
    - tuple: A tuple containing the current time and an array of correlation values.
    """

    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    v = np.zeros(7, dtype=np.float64)
    _, v[0:3] = _msd(tau, inter, n, state, t, x)
    _, v[3:6] = _msd(tau, inter, n, state[3 * 2 * n :], t, y)
    tmp = (x - v[0]) * (y - v[3])
    _, ma_xy = ma(tau, inter, n, state[3 * 4 * n :], t, tmp)
    tmp = v[2] * v[5]
    if tmp == 0:
        v[6] = 0
    else:
        v[6] = ma_xy / (tmp)
    return t, v


@nb.jit(nopython=True)
def cor(tau, inter, n, state, t, x, y):
    """
    Calculate the correlation between two signals.

    Parameters:
    tau (float): Time delay between the two signals.
    inter (float): Interpolation factor.
    n (int): Number of samples.
    state (list): state containing the signals.
    t (float): Time value.
    x (float): Signal x.
    y (float): Signal y.

    Returns:
    tuple: A tuple containing the time value and the correlation value.
    """

    _, v = _cor(tau, inter, n, state, t, x, y)
    return t, v[6]


@nb.jit(nopython=True)
def cor2(tau, inter, n, state, t, x, y):
    """
    Calculates the correlation between two time series.

    Parameters:
    - tau (float): Time constant for the moving average.
    - inter (float): Interval between data points.
    - n (int): Number of data points.
    - state (list): List of states containing data points.
    - t (float): Current time.
    - x (float): Value of the first time series at time t.
    - y (float): Value of the second time series at time t.

    Returns:
    - tuple: A tuple containing the current time (t) and the calculated correlation value (v[6]).
    """

    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    v = np.zeros(7, dtype=np.float64)
    _, v[0:3] = _msd(tau, inter, n, state, t, x)
    _, v[3:6] = _msd(tau, inter, n, state[2 * n * 3 :], t, y)
    if (v[2] * v[5]) == 0:
        # todo: this might not be a good choice
        tmp = (x - v[0]) * (y - v[3]) / 1
    else:
        tmp = (x - v[0]) * (y - v[3]) / (v[2] * v[5])
    _, v[6] = ma(tau, inter, n, state[4 * n * 3], t, tmp)
    return t, v[6]


@nb.jit(nopython=True)
def linear_regression(tau, inter, n, state, t, x, y):
    """
    Perform linear regression on the given data points.

    Parameters:
    - tau (float): The time constant.
    - inter (float): The interpolation factor.
    - n (int): The number of data points.
    - state (numpy.ndarray): The state containing the data points.
    - t (float): The current time.
    - x (float): The x-coordinate of the data point.
    - y (float): The y-coordinate of the data point.

    Returns:
    - tuple: A tuple containing the current time and the calculated beta value.
    """
    r = np.empty(5, dtype=np.float64)
    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    _, v = _cor(tau, inter, n, state, t, x, y)
    if v[2] == 0:
        beta = 0
    else:
        beta = v[6] * v[5] / v[2]
    alpha = v[3] - (beta * v[0])
    _delta = y - (alpha + beta * x)
    _, mse = ma(tau, inter, n, state[5 * n * 3 :], t, _delta * _delta)
    rmse = math.sqrt(mse)
    r[4] = v[6]
    r[3] = rmse
    r[2] = mse
    r[1] = beta
    r[0] = alpha
    return t, beta


@nb.jit(nopython=True)
def linear_regression2(tau, inter, n, state, t, x, y, index=0):
    """
    Perform linear regression on the given data.

    Args:
        tau (float): The time constant for the moving average.
        inter (float): The interpolation factor for the moving average.
        n (int): The number of samples for the moving average.
        state (list): A list of states for intermediate calculations.
        t (float): The time value.
        x (float): The input data.
        y (float): The output data.
        - index (int, optional):
        The index of the result array to store the calculated beta value. Defaults to 4.
        0 = alpha
        1 = beta
        2 = mse
        3 = rmse

    Returns:
        tuple: A tuple containing the time value and the calculated beta value.

    """
    r = np.empty(4)
    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    _, mx = ma(tau, inter, n, state, t, x)
    _, my = ma(tau, inter, n, state[1 * n * 3 :], t, y)
    _, mx_xy = ma(tau, inter, n, state[2 * n * 3 :], t, (x - mx) * (y - my))
    _, mx_x2 = ma(tau, inter, n, state[3 * n * 3 :], t, (x - mx) * (x - mx))
    if mx_x2 > 0:
        beta = mx_xy / mx_x2
        alpha = my - beta * mx
    else:
        beta = np.nan
        alpha = np.nan

    delta = y - (alpha + beta * x)
    _, mse = ma(tau, inter, n, state[4 * n * 3 :], t, delta * delta)
    rmse = math.sqrt(mse)
    r[3] = rmse
    r[2] = mse
    r[1] = beta
    r[0] = alpha
    return t, r[index]


@nb.jit(nopython=True)
def downsample(state, t, z, delta):
    # state[0] = tp, state[1] = zp, state[2] = count
    n = math.floor(t / delta)
    tn = n * delta
    if state[0] == 0:
        state[0] = tn
        state[1] = z
        state[2] = 0
        if abs(t - tn) < 1e-6:
            return tn, z
        return 0, 0

    if tn == state[0]:
        return 0, 0

    zn = np.interp(tn, [state[0], t], [state[1], z], left=state[1], right=z)

    state[0] = tn
    state[1] = z
    state[2] = 0
    return tn, zn


@nb.jit(
    "Tuple((f8, f8))(f8, f8, optional(f8), optional(f8), optional(f8), optional(f8), optional(f8), optional(f8))",
    nopython=True,
)
def limit(t, z, lo, hi, lolo, hihi, lololo, hihihi):
    _z = 0
    if lo is not None and z < lo:
        _z = -1
    if hi is not None and z > hi:
        _z = 1
    if lolo is not None and z < lolo:
        _z = -2
    if hihi is not None and z > hihi:
        _z = 2
    if lololo is not None and z < lololo:
        _z = -3
    if hihihi is not None and z > hihihi:
        _z = 3
    return t, _z


@nb.jit(nopython=True)
def outlier(state, t, z, max_diff, max_count):
    """
    Determines if a data point is an outlier based on the given state and thresholds.

    Args:
        state (list): A list containing the previous data point's timestamp, value, and count.
        t (float): The timestamp of the current data point.
        z (float): The value of the current data point.
        max_diff (float): The maximum difference allowed between the current and previous data point.
        max_count (int): The maximum number of consecutive outliers allowed.

    Returns:
        tuple: A tuple containing the updated timestamp and value based on the outlier detection.

    """
    if state[0] == 0:
        state[0] = t
        state[1] = z
        state[2] = 0
        return t, z

    if abs(z - state[1]) > max_diff and state[2] < max_count:
        state[0] = t
        state[2] += 1
        return t, state[1]

    state[0] = t
    state[1] = z
    state[2] = 0
    return t, z


@nb.jit(nopython=True)
def outlier_slope(state, t, z, max_slope, max_count):
    """
    Checks if the current data point is an outlier based on the slope of the data points.

    Args:
        state (list): A list containing the previous data point information.
        t (float): The current time value.
        z (float): The current data value.
        max_slope (float): The maximum allowable slope between consecutive data points.
        max_count (int): The maximum number of consecutive outliers allowed.

    Returns:
        tuple: A tuple containing the updated time value and the corresponding data value.

    """
    # state[0] = tp, state[1] = zp
    if state[0] == 0:
        state[0] = t
        state[1] = z
        state[2] = 0
        return t, z

    if abs((z - state[1]) / (t - state[0])) > max_slope and state[2] <= max_count:
        state[0] = t
        state[2] += 1
        return t, state[1]

    state[0] = t
    state[1] = z
    state[2] = 0
    return t, z
