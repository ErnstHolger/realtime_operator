import math
import numba as nb
import numpy as np

BLOCK_SIZE = 512
START_TIME="2024-01-01 08:00:00"


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

def create_sinusoid(n,step, amplitude, shift, frequency):
    start=np.datetime64(START_TIME)
    t = np.arange(0,n,step)+start
    z = amplitude * np.sin(2 * np.pi * frequency * (t + shift))
    return t,z

@nb.jit("i8(f8[:],i8)", nopython=True)
def update_buffer(buffer, increment):
    """
    Update the buffer by incrementing the value at index 0.

    Args:
        buffer (list): The buffer to be updated.
        increment (int): The value to increment the buffer by.

    Returns:
        int: The original value at index 0 of the buffer.

    Raises:
        ValueError: If the buffer size is smaller than the increment value.

    """
    idx = int(buffer[0])
    if idx == 0:
        idx = 1
    if buffer.size < +increment:
        raise ValueError("Buffer too small")
        # buffer = buffer.resize(buffer.size + BLOCK_SIZE, refcheck=False)
    buffer[0] = idx + increment
    return idx


@nb.jit("Tuple((f8, f8))(f8[:], f8, f8)", nopython=True)
def tick(buffer, t, z):
    """
    Update the buffer with the given time and value, and calculate the difference in time between the current and previous tick.

    Args:
        buffer (list): A list containing the previous time and value.
        t (float): The current time.
        z (float): The current value.

    Returns:
        tuple: A tuple containing the current time and the difference in time between the current and previous tick.
    """
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        return t, 0

    value = t - buffer[0]
    buffer[0] = t
    buffer[1] = z
    return t, value


@nb.jit("Tuple((f8, f8))(f8[:], f8, f8)", nopython=True)
def diff(buffer, t, z):
    """
    Calculate the difference between the current value 'z' and the previous value stored in 'buffer'.

    Args:
        buffer (list): A list containing the previous time and value.
        t (float): The current time.
        z (float): The current value.

    Returns:
        tuple: A tuple containing the current time and the difference between the current value and the previous value.
    """
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        return t, 0

    value = z - buffer[1]
    buffer[0] = t
    buffer[1] = z
    return t, value


@nb.jit("Tuple((f8, f8))(f8[:], f8, f8)", nopython=True)
def diff_slope(buffer, t, z):
    """
    Calculate the difference in slope between the current point and the previous point.

    Args:
        buffer (list): A list containing the previous point's time and value.
        t (float): The current point's time.
        z (float): The current point's value.

    Returns:
        tuple: A tuple containing the current point's time and the difference in slope.

    """
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        return t, 0

    value = (z - buffer[+1]) / (t - buffer[+0])
    buffer[0] = t
    buffer[1] = z
    return t, value


@nb.jit("Tuple((f8, f8))(f8[:], f8, f8)", nopython=True)
import math

def log_return(buffer, t, z):
    """
    Calculate the logarithmic return based on the given buffer, time, and value.

    Args:
        buffer (list): A list containing the previous time and value.
        t (float): The current time.
        z (float): The current value.

    Returns:
        tuple: A tuple containing the current time and the calculated logarithmic return.
    """
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        return t, 0

    value = math.log(z / buffer[+1])
    buffer[0] = t
    buffer[1] = z
    return t, value


@nb.jit("Tuple((f8, f8))(f8,i8,f8[:], f8, f8)", nopython=True)
def ema(tau, inter, buffer, t, z):
    """
    Calculates the exponential moving average (EMA) based on the given parameters.

    Parameters:
    - tau (float): The time constant for the EMA calculation.
    - inter (int): The interpolation method to use (-1, 0, or 1).
    - buffer (list): A list containing the buffer values for the EMA calculation.
    - t (float): The current time value.
    - z (float): The current input value.

    Returns:
    - tuple: A tuple containing the updated time value and the calculated EMA value.
    """

    if buffer[1] == 0:
        buffer[0] = z
        buffer[1] = t
        buffer[2] = z
        return t, buffer[+0]

    delta = t - buffer[+1]
    if delta == 0:
        delta = 1e-6
    alpha = delta / tau
    mu = math.exp(-alpha)
    nu = 0
    if inter == -1:
        nu = 1
    elif inter == 0:
        nu = (1 - mu) / alpha
    elif inter == 1:
        nu = mu
    buffer[+0] = mu * buffer[+0] + (nu - mu) * buffer[+2] + (1 - nu) * z
    buffer[+1] = t
    buffer[+2] = z
    return t, buffer[+0]


@nb.jit("Tuple((f8, f8))(f8, i8, i8, f8[:], f8, f8)", nopython=True)
def nema(tau, inter, n, buffer, t, z):
    """
    Calculates the NEMA (Normalized Exponential Moving Average) value.

    Parameters:
    - tau (float): The time constant for the exponential moving average.
    - inter (int): The interval for the exponential moving average.
    - n (int): The number of iterations.
    - buffer (numpy.ndarray): The buffer containing the data.
    - t (float): The current time.
    - z (float): The previous value.

    Returns:
    - tuple: A tuple containing the current time and the NEMA value.
    """

    v = np.zeros(n, dtype=np.float64)
    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    _tau = tau / float(n)
    _, v[0] = ema(_tau, inter, buffer[0:], t, z)
    for i in range(1, n):
        _, v[i] = ema(_tau, 0, buffer[3 * i :], t, v[i - 1])
    return t, v[-1]


@nb.jit("Tuple((f8, f8))(f8, f8[:], f8, f8)", nopython=True)
def delta(tau, buffer, t, z):
    """
    Calculate the delta value based on the given parameters.

    Args:
        tau (float): The tau value.
        buffer (list): The buffer containing data points.
        t (float): The t value.
        z (float): The z value.

    Returns:
        tuple: A tuple containing the calculated t value and the delta value.
    """
    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    gamma = 1.22208
    beta = 0.65
    alpha = 1 / (gamma * (8 * beta - 3))
    _, n0 = ema(alpha * tau, 0, buffer[0:], t, z)
    _, n1 = ema(alpha * tau, 0, buffer[1 * 3 :], t, z)
    _, n1 = ema(alpha * tau, 0, buffer[2 * 3 :], t, n1)
    _, n2 = ema(alpha * beta * tau, 0, buffer[3 * 3 :], t, z)
    for i in range(4, 7):
        _, n2 = ema(alpha * beta * tau, 0, buffer[i * 3 :], t, n2)
    return t, gamma * (n0 + n1 - 2 * n2)


@nb.jit("Tuple((f8, f8))(f8, f8[:], f8, f8)", nopython=True)
def delta2nd(tau, buffer, t, z):
    """
    Calculates the second-order delta value based on the given parameters.

    Args:
        tau (float): The tau value.
        buffer (list): The buffer list.
        t (float): The t value.
        z (float): The z value.

    Returns:
        tuple: A tuple containing the updated t and z values.
    """
    _tau = tau / 2  # according to the book, shifted twice
    t, z1 = delta(_tau, buffer[0:], t, z)
    t, z2 = delta(_tau, buffer[7 * 3 :], t, z1)
    return t, z2


@nb.jit("Tuple((f8, f8))(f8, f8[:], f8, f8, f8)", nopython=True)
def xy_slope(tau, buffer, t, x, y):
    """
    Calculate the slope of the y-coordinate with respect to the x-coordinate.

    Parameters:
    - tau (float): The time constant.
    - buffer (list): The buffer containing data points.
    - t (float): The current time.
    - x (float): The x-coordinate.
    - y (float): The y-coordinate.

    Returns:
    - tuple: A tuple containing the current time and the slope of the y-coordinate with respect to the x-coordinate.
    """
    _, dx = delta(tau, buffer[0:], t, x)
    _, dy = delta(tau, buffer[7 * 3 :], t, y)
    if dx == 0:
        return t, 0
    return t, dy / dx


@nb.jit("Tuple((f8, f8))(f8, i8, i8, f8[:], f8, f8)", nopython=True)
def ma(tau, inter, n, buffer, t, z):
    """
    Calculates the moving average of a given time series.

    Parameters:
    - tau (float): Time constant for the exponential moving average.
    - inter (int): Interpolation method.
    - n (int): Number of data points to consider for the moving average.
    - buffer (numpy.ndarray): Array containing the time series data.
    - t (float): Current time.
    - z (float): Previous value of the moving average.

    Returns:
    - tuple: A tuple containing the current time and the moving average value.
    """
    v = np.zeros(n, dtype=np.float64)
    rsum = 0
    _tau = 2 * tau / (n + 1.0)
    _, v[0] = ema(_tau, inter, buffer, t, z)
    for i in range(1, n):
        _, v[i] = ema(_tau, 0, buffer[3 * n :], t, v[i - 1])
    for i in range(0, n):
        rsum += v[i]
    return t, rsum / n


@nb.jit("Tuple((f8, f8[:]))(f8, i8, i8, f8[:], f8, f8)", nopython=True)
def _msd(tau, inter, n, buffer, t, z):
    """
    Calculate the mean squared displacement (MSD) using the given parameters.

    Parameters:
    - tau: The tau parameter.
    - inter: The inter parameter.
    - n: The n parameter.
    - buffer: The buffer parameter.
    - t: The t parameter.
    - z: The z parameter.

    Returns:
    - t: The t parameter.
    - v: The calculated mean squared displacement (MSD) values.
    """
    v = np.zeros(3, dtype=np.float64)
    _, v[0] = ma(tau, inter, n, buffer, t, z)
    tmp = (v[0] - z) * (v[0] - z)
    _, v[1] = ma(tau, inter, n, buffer[3 * n :], t, tmp)
    v[2] = math.sqrt(v[1])
    return t, v


@nb.jit("Tuple((f8, f8))(f8, i8, i8, f8[:], f8, f8)", nopython=True)
def msd(tau, inter, n, buffer, t, z):
    """
    Calculate the mean standard deviation (MSD) for a given set of parameters.

    Parameters:
    - tau: The tau parameter.
    - inter: The inter parameter.
    - n: The n parameter.
    - buffer: The buffer parameter.
    - t: The t parameter.
    - z: The z parameter.

    Returns:
    - tuple: A tuple containing the time values and the MSD values.

    """
    _, v = _msd(tau, inter, n, buffer, t, z)
    return t, v[2]


@nb.jit(nopython=True)
def zscore(tau, inter, n, buffer, t, z):
    """
    Calculate the z-score of a given value.

    Parameters:
    - tau (float): Time constant for the moving average calculation.
    - inter (float): Interval between data points.
    - n (int): Number of data points to consider for the moving average.
    - buffer (list): List of buffers containing previous data points.
    - t (float): Current time.
    - z (float): Value for which to calculate the z-score.

    Returns:
    - tuple: A tuple containing the current time and the calculated z-score.
    """

    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    v = np.zeros(3, dtype=np.float64)
    _, v[0] = ma(tau, inter, n, buffer[0], t, z)
    tmp = (v[0] - z) * (v[0] - z)
    _, v[1] = ma(tau, inter, n, buffer[1], t, tmp)
    v[2] = math.sqrt(v[1])
    return t, (z - v[0]) / v[2]


@nb.jit(nopython=True)
def _cor(tau, inter, n, buffer, t, x, y):
    """
    Calculate the correlation between two time series.

    Parameters:
    - tau (float): Time constant for the moving average.
    - inter (float): Interval between data points.
    - n (int): Number of data points.
    - buffer (numpy.ndarray): Buffer containing the data.
    - t (float): Current time.
    - x (numpy.ndarray): First time series.
    - y (numpy.ndarray): Second time series.

    Returns:
    - tuple: A tuple containing the current time and an array of correlation values.
    """

    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    v = np.zeros(7, dtype=np.float64)
    _, v[0:3] = _msd(tau, inter, n, buffer, t, x)
    _, v[3:6] = _msd(tau, inter, n, buffer[3 * 2 * n :], t, y)
    tmp = (x - v[0]) * (y - v[3])
    _, ma_xy = ma(tau, inter, n, buffer[3 * 3 * n :], t, tmp)
    tmp = v[2] * v[5]
    if tmp == 0:
        v[6] = 0
    else:
        v[6] = ma_xy / (tmp)
    return t, v


@nb.jit(nopython=True)
def cor(tau, inter, n, buffer, t, x, y):
    """
    Calculate the correlation between two signals.

    Parameters:
    tau (float): Time delay between the two signals.
    inter (float): Interpolation factor.
    n (int): Number of samples.
    buffer (list): Buffer containing the signals.
    t (float): Time value.
    x (float): Signal x.
    y (float): Signal y.

    Returns:
    tuple: A tuple containing the time value and the correlation value.
    """

    _, v = _cor(tau, inter, n, buffer, t, x, y)
    return t, v[6]


@nb.jit(nopython=True)
def cor2(tau, inter, n, buffer, t, x, y):
    """
    Calculates the correlation between two time series.

    Parameters:
    - tau (float): Time constant for the moving average.
    - inter (float): Interval between data points.
    - n (int): Number of data points.
    - buffer (list): List of buffers containing data points.
    - t (float): Current time.
    - x (float): Value of the first time series at time t.
    - y (float): Value of the second time series at time t.

    Returns:
    - tuple: A tuple containing the current time (t) and the calculated correlation value (v[6]).
    """

    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    v = np.zeros(7, dtype=np.float64)
    _, v[0:3] = _msd(tau, inter, n, buffer[0:2], t, x)
    _, v[3:6] = _msd(tau, inter, n, buffer[2:4], t, y)
    if (v[2] * v[5]) == 0:
        # todo: this might not be a good choice
        tmp = (x - v[0]) * (y - v[3]) / 1
    else:
        tmp = (x - v[0]) * (y - v[3]) / (v[2] * v[5])
    _, v[6] = ma(tau, inter, n, buffer[4], t, tmp)
    return t, v[6]





@nb.jit(nopython=True)
def linear_regression(tau, inter, n, buffer, t, x, y):
    """
    Perform linear regression on the given data points.

    Parameters:
    - tau (float): The time constant.
    - inter (float): The interpolation factor.
    - n (int): The number of data points.
    - buffer (numpy.ndarray): The buffer containing the data points.
    - t (float): The current time.
    - x (float): The x-coordinate of the data point.
    - y (float): The y-coordinate of the data point.

    Returns:
    - tuple: A tuple containing the current time and the calculated beta value.
    """
    r = np.empty(5, dtype=np.float64)
    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    _, v = _cor(tau, inter, n, buffer[0:5], t, x, y)
    if v[2] == 0:
        beta = 0
    else:
        beta = v[6] * v[5] / v[2]
    alpha = v[3] - (beta * v[0])
    _delta = y - (alpha + beta * x)
    _, mse = ma(tau, inter, n, buffer[6], t, _delta * _delta)
    rmse = math.sqrt(mse)
    r[4] = v[6]
    r[3] = rmse
    r[2] = mse
    r[1] = beta
    r[0] = alpha
    return t, beta


@nb.jit(nopython=True)
def linear_regression2(tau, inter, n, buffer, t, x, y):
    """
    Perform linear regression on the given data.

    Args:
        tau (float): The time constant for the moving average.
        inter (float): The interpolation factor for the moving average.
        n (int): The number of samples for the moving average.
        buffer (list): A list of buffers for intermediate calculations.
        t (float): The time value.
        x (float): The input data.
        y (float): The output data.

    Returns:
        tuple: A tuple containing the time value and the calculated beta value.

    """
    r = np.empty(4)
    # ema[0]=ema | ema[1]=tp | ema[2]=zp
    _, mx = ma(tau, inter, n, buffer[0], t, x)
    _, my = ma(tau, inter, n, buffer[1], t, y)
    _, mx_xy = ma(tau, inter, n, buffer[2], t, (x - mx) * (y - my))
    _, mx_x2 = ma(tau, inter, n, buffer[3], t, (x - mx) * (x - mx))
    if mx_x2 > 0:
        beta = mx_xy / mx_x2
        alpha = my - beta * mx
    else:
        beta = np.NAN
        alpha = np.NAN

    delta = y - (alpha + beta * x)
    _, mse = ma(tau, inter, n, buffer[4], t, delta * delta)
    rmse = math.sqrt(mse)
    r[3] = rmse
    r[2] = mse
    r[1] = beta
    r[0] = alpha
    return t, beta


@nb.jit(nopython=True)
def downsample(buffer, t, z, delta):

    # buffer[0] = tp, buffer[1] = zp, buffer[2] = count
    n = math.floor(t / delta)
    tn = n * delta
    if buffer[0] == 0:
        buffer[0] = tn
        buffer[1] = z
        buffer[2] = 0
        if abs(t - tn) < 1e-6:
            return tn, z
        return 0, 0

    if tn == buffer[0]:
        return 0, 0

    zn = np.interp(tn, [buffer[0], t], [buffer[1], z], left=buffer[1], right=z)

    buffer[0] = tn
    buffer[1] = z
    buffer[2] = 0
    return tn, zn


@nb.jit(nopython=True)
def outlier(buffer, t, z, max_diff, max_count):
    """
    Determines if a data point is an outlier based on the given buffer and thresholds.

    Args:
        buffer (list): A list containing the previous data point's timestamp, value, and count.
        t (float): The timestamp of the current data point.
        z (float): The value of the current data point.
        max_diff (float): The maximum difference allowed between the current and previous data point.
        max_count (int): The maximum number of consecutive outliers allowed.

    Returns:
        tuple: A tuple containing the updated timestamp and value based on the outlier detection.

    """
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        buffer[2] = 0
        return t, z

    if abs(z - buffer[1]) > max_diff and buffer[2] < max_count:
        buffer[0] = t
        buffer[2] += 1
        return t, buffer[1]

    buffer[0] = t
    buffer[1] = z
    buffer[2] = 0
    return t, z


@nb.jit(nopython=True)
def outlier_slope(buffer, t, z, max_slope, max_count):
    """
    Checks if the current data point is an outlier based on the slope of the data points.

    Args:
        buffer (list): A list containing the previous data point information.
        t (float): The current time value.
        z (float): The current data value.
        max_slope (float): The maximum allowable slope between consecutive data points.
        max_count (int): The maximum number of consecutive outliers allowed.

    Returns:
        tuple: A tuple containing the updated time value and the corresponding data value.

    """
    # buffer[0] = tp, buffer[1] = zp
    if buffer[0] == 0:
        buffer[0] = t
        buffer[1] = z
        buffer[2] = 0
        return t, z

    if abs((z - buffer[1]) / (t - buffer[0])) > max_slope and buffer[2] <= max_count:
        buffer[0] = t
        buffer[2] += 1
        return t, buffer[1]

    buffer[0] = t
    buffer[1] = z
    buffer[2] = 0
    return t, z
