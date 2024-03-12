# ReadMe for Time Series Analysis and Signal Processing Library

## Overview

This library offers a comprehensive suite of tools for time series analysis and signal processing, leveraging the power of NumPy and Numba for high performance computations. It includes functionalities for datetime manipulation, signal generation, buffer updates, moving averages, and much more. Designed for researchers and engineers working in data analysis, finance, or signal processing, this library streamlines complex numerical computations and analyses.

## Features

- **Datetime Conversion**: Convert between numpy.datetime64 objects and seconds since the Unix epoch.
- **Signal Generation**: Create sinusoidal signals with specified parameters.
- **Buffer Management**: Efficiently manage data buffers with functions to update, tick, and perform operations on buffered data.
- **Moving Averages**: Calculate various forms of moving averages, including Exponential Moving Average (EMA) and Normalized EMA (NEMA).
- **Difference and Slope Calculations**: Compute differences, slopes, and logarithmic returns from sequential data points.
- **Statistical Functions**: Implementations of z-score, mean squared displacement (MSD), and correlation calculations.
- **Linear Regression**: Perform linear regression on data points to find relationships between variables.
- **Outlier Detection**: Identify outliers based on value differences, slope, and other criteria.
- **Data Downsampling**: Reduce data resolution while preserving the integrity of the original signal.

## Installation

To use this library, ensure that you have Python installed on your system along with the `numpy` and `numba` packages. You can install these dependencies using pip:

```sh
pip install numpy numba
```

## Usage

### Importing the Library

```python
import numpy as np
import numba as nb
from your_library_name import *  # Replace 'your_library_name' with the name of this library file
```

### Converting Datetimes

```python
dt64 = np.datetime64('2024-01-01 08:00:00')
seconds = to_utc_seconds(dt64)
print(f"UTC Seconds: {seconds}")

dt64_back = from_utc_seconds(seconds)
print(f"Back to Datetime: {dt64_back}")
```

### Generating a Sinusoidal Signal

```python
n = 100  # Number of points
step = 1  # Step size
amplitude = 1.0  # Amplitude of the sinusoid
shift = 0  # Phase shift
frequency = 0.1  # Frequency of the sinusoid

times, values = create_sinusoid(n, step, amplitude, shift, frequency)
```

### Updating a Buffer and Calculating EMA

```python
buffer = np.zeros(10)  # Example buffer
increment = 5

original_value = update_buffer(buffer, increment)
print(f"Original value: {original_value}")

tau = 10.0  # Time constant for EMA
inter = 0  # Interpolation method
t = 1.0  # Current time
z = 2.0  # Current value

time, ema_value = ema(tau, inter, buffer, t, z)
print(f"EMA Value: {ema_value}")
```

### Performing Linear Regression

```python
tau = 10.0
inter = 0
n = 100
buffer = np.zeros((15, 3))  # Example buffer setup
t = 1.0
x = np.random.rand(100)
y = 2.0 * x + np.random.normal(size=100)  # Generating a simple linear relationship with noise

beta = linear_regression(tau, inter, n, buffer.flatten(), t, x, y)
print(f"Beta Value: {beta}")
```

## Contributions

Contributions to this library are welcome. Please ensure that any pull requests or issues adhere to the project's coding standards and provide tests for new functionalities.

## License

This library is licensed under the MIT License. See the LICENSE file for more details.

