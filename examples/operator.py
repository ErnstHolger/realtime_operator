import math
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from realtime_operator.single_operator import create_sinusoid
import matplotlib.pyplot as plt


if __name__ == "__main__":

    t = np.arange(1, 100, 1).astype(float)
    z = create_sinusoid(1000, 1, 0, 1)

    # Plot the timeseries
    plt.plot(t, z)
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.title("Timeseries Plot")
    plt.show()

    # Example usage
