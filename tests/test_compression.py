import numpy as np
import pytest
import math

from compression import (
    timedelta_min,
    exception_deviation,
    pi_exception_deviation,
    swinging_door
)

t = np.arange(1,20,dtype=float)+1
z= np.arange(1,20,dtype=float)
buffer = np.zeros(4,dtype=float)

def test_timedelta_min():
    n=len(z)
    buffer = np.zeros(2, dtype=float)
    result=[]
    for i in range(n):
        _, zn, _ = timedelta_min(3, buffer, t[i], z[i])
        for i in zn:
            result.append(i)

    desired_array = np.array([1., 4., 7., 10., 13., 16., 19.])
    assert(np.all(np.array(result) == desired_array))

def test_exception_deviation():
    n=len(z)
    buffer = np.zeros(2, dtype=float)
    result=[]
    for i in range(n):
        _, zn, _ = exception_deviation(3, buffer, t[i], z[i])
        for i in zn:
            result.append(i)

    desired_array = np.array([1., 4., 7., 10., 13., 16., 19.])
    assert(np.all(np.array(result) == desired_array))

def test_pi_exception_deviation():
    n=len(z)
    buffer = np.zeros(2, dtype=float)
    result=[]
    for i in range(n):
        _, zn, _ = pi_exception_deviation(4, buffer, t[i], z[i])
        for i in zn:
            result.append(i)

    desired_array = np.array([1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0, 16.0, 17.0])
    assert(np.all(np.array(result) == desired_array))

def test_swinging_door():
    zp=[1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1]
    n=len(zp)
    buffer = np.zeros(6, dtype=float)
    result=[]
    for i in range(n):
        _, zn, _ = swinging_door(4, buffer, t[i], zp[i])
        for i in zn:
            result.append(i)

    desired_array = np.array([1.0, 10.0])
    assert(np.all(np.array(result) == desired_array))