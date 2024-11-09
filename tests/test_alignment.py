import numpy as np
import time
import sys
import pytest
from realtime_operator.alignment import dtw,backtracking


def test_dtw():
    x=np.array([1,2,3,4,5,6,7,8,9,10,11,12],dtype=np.float64)
    y=np.array([1,2,3,4,5,6,7,8,9],dtype=np.float64)
    _arr=dtw(x,y)
    pytest.fail(f"Debug output: {_arr}")
    path=backtracking(_arr)
    pytest.fail(f"Debug output: {path}")
