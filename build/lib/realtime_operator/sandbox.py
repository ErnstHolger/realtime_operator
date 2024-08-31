import numpy as np
import time
from single_operator import get_median

state_SIZE=50
n=100_000
t = np.arange(1, n, 1).astype(float)
z = np.arange(1, n, 1).astype(float)
state = np.zeros(state_SIZE, dtype=float)


start=time.time()
for i in range(n-1):
    _t,_z=get_median(state,t[i],z[i],5,0,0)

print(time.time()-start)


    


   