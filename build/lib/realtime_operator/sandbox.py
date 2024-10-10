import numpy as np
import time
import matplotlib.pyplot as plt
from single_operator import median
from compression import any_compression,interpolate_fast,segment_resample
state_SIZE=50
n=100_000
t = np.arange(1, n, 1).astype(float)
z = np.arange(1, n, 1).astype(float)
state = np.zeros(state_SIZE, dtype=float)

start=5*np.ceil(t[1] / 5).astype(int)

ti,zi, _=segment_resample(t, z, 0.2,0.8,5000)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, z, label='Original Data', marker='o')
plt.plot(ti, zi, label='Interpolated Data', marker='^')
plt.xlabel('t')
plt.ylabel('z / zi')
plt.title('Original and Interpolated Data')
plt.legend()
plt.grid(True)
plt.show()

start=time.time()


print(time.time()-start)


    


   