# WS 12/13/20
# time_calc0.py
# from 'hands-on gpu programming with python and cuda book'

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time

host_data = np.float32(np.random.random(50000000))

t1 = time()
host_data_x2 = host_data * np.float32(2)
t2 = time()

print('total CPU time: {} sec'.format(t2 - t1))

device_data = gpuarray.to_gpu(host_data)

t1 = time()
device_data_x2 = device_data * np.float32(2)
t2 = time()

from_device = device_data_x2.get()

print('total GPU time: {} sec'.format(t2 - t1))

#print('host computation = GPU computation? {}'.format(np.allclose(from_device, host_data_x2)))


