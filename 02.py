# WS 12/13/20
# 02.py
# from 'hands-on gpu programming with python and cuda book'

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray

host_data     = np.array([1,2,3,4,5], dtype=np.float32)
device_data   = gpuarray.to_gpu(host_data)
device_datax2 = 2 * device_data
host_datax2   = device_datax2.get()

print(host_datax2)



