# WS 12/13/20
# simple_element_kernel_example0.py
# from 'hands-on gpu programming with python and cuda book'

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from time import time

host_data = np.float32(np.random.random(50000000))

gpu_2x_ker = ElementwiseKernel("float *in, float *out", "out[i] = 2*in[i]", "gpu_2x_ker")

def speedcomparison():

    t1 = time()
    host_data_x2 = host_data * np.float32(2)
    t2 = time()

    print('total CPU time: {} sec'.format(t2 - t1))

    device_data = gpuarray.to_gpu(host_data)

    device_data_2x = gpuarray.empty_like(device_data)

    t1 = time()
    gpu_2x_ker(device_data, device_data_2x)
    t2 = time()

    from_device = device_data_2x.get()

    print('total GPU time: {} sec'.format(t2 - t1))

    #print('host computation = GPU computation? {}'.format(np.allclose(from_device, host_data_x2)))

if __name__ == '__main__':
    speedcomparison()


