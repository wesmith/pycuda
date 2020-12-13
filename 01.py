# WS 12/12/20
# 01.py
# first pycuda program, from 'hands-on gpu programming with python and cuda book', just hand-copying
# the code at present

import pycuda.driver as drv 
drv.init()

#print('Number of detected CUDA devices: {}'.format(drv.Device.count()))

for i in range(drv.Device.count()):
    gpu_device = drv.Device(i)
    print('Device {}: {}'.format(i, gpu_device.name()))
    compute_capability = float('%d.%d' % gpu_device.compute_capability())
    print('\t Compute Capability: {}'.format(compute_capability))
    print('\t Total Memory: {} MB'.format(gpu_device.total_memory()//1024**2))
    # WS note: in python 2.7 (author's version) it is .iteritems(); in python 3 it is .items()
    device_attributes_tuples = gpu_device.get_attributes().items()
    device_attributes = {}
    for k, v in device_attributes_tuples:
        device_attributes[str(k)] = v
        #print('\t', k, v)
    num_mp = device_attributes['MULTIPROCESSOR_COUNT']
    cuda_cores_per_mp = {5.3 : 128}[compute_capability]  # just hardwired this: see enotes for more info, sources
    print('\t {} Multiprocessors x {} CUDA cores/MP = {} Total CUDA cores'. format(num_mp, 
        cuda_cores_per_mp, num_mp * cuda_cores_per_mp))

    device_attributes.pop('MULTIPROCESSOR_COUNT')

    for k in device_attributes.keys():
        print('\t {}: {}'.format(k, device_attributes[k]))
