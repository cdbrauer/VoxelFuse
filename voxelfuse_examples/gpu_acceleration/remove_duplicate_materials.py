"""
Demonstrate how to enable CUDA and the effect it has on the removeDuplicateMaterials function.

----

Copyright 2021 - Cole Brauer, Dan Aukes
"""
import time
from voxelfuse.voxel_model import VoxelModel, GpuSettings
from voxelfuse.primitives import *

if __name__=='__main__':
    gpu = GpuSettings()

    model1 = cube(100, (0, 0, 0), 1)
    model1 = model1 | cube(100, (1500, 0, 0), 1)
    model1 = model1 | cube(100, (3000, 0, 0), 1)
    model1 = model1 | cube(100, (4500, 0, 0), 2)
    model1 = model1 | cube(100, (6000, 0, 0), 2)
    model1 = model1 | cube(100, (7500, 0, 0), 2)

    print('Model size: ' + str(model1.voxels.shape))
    print('Default CUDA settings: ' + str(gpu.CUDA_enable) + ', ' + str(gpu.CUDA_device))

    gpu.setCUDA(False)
    gpu.applySettings()
    start = time.time()
    cuda0_result = model1.removeDuplicateMaterials()
    end = time.time()
    processingTime = (end - start)
    print('CUDA off: ' + str(processingTime) + ' sec')

    gpu.setCUDA(True, 0)
    gpu.applySettings()
    start = time.time()
    cuda1_result = model1.removeDuplicateMaterials()
    end = time.time()
    processingTime = (end - start)
    print('CUDA on, GPU 0: ' + str(processingTime) + ' sec')