"""
Generate different size spheres and plot the change in processing time and memory requirements.

----

Copyright 2020 - Cole Brauer, Dan Aukes
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import sphere

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    cpu = process.cpu_percent()
    mem = process.memory_info()[0] / float(2 ** 20)
    return [cpu, mem]

if __name__ == '__main__':
    process_times = []
    max_memory_values = []

    scales = [0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for scale in scales:
        mem_psutil = []
        time_process_started = time.time()

        radius = round(5 * scale)

        ditherResult = sphere(radius)
        mem_psutil.append(memory_usage_psutil())
        print('Model Created')

        # ditherMesh = Mesh.fromVoxelModel(ditherResult)
        # mem_psutil.append(memory_usage_psutil())

        time_process_finished = time.time()

        if scale > 0.5: # Skip first iteration to avoid "warm up" on first loop
            process_times.append(time_process_finished - time_process_started)
            max_memory_values.append(np.max(np.array(mem_psutil)[:, 1]))

    # Plot results ##########################
    plt.subplot(2, 1, 1)
    plt.subplots_adjust(left=0.12, right=0.95, hspace=0.6)
    plt.plot(scales[1:], max_memory_values)
    plt.title("Max Memory Usage")
    plt.xlabel("Scale")
    plt.ylabel("MB")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(scales[1:], process_times)
    plt.title("Total Process Time")
    plt.xlabel("Scale")
    plt.ylabel("Seconds")
    plt.grid()

    plt.show()