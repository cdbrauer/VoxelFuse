import numpy as np
import pyopencl as cl
from numba import njit, prange

@njit
def flatten_3d_to_2d(a):
    x_len = len(a[:, 0, 0])
    y_len = len(a[0, :, 0])
    z_len = len(a[0, 0, :])

    return a.reshape((x_len * y_len, z_len))

def opencl_dot3d(a, b):
    x_len = np.int32(len(a[:, 0, 0]))
    y_len = np.int32(len(a[0, :, 0]))
    z_len = np.int32(len(b[:, 0]))

    a_flat = flatten_3d_to_2d(a)

    print(x_len)
    print(y_len)
    print(z_len)

    platform = cl.get_platforms()[0]
    print(platform.name)

    device = platform.get_devices()[0]
    print(device.name)

    context = cl.Context([device])
    # context = cl.create_some_context()

    program = cl.Program(context, """
        __kernel void matrix_dot_vector(__global const double4 *a, __global const double4 *b, const unsigned int x_len, const int y_len, const int z_len, __global double *result) {
            int gid = get_global_id(0);
            
            int z = gid / (x_len * y_len);
            int l = gid % (x_len * y_len);
            int y = l / x_len;
            int x = l % x_len;
            
            result[gid] = dot(a[y*x_len + x], b[z]);
        }
    """).build()

    queue = cl.CommandQueue(context)

    mem_flags = cl.mem_flags
    a_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a_flat)
    b_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=b)

    result = np.zeros((x_len * y_len * z_len), np.float64)
    result_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, result.nbytes)

    program.matrix_dot_vector(queue, result.shape, None, a_buf, b_buf, x_len, y_len, z_len, result_buf)

    cl.enqueue_copy(queue, result, result_buf)

    print(len(result))

    #charles' crappy attempt at a deflattener
    unflattened = np.zeros((x_len, y_len, z_len), dtype=np.float32)
    index_result = 0
    for z in range(z_len):
        for x in range(x_len):
            for y in range(y_len):
                unflattened[x, y, z] = result[index_result]
                index_result += 1

    #return result
    return unflattened

@njit(parallel=True)
def numba_dot3d(a, b):
    x_len = len(a[:, 0, 0])
    y_len = len(a[0, :, 0])
    z_len = len(b[:, 0])

    result = np.zeros((x_len, y_len, z_len), dtype=np.float32)

    for x in prange(x_len):
        for y in prange(y_len):
            for z in prange(z_len):
                result[x, y, z] = a[x, y, :].dot(b[z, :])

    return result

if __name__ == "__main__":
    x = np.random.rand(133, 4, 4).astype(np.float64)
    y = np.random.rand(3960, 4).astype(np.float64)

    opencl_res = opencl_dot3d(x, y)
    numba_res = numba_dot3d(x, y)

    print("OpenCL:")
    print(opencl_res)
    print("Numba:")
    print(numba_res)

    print(np.allclose(opencl_res, numba_res))
