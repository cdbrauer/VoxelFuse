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
        __kernel void matrix_dot_vector(__global const float4 *a, __global const float4 *b, const unsigned int x_len, const int y_len, const int z_len, __global float *result) {
            int gid = get_global_id(0);
            int i = gid % (z_len*4);
            int j = gid / (x_len * y_len);
            result[gid] = dot(a[i], b[j]);
            //result[gid] = b[z].s0;
            //result[gid] = a[y].s0;
        }
    """).build()

    queue = cl.CommandQueue(context)

    mem_flags = cl.mem_flags
    a_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a_flat)
    b_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=b)
    # x_len_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=np.int32(x_len))
    # y_len_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=np.int32(y_len))
    # z_len_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=np.int32(z_len))

    result = np.zeros((x_len * y_len * z_len), np.float32)
    # result = np.zeros(4, np.float32)
    result_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, result.nbytes)

    program.matrix_dot_vector(queue, result.shape, None, a_buf, b_buf, x_len, y_len, z_len, result_buf)

    cl.enqueue_copy(queue, result, result_buf)

    print(len(result))

    #charles' crappy attempt at a deflattener
    unflattened = np.zeros((x_len, y_len, z_len), dtype=np.float32)
    index_result = 0
    for z in prange(z_len):
        for x in prange(x_len):
            for y in prange(y_len):
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
    x = np.array([[
        [1, 2, 4, 8],
        [16, 32, 64, 128],
        [3, 6, 9, 12],
        [-0.6820, 0.0755, -0.3125, -21.8564],
        [3, 6, 9, 12],
        [-0.6820, 0.0755, -0.3125, -21.8564],
        [3, 6, 9, 12],
        [-0.6820, 0.0755, -0.3125, -21.8564],
        [16, 32, 64, 128]
    ], [
        [16, 32, 64, 128],
        [1, 2, 4, 8],
        [3, 6, 9, 12],
        [-0.6820, 0.0755, -0.3125, -21.8564],
        [3, 6, 9, 12],
        [-0.6820, 0.0755, -0.3125, -21.8564],
        [3, 6, 9, 12],
        [-0.6820, 0.0755, -0.3125, -21.8564],
        [16, 32, 64, 128]
    ]], dtype=np.float32)

    y = np.array([
        [1, 2, 4, 8],
        [16, 32, 64, 128],
        [3, 6, 9, 12],
        [-81.5, -9.5, 0.5, 1.0]
    ], dtype=np.float32)

    # x_len = 3
    # y_len = 3
    # z_len = 3
    # for i in range(27):
    #     z = i // (x_len * y_len)
    #     l = i % (z*(x_len * y_len))
    #     y = l // x_len
    #     x = l % x_len
    #     print(x)
    #     print(y)
    #     print(z)

    print("OpenCL:")
    print(opencl_dot3d(x, y))
    print("Numba:")
    print(numba_dot3d(x, y))
