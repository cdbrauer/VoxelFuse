import numpy as np
import pyopencl as cl
from numba import njit, prange

@njit
def flatten(a):
    x_len = len(a[:, 0, 0])
    y_len = len(a[0, :, 0])
    z_len = len(a[0, 0, :])

    return a.reshape((x_len * y_len, z_len))

@njit(parallel=True)
def unflatten(a, b):
    x_len = len(b[:, 0, 0])
    y_len = len(b[0, :, 0])
    z_len = len(b[0, 0, :])

    for z in prange(z_len):
        for x in prange(x_len):
            for y in prange(y_len):
                i = z*x_len + x*y_len + y
                b[x, y, z] = a[i]

@njit(parallel=True)
def unflatten_and_append(a, b, start):
    x_len = len(a[:, 0, 0])
    y_len = len(a[0, :, 0])
    z_len = len(a[0, 0, :])

    for i in prange(len(b)):
        index = i + start
        z = index // (x_len * y_len)
        l = index % (x_len * y_len)
        y = l // x_len
        x = l % x_len

        a[x, y, z] = b[i]

def opencl_dot3d(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    x_len = np.int32(len(a[:, 0, 0]))
    y_len = np.int32(len(a[0, :, 0]))
    z_len = np.int32(len(b[:, 0]))

    a_flat = flatten(a)

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
        __kernel void matrix_dot_vector(__global const float4 *a, __global const float4 *b, const unsigned int x_len, const int y_len, const int start, __global float *result) {
            int gid = get_global_id(0);
            int i = gid + start;
            
            int z = i / (x_len * y_len);
            int l = i % (x_len * y_len);
            int y = l / x_len;
            int x = l % x_len;
            
            result[gid] = dot(a[y*x_len + x], b[z]);
        }
    """).build()

    queue = cl.CommandQueue(context)

    mem_flags = cl.mem_flags
    a_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a_flat)
    b_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=b)

    result_len = x_len * y_len * z_len

    # TODO: Calculate how many parts we need to split the result into in order to not run out of memory
    n_parts = 2

    # result = np.array([], np.float32)
    result = np.zeros((x_len, y_len, z_len), np.float32)

    for i in range(n_parts):
        start = np.int32(result_len - (result_len // (i+1)))
        result_part = np.zeros(result_len // n_parts, np.float32)

        result_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, result_part.nbytes)
        program.matrix_dot_vector(queue, result_part.shape, None, a_buf, b_buf, x_len, y_len, start, result_buf)
        cl.enqueue_copy(queue, result_part, result_buf)

        # result.append(result_part)
        print("Adding")
        # result = np.append(result, result_part, axis=0)
        unflatten_and_append(result, result_part, start)
        print("Done adding")

    #charles' crappy attempt at a deflattener
    # print("Unflattening")
    # unflattened = np.zeros((x_len, y_len, z_len), dtype=np.float32)
    # unflatten(result, unflattened)
    # print("Done unflattening")

    # index_result = 0
    # for z in range(z_len):
    #     for x in range(x_len):
    #         for y in range(y_len):
    #             unflattened[x, y, z] = result[index_result]
    #             index_result += 1

    return result
    # return unflattened


def opencl_dot2d(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    y_len = np.int32(len(a[:, 0]))
    z_len = np.int32(len(b[:, 0]))

    print(y_len)
    print(z_len)

    platform = cl.get_platforms()[0]
    print(platform.name)

    device = platform.get_devices()[0]
    print(device.name)

    context = cl.Context([device])

    program = cl.Program(context, """
        __kernel void matrix_dot_vector(__global const float4 *a, __global const float4 *b, const unsigned int x_len, const int y_len, const int start, __global float *result) {
            int gid = get_global_id(0);
            int i = gid + start;

            int z = i / (x_len * y_len);
            int l = i % (x_len * y_len);
            int y = l / x_len;
            int x = l % x_len;

            result[gid] = dot(a[y*x_len + x], b[z]);
        }
    """).build()

    queue = cl.CommandQueue(context)

    mem_flags = cl.mem_flags
    a_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=b)

    result_len = x_len * y_len * z_len

    # TODO: Calculate how many parts we need to split the result into in order to not run out of memory
    n_parts = 1

    # result = np.array([], np.float32)
    result = np.zeros((x_len, y_len, z_len), np.float32)

    for i in range(n_parts):
        start = np.int32(result_len - (result_len // (i + 1)))
        result_part = np.zeros(result_len // n_parts, np.float32)

        result_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, result_part.nbytes)
        program.matrix_dot_vector(queue, result_part.shape, None, a_buf, b_buf, x_len, y_len, start, result_buf)
        cl.enqueue_copy(queue, result_part, result_buf)

        # result.append(result_part)
        print("Adding")
        # result = np.append(result, result_part, axis=0)
        unflatten_and_append(result, result_part, start)
        print("Done adding")


    return result

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
