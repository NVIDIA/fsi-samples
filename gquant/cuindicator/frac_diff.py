import numba
import cmath
import numpy as np
from numba import cuda


def get_weights_floored(d, num_k, floor=1e-3):
    r"""Calculate weights ($w$) for each lag ($k$) through
    $w_k = -w_{k-1} \frac{d - k + 1}{k}$ provided weight above a minimum value
    (floor) for the weights to prevent computation of weights for the entire
    time series.

    Args:
        d (int): differencing value.
        num_k (int): number of lags (typically length of timeseries) to
        calculate w.
        floor (float): minimum value for the weights for computational
        efficiency.
    """
    w_k = np.array([1])
    k = 1

    while k < num_k:
        w_k_latest = -w_k[-1] * ((d - k + 1)) / k
        if abs(w_k_latest) <= floor:
            break

        w_k = np.append(w_k, w_k_latest)

        k += 1

    w_k = w_k.reshape(-1, 1)

    return w_k


@cuda.jit(device=True)
def conv_window(shared, history_len, out_arr, window_size,
                arr_len, offset, offset2, min_size):
    """
    This function is to compute the sum for the window
    See `window_kernel` for detailed arguments
    """
    for i in range(arr_len):
        if i + history_len < window_size-1:
            out_arr[i] = np.nan
        else:
            s = 0.0
            average_size = 0
            for j in range(0, window_size):
                if not (cmath.isnan(
                        shared[offset + i - j])):
                    s += (shared[offset + i - j] *
                          shared[offset2 + window_size - 1 - j])
                    average_size += 1
            if average_size >= min_size:
                out_arr[i] = s
            else:
                out_arr[i] = np.nan


@cuda.jit
def kernel(in_arr, weight_arr, out_arr, backward_length,
           arr_len, thread_tile, min_size):
    """
    This kernel is to copy input array elements into shared array.
    The total window size is backward_length + forward_length. To compute
    output element at i, it uses [i - backward_length - 1, i] elements in
    history, and [i + 1, i + forward_lengh] elements in the future.
    Arguments:
        in_arr: input gpu array
        out_arr: output gpu_array
        backward_length: the history elements in the windonw
        arr_len: the input/output array length
        thread_tile: each thread is responsible for `thread_tile` number
                     of elements
        min_size: the minimum number of non-na elements
    """
    shared = cuda.shared.array(shape=0,
                               dtype=numba.float64)
    block_size = cuda.blockDim.x  # total number of threads
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    bid = cuda.blockIdx.x
    starting_id = bid * block_size * thread_tile

    # copy the thread_tile * number_of_thread_per_block into the shared
    for j in range(thread_tile):
        offset = tx + j * block_size
        if (starting_id + offset) < arr_len:
            shared[offset + backward_length - 1] = in_arr[
                starting_id + offset]
        cuda.syncthreads()

    # copy the backward_length - 1 into the shared
    for j in range(0, backward_length - 1, block_size):
        if (((tx + j) <
             backward_length - 1) and (
                 starting_id - backward_length + 1 + tx + j >= 0)):
            shared[tx + j] = \
                in_arr[starting_id - backward_length + 1 + tx + j]
        cuda.syncthreads()
    # copy the weights into the shared
    for j in range(0, backward_length, block_size):
        element_id = tx + j
        if (((tx + j) < backward_length) and (element_id < backward_length)):
            shared[thread_tile * block_size + backward_length - 1 + tx +
                   j] = weight_arr[tx + j]
        cuda.syncthreads()
    # slice the shared memory for each threads
    start_shared = tx * thread_tile
    his_len = min(backward_length - 1,
                  starting_id + tx * thread_tile)
    # slice the global memory for each threads
    start = starting_id + tx * thread_tile
    end = min(starting_id + (tx + 1) * thread_tile, arr_len)
    sub_outarr = out_arr[start:end]
    sub_len = end - start
    conv_window(shared, his_len, sub_outarr,
                backward_length, sub_len,
                backward_length - 1 + start_shared,
                thread_tile * block_size + backward_length - 1,
                min_size)


class FracDiff(object):

    def __init__(self, input_arr, d=0.5, floor=1e-3, min_periods=None,
                 thread_tile=48, number_of_threads=64):
        """
        The Frac Diff class that is used to do compute the fractional
        difference

        Arguments:
            input_arr: the input GPU array or cudf.Series
            d: the fraction number
            floor: the cut off threashold to compute the fractional weights
            thread_tile: each thread will be responsible for `thread_tile`
                         number of elements in window computation
            number_of_threads: num. of threads in a block for CUDA computation
        """
        if isinstance(input_arr, numba.cuda.cudadrv.devicearray.DeviceNDArray):
            self.gpu_in = input_arr
        else:
            self.gpu_in = input_arr.data.to_gpu_array()

        self.weights = get_weights_floored(d=d,
                                           num_k=len(input_arr),
                                           floor=floor)[::-1, 0]
        self.weights = numba.cuda.to_device(np.ascontiguousarray(self.weights))
        self.window = len(self.weights)
        if min_periods is None:
            self.min_periods = self.window
        else:
            self.min_periods = min_periods
        self.number_of_threads = number_of_threads
        self.array_len = len(self.gpu_in)
        self.gpu_out = numba.cuda.device_array_like(self.gpu_in)
        self.thread_tile = thread_tile
        self.number_of_blocks = \
            (self.array_len + (number_of_threads * thread_tile - 1)) // \
            (number_of_threads * thread_tile)

        self.shared_buffer_size = (self.number_of_threads * self.thread_tile +
                                   self.window - 1 + self.window)

    def compute(self):
        gpu_out = numba.cuda.device_array_like(self.gpu_in)
        # gpu_out = cudf.Series(gpu_out)
        kernel[(self.number_of_blocks,),
               (self.number_of_threads,),
               0,
               self.shared_buffer_size * 8](self.gpu_in,
                                            self.weights,
                                            gpu_out,
                                            self.window,
                                            self.array_len,
                                            self.thread_tile,
                                            self.min_periods)
        # numba.cuda.synchronize()
        return gpu_out

np.random.seed(3)
num = np.random.rand(10000000)
arr = cuda.to_device(num)
start = time.time()
f = FracDiff(arr, d=0.3, floor=1e-4, thread_tile=2, number_of_threads=1024)
fnum = f.compute()
end = time.time()

print(f'compile Time {end-start} s')

start = time.time()
f = FracDiff(arr, d=0.5, floor=1e-4, thread_tile=2, number_of_threads=1024)
fnum = f.compute()
end = time.time()

print(f'no compile Time {end-start} s')


