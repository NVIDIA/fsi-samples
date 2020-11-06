import numba
import cmath
import numpy as np
from numba import cuda
from .util import port_mask_nan

__all__ = ["fractional_diff", "get_weights_floored", "port_fractional_diff"]


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
    This function is to do convolution for one thread

    Arguments:
    ------
     shared: numba.cuda.DeviceNDArray
        3 chunks of data are stored in the shared memory
        the first [0, window_size) elements is the chunk of data that is
        necessary to compute the first convolution element.
        then [window_size, window_size + thread_tile * blockDim) elements
        are the inputs allocated for this block of threads
        the last [window_size + thread_tile,
        window_size + thread_tile + window_size) is to store the kernel values
     history_len: int
        total number of historical elements available for this chunk of data
     out_arr: numba.cuda.DeviceNDArray
        output gpu_array of size of `thread_tile`
     window_size: int
        the number of elements in the kernel
     arr_len: int
        the chunk array length, same as `thread_tile`
     offset: int
        indicate the starting index of the chunk array in the shared for
        this thread.
     offset: int
        indicate the starting position of the weights/kernel array
     min_size: int
         the minimum number of non-na elements
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
def kernel(in_arr, weight_arr, out_arr, window,
           arr_len, thread_tile, min_size):
    """
    This kernel is to do 1D convlution on `in_arr` array with `weight_arr`
    as kernel. The results is saved on `out_arr`.

    Arguments:
    ------
     in_arr: numba.cuda.DeviceNDArray
        input gpu array
     weight_arr: numba.cuda.DeviceNDArray
        convolution kernel gpu array
     out_arr: numba.cuda.DeviceNDArray
        output gpu_array
     window: int
        the number of elements in the weight_arr
     arr_len: int
        the input/output array length
     thread_tile: int
        each thread is responsible for `thread_tile` number of elements
     min_size: int
         the minimum number of non-na elements
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
            shared[offset + window - 1] = in_arr[
                starting_id + offset]
        cuda.syncthreads()

    # copy the window - 1 into the shared
    for j in range(0, window - 1, block_size):
        if (((tx + j) <
             window - 1) and (
                 starting_id - window + 1 + tx + j >= 0)):
            shared[tx + j] = \
                in_arr[starting_id - window + 1 + tx + j]
        cuda.syncthreads()
    # copy the weights into the shared
    for j in range(0, window, block_size):
        element_id = tx + j
        if (((tx + j) < window) and (element_id < window)):
            shared[thread_tile * block_size + window - 1 + tx +
                   j] = weight_arr[tx + j]
        cuda.syncthreads()
    # slice the shared memory for each threads
    start_shared = tx * thread_tile
    his_len = min(window - 1,
                  starting_id + tx * thread_tile)
    # slice the global memory for each threads
    start = starting_id + tx * thread_tile
    end = min(starting_id + (tx + 1) * thread_tile, arr_len)
    sub_outarr = out_arr[start:end]
    sub_len = end - start
    conv_window(shared, his_len, sub_outarr,
                window, sub_len,
                window - 1 + start_shared,
                thread_tile * block_size + window - 1,
                min_size)


def fractional_diff(input_arr, d=0.5, floor=1e-3, min_periods=None,
                    thread_tile=2, number_of_threads=512):
    """
    The fractional difference computation method.

    Arguments:
    -------
      input_arr: numba.cuda.DeviceNDArray or cudf.Series
        the input array to compute the fractional difference
      d: float
        the differencing value. range from 0 to 1
      floor: float
        minimum value for the weights for computational efficiency.
      min_periods: int
        default the lengths of the weights. Need at least min_periods of
        non-na elements to get fractional difference value
      thread_tile: int
        each thread will be responsible for `thread_tile` number of
        elements in window computation
      number_of_threads: int
        number of threads in a block for CUDA computation

    Returns
    -------
    (numba.cuda.DeviceNDArray, np.array)
        the computed fractional difference array and the weight array tuple

    """
    if isinstance(input_arr, numba.cuda.cudadrv.devicearray.DeviceNDArray):
        gpu_in = input_arr
    else:
        gpu_in = input_arr.to_gpu_array()

    # compute the weights for the fractional difference
    weights = get_weights_floored(d=d,
                                  num_k=len(input_arr),
                                  floor=floor)[::-1, 0]
    weights_out = np.ascontiguousarray(weights)
    weights = numba.cuda.to_device(weights_out)

    window = len(weights)

    if min_periods is None:
        min_periods = window
    else:
        min_periods = min_periods

    number_of_threads = number_of_threads
    array_len = len(gpu_in)

    # allocate the output array
    gpu_out = numba.cuda.device_array_like(gpu_in)

    number_of_blocks = \
        (array_len + (number_of_threads * thread_tile - 1)) // \
        (number_of_threads * thread_tile)

    shared_buffer_size = (number_of_threads * thread_tile +
                          window - 1 + window)

    # call the conv kernel
    kernel[(number_of_blocks,),
           (number_of_threads,),
           0,
           shared_buffer_size * 8](gpu_in,
                                   weights,
                                   gpu_out,
                                   window,
                                   array_len,
                                   thread_tile,
                                   min_periods)
    return gpu_out, weights_out


def port_fractional_diff(asset_indicator, input_arr, d=0.5, floor=1e-3,
                         min_periods=None, thread_tile=2,
                         number_of_threads=512):
    """
    Calculate the fractional differencing signal for all the financial
    assets indicated by asset_indicator.


    Arguments:
    -------
      asset_indicator: cudf.Series
        the integer indicator array to indicate the start of the different
        asset
      input_arr: numba.cuda.DeviceNDArray or cudf.Series
        the input array to compute the fractional difference
      d: float
        the differencing value. range from 0 to 1
      floor: float
        minimum value for the weights for computational efficiency.
      min_periods: int
        default the lengths of the weights. Need at least min_periods of
        non-na elements to get fractional difference value
      thread_tile: int
        each thread will be responsible for `thread_tile` number of
        elements in window computation
      number_of_threads: int
        number of threads in a block for CUDA computation

    Returns
    -------
    (numba.cuda.DeviceNDArray, np.array)
        the computed fractional difference array and the weight array tuple
    """
    out, weights = fractional_diff(input_arr, d=d, floor=floor,
                                   min_periods=min_periods,
                                   thread_tile=thread_tile,
                                   number_of_threads=number_of_threads)
    port_mask_nan(asset_indicator.to_gpu_array(), out, 0,
                  len(weights) - 1)
    return out, weights
