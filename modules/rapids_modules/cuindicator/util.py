from .rolling import Rolling
from numba import cuda
import math
import numba


number_of_threads = 128


def diff(in_arr, n):
    if n < 0:
        return Rolling(1, in_arr, forward_window=-n).forward_diff()
    elif n > 0:
        return Rolling(n + 1, in_arr).backward_diff()
    else:
        return in_arr


def shift(in_arr, n):
    if n < 0:
        return Rolling(1, in_arr, forward_window=-n).forward_shift()
    elif n > 0:
        return Rolling(n + 1, in_arr).backward_shift()
    else:
        return in_arr


@cuda.jit
def ultimate_oscillator_kernel(high_arr, low_arr, close_arr, TR_arr, BP_arr,
                               arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if i == 0:
            TR_arr[i] = 0
            BP_arr[i] = 0
        else:
            TR = (max(high_arr[i],
                      close_arr[i - 1]) - min(low_arr[i], close_arr[i - 1]))
            TR_arr[i] = TR
            BP = close_arr[i] - min(low_arr[i], close_arr[i - 1])
            BP_arr[i] = BP


@cuda.jit
def port_ultimate_oscillator_kernel(asset_ind, high_arr, low_arr, close_arr,
                                    TR_arr, BP_arr,
                                    arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if asset_ind[i] == 1:
            TR_arr[i] = 0
            BP_arr[i] = 0
        else:
            TR = (max(high_arr[i],
                      close_arr[i - 1]) - min(low_arr[i], close_arr[i - 1]))
            TR_arr[i] = TR
            BP = close_arr[i] - min(low_arr[i], close_arr[i - 1])
            BP_arr[i] = BP


@cuda.jit
def moneyflow_kernel(pp_arr, volume_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if i == 0:
            out_arr[i] = 0
        else:
            if pp_arr[i] > pp_arr[i - 1]:
                out_arr[i] = pp_arr[i] * volume_arr[i]
            else:
                out_arr[i] = 0.0


@cuda.jit
def port_moneyflow_kernel(asset_ind, pp_arr, volume_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if asset_ind[i] == 1:
            out_arr[i] = 0
        else:
            if pp_arr[i] > pp_arr[i - 1]:
                out_arr[i] = pp_arr[i] * volume_arr[i]
            else:
                out_arr[i] = 0.0


@cuda.jit
def onbalance_kernel(close_arr, volume_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:

        if i == 0:
            out_arr[i] = 0
        else:
            if close_arr[i] - close_arr[i - 1] > 0:
                out_arr[i] = volume_arr[i]
            elif close_arr[i] - close_arr[i - 1] == 0:
                out_arr[i] = 0.0
            else:
                out_arr[i] = -volume_arr[i]


@cuda.jit
def port_onbalance_kernel(asset_ind, close_arr, volume_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if asset_ind[i] == 1:
            out_arr[i] = 0
        else:
            if close_arr[i] - close_arr[i - 1] > 0:
                out_arr[i] = volume_arr[i]
            elif close_arr[i] - close_arr[i - 1] == 0:
                out_arr[i] = 0.0
            else:
                out_arr[i] = -volume_arr[i]


@cuda.jit
def average_price_kernel(high_arr, low_arr, close_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        out_arr[i] = (high_arr[i] + low_arr[i] + close_arr[i]) / 3.0


@cuda.jit
def true_range_kernel(high_arr, low_arr, close_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if i == 0:
            out_arr[i] = 0
        else:
            out_arr[i] = max(high_arr[i],
                             close_arr[i - 1]) - min(low_arr[i],
                                                     close_arr[i - 1])


@cuda.jit
def port_true_range_kernel(asset_ind, high_arr, low_arr, close_arr, out_arr,
                           arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if asset_ind[i] == 1:
            out_arr[i] = 0
        else:
            out_arr[i] = max(high_arr[i],
                             close_arr[i - 1]) - min(low_arr[i],
                                                     close_arr[i - 1])


@cuda.jit
def port_mask_kernel(asset_ind, beg, end, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if asset_ind[i] == 1:
            if beg + i >= 0:
                for j in range(beg + i, min(end + i, arr_len)):
                    out_arr[j] = math.nan
            else:
                for j in range(beg + i + arr_len, min(end + i + arr_len,
                                                      arr_len)):
                    out_arr[j] = math.nan
                for j in range(0, min(end + i, arr_len)):
                    out_arr[j] = math.nan


@cuda.jit
def port_mask_zero_kernel(asset_ind, beg, end, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if asset_ind[i] == 1:
            if beg + i >= 0:
                for j in range(beg + i, min(end + i, arr_len)):
                    out_arr[j] = 0
            else:
                for j in range(beg + i + arr_len, min(end + i + arr_len,
                                                      arr_len)):
                    out_arr[j] = 0
                for j in range(0, min(end + i, arr_len)):
                    out_arr[j] = 0


@cuda.jit
def lowhigh_diff_kernel(high_arr, low_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if i == 0:
            out_arr[i] = 0
        else:
            out_arr[i] = abs(high_arr[i] - low_arr[i - 1]) - \
                         abs(low_arr[i] - high_arr[i - 1])


@cuda.jit
def port_lowhigh_diff_kernel(asset_ind, high_arr, low_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if asset_ind[i] == 1:
            out_arr[i] = 0
        else:
            out_arr[i] = abs(high_arr[i] - low_arr[i - 1]) - \
                         abs(low_arr[i] - high_arr[i - 1])


@cuda.jit
def up_down_kernel(high_arr, low_arr, upD_arr, doD_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len - 1:
        if (math.isnan(high_arr[i]) or math.isnan(high_arr[i + 1]) or
                math.isnan(low_arr[i]) or math.isnan(low_arr[i + 1])):
            upD_arr[i] = math.nan
            doD_arr[i] = math.nan
        else:
            upMove = high_arr[i + 1] - high_arr[i]
            doMove = low_arr[i] - low_arr[i + 1]
            if upMove > doMove and upMove > 0:
                upD_arr[i] = upMove
            else:
                upD_arr[i] = 0
            if doMove > upMove and doMove > 0:
                doD_arr[i] = doMove
            else:
                doD_arr[i] = 0
    elif i == arr_len - 1:
        upD_arr[i] = math.nan
        doD_arr[i] = math.nan


@cuda.jit
def abs_kernel(in_arr, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if math.isnan(in_arr[i]):
            out_arr[i] = math.nan
        else:
            out_arr[i] = abs(in_arr[i])


@cuda.jit
def binary_substract(in_arr1, in_arr2, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if (math.isnan(in_arr1[i]) or math.isnan(in_arr2[i])):
            out_arr[i] = math.nan
        else:
            out_arr[i] = in_arr1[i] - in_arr2[i]


@cuda.jit
def binary_sum(in_arr1, in_arr2, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if (math.isnan(in_arr1[i]) or math.isnan(in_arr2[i])):
            out_arr[i] = math.nan
        else:
            out_arr[i] = in_arr1[i] + in_arr2[i]


@cuda.jit
def binary_multiply(in_arr1, in_arr2, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if (math.isnan(in_arr1[i]) or math.isnan(in_arr2[i])):
            out_arr[i] = math.nan
        else:
            out_arr[i] = in_arr1[i] * in_arr2[i]


@cuda.jit
def binary_div(in_arr1, in_arr2, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if (math.isnan(in_arr1[i]) or math.isnan(in_arr2[i])):
            out_arr[i] = math.nan
        else:
            if in_arr2[i] == 0 and in_arr1[i] == 0:
                out_arr[i] = math.nan
            elif in_arr2[i] == 0 and in_arr1[i] > 0:
                out_arr[i] = math.inf
            elif in_arr2[i] == 0 and in_arr1[i] < 0:
                out_arr[i] = -math.inf
            else:
                out_arr[i] = in_arr1[i] / in_arr2[i]


@cuda.jit
def scale_kernel(in_arr, scaler, out_arr, arr_len):
    i = cuda.grid(1)
    if i < arr_len:
        if math.isnan(in_arr[i]):
            out_arr[i] = math.nan
        else:
            out_arr[i] = in_arr[i] * scaler


def upDownMove(high_arr, low_arr):
    upD_arr = cuda.device_array_like(high_arr)
    doD_arr = cuda.device_array_like(high_arr)
    array_len = len(high_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    up_down_kernel[(number_of_blocks,), (number_of_threads,)](high_arr,
                                                              low_arr,
                                                              upD_arr,
                                                              doD_arr,
                                                              array_len)
    return upD_arr, doD_arr


def ultimate_osc(high_arr, low_arr, close_arr):
    TR_arr = cuda.device_array_like(high_arr)
    BP_arr = cuda.device_array_like(high_arr)
    array_len = len(high_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    ultimate_oscillator_kernel[(number_of_blocks,),
                               (number_of_threads,)](high_arr,
                                                     low_arr,
                                                     close_arr,
                                                     TR_arr,
                                                     BP_arr,
                                                     array_len)
    return TR_arr, BP_arr


def port_ultimate_osc(asset_ind, high_arr, low_arr, close_arr):
    TR_arr = cuda.device_array_like(high_arr)
    BP_arr = cuda.device_array_like(high_arr)
    array_len = len(high_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    port_ultimate_oscillator_kernel[(number_of_blocks,),
                                    (number_of_threads,)](asset_ind,
                                                          high_arr,
                                                          low_arr,
                                                          close_arr,
                                                          TR_arr,
                                                          BP_arr,
                                                          array_len)
    return TR_arr, BP_arr


def abs_arr(in_arr):
    out_arr = cuda.device_array_like(in_arr)
    array_len = len(in_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    abs_kernel[(number_of_blocks,), (number_of_threads,)](in_arr,
                                                          out_arr,
                                                          array_len)
    return out_arr


def true_range(high_arr, low_arr, close_arr):
    out_arr = cuda.device_array_like(high_arr)
    array_len = len(high_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    true_range_kernel[(number_of_blocks,), (number_of_threads,)](high_arr,
                                                                 low_arr,
                                                                 close_arr,
                                                                 out_arr,
                                                                 array_len)
    return out_arr


def port_true_range(asset_indicator, high_arr, low_arr, close_arr):
    out_arr = cuda.device_array_like(high_arr)
    array_len = len(high_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    port_true_range_kernel[(number_of_blocks,),
                           (number_of_threads,)](asset_indicator,
                                                 high_arr,
                                                 low_arr,
                                                 close_arr,
                                                 out_arr,
                                                 array_len)
    return out_arr


def port_mask_nan(asset_indicator, input_arr, beg, end):
    array_len = len(input_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    port_mask_kernel[(number_of_blocks,),
                     (number_of_threads,)](asset_indicator,
                                           beg,
                                           end,
                                           input_arr,
                                           array_len)


def port_mask_zero(asset_indicator, input_arr, beg, end):
    array_len = len(input_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    port_mask_zero_kernel[(number_of_blocks,),
                          (number_of_threads,)](asset_indicator,
                                                beg,
                                                end,
                                                input_arr,
                                                array_len)


def average_price(high_arr, low_arr, close_arr):
    out_arr = cuda.device_array_like(high_arr)
    array_len = len(high_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    average_price_kernel[(number_of_blocks,), (number_of_threads,)](high_arr,
                                                                    low_arr,
                                                                    close_arr,
                                                                    out_arr,
                                                                    array_len)
    return out_arr


def money_flow(pp_arr, volume_arr):
    out_arr = cuda.device_array_like(pp_arr)
    array_len = len(pp_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    moneyflow_kernel[(number_of_blocks,), (number_of_threads,)](pp_arr,
                                                                volume_arr,
                                                                out_arr,
                                                                array_len)
    return out_arr


def port_money_flow(asset_ind, pp_arr, volume_arr):
    out_arr = cuda.device_array_like(pp_arr)
    array_len = len(pp_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    port_moneyflow_kernel[(number_of_blocks,),
                          (number_of_threads,)](asset_ind,
                                                pp_arr,
                                                volume_arr,
                                                out_arr,
                                                array_len)
    return out_arr


def onbalance_volume(close_arr, volume_arr):
    out_arr = cuda.device_array_like(close_arr)
    array_len = len(close_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    onbalance_kernel[(number_of_blocks,), (number_of_threads,)](close_arr,
                                                                volume_arr,
                                                                out_arr,
                                                                array_len)
    return out_arr


def port_onbalance_volume(asset_ind, close_arr, volume_arr):
    out_arr = cuda.device_array_like(close_arr)
    array_len = len(close_arr)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    port_onbalance_kernel[(number_of_blocks,),
                          (number_of_threads,)](asset_ind,
                                                close_arr,
                                                volume_arr,
                                                out_arr,
                                                array_len)
    return out_arr


def lowhigh_diff(high_arr, low_arr):
    out_arr = cuda.device_array_like(high_arr)
    array_len = len(high_arr)
    number_of_blocks = \
        (array_len + (number_of_threads - 1)) // number_of_threads
    lowhigh_diff_kernel[(number_of_blocks,), (number_of_threads,)](high_arr,
                                                                   low_arr,
                                                                   out_arr,
                                                                   array_len)
    return out_arr


def port_lowhigh_diff(asset_ind, high_arr, low_arr):
    out_arr = cuda.device_array_like(high_arr)
    array_len = len(high_arr)
    number_of_blocks = \
        (array_len + (number_of_threads - 1)) // number_of_threads
    port_lowhigh_diff_kernel[(number_of_blocks,),
                             (number_of_threads,)](asset_ind,
                                                   high_arr,
                                                   low_arr,
                                                   out_arr,
                                                   array_len)
    return out_arr


def substract(in_arr1, in_arr2):
    out_arr = cuda.device_array_like(in_arr1)
    array_len = len(in_arr1)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    binary_substract[(number_of_blocks,), (number_of_threads,)](in_arr1,
                                                                in_arr2,
                                                                out_arr,
                                                                array_len)
    return out_arr


def summation(in_arr1, in_arr2):
    out_arr = cuda.device_array_like(in_arr1)
    array_len = len(in_arr1)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    binary_sum[(number_of_blocks,), (number_of_threads,)](in_arr1,
                                                          in_arr2,
                                                          out_arr,
                                                          array_len)
    return out_arr


def multiply(in_arr1, in_arr2):
    out_arr = cuda.device_array_like(in_arr1)
    array_len = len(in_arr1)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    binary_multiply[(number_of_blocks,), (number_of_threads,)](in_arr1,
                                                               in_arr2,
                                                               out_arr,
                                                               array_len)
    return out_arr


def division(in_arr1, in_arr2):
    out_arr = cuda.device_array_like(in_arr1)
    array_len = len(in_arr1)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    binary_div[(number_of_blocks,), (number_of_threads,)](in_arr1,
                                                          in_arr2,
                                                          out_arr,
                                                          array_len)
    return out_arr


def scale(in_arr1, scaler):
    out_arr = cuda.device_array_like(in_arr1)
    array_len = len(in_arr1)
    number_of_blocks = (array_len + (
        number_of_threads - 1)) // number_of_threads
    scale_kernel[(number_of_blocks,), (number_of_threads,)](in_arr1,
                                                            scaler,
                                                            out_arr,
                                                            array_len)
    return out_arr


@cuda.jit
def cumsum_kernel(in_arr, out_arr, block_arr, arr_len):
    shared = cuda.shared.array(shape=0, dtype=numba.float64)
    num_threads = cuda.blockDim.x
    tx = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    partial_sum_offset = num_threads * 2
    starting_id = bid * partial_sum_offset

    # load the in_arr to shared
    for j in range(2):
        offset = tx + j * num_threads
        if (offset + starting_id) < arr_len:
            shared[offset] = in_arr[offset + starting_id]
        else:
            shared[offset] = 0.0
        cuda.syncthreads()

    offset = 1

    d = num_threads

    while d > 0:
        cuda.syncthreads()
        if (tx < d):
            ai = offset*(2*tx+1)-1
            bi = offset*(2*tx+2)-1
            shared[bi] += shared[ai]
        offset *= 2
        d = d // 2

    if (tx == 0):
        block_arr[bid] = shared[2 * num_threads - 1]
        shared[2 * num_threads - 1] = 0.0

    d = 1
    while d < 2 * num_threads:
        offset = offset // 2
        cuda.syncthreads()
        if tx < d:
            ai = offset*(2*tx+1)-1
            bi = offset*(2*tx+2)-1
            t = shared[ai]
            shared[ai] = shared[bi]
            shared[bi] += t
        d *= 2
    cuda.syncthreads()

    # load back to the output
    for j in range(2):
        offset = tx + j * num_threads
        if (offset + starting_id) < arr_len and offset + 1 < 2 * num_threads:
            out_arr[offset + starting_id] = shared[offset + 1]

    if tx == 0:
        arr_id = min(arr_len - 1, starting_id + 2 * num_threads - 1)
        out_arr[arr_id] = block_arr[bid]


@cuda.jit
def correct_kernel(in_arr, block_arr, arr_len):
    num_threads = cuda.blockDim.x
    tx = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    partial_sum_offset = num_threads * 2
    starting_id = bid * partial_sum_offset

    for j in range(2):
        offset = tx + j * num_threads
        lookup = bid - 1

        if lookup >= 0 and (offset + starting_id) < arr_len:
            in_arr[offset + starting_id] += block_arr[lookup]


def cumsum(g_input, number_of_threads=1024):
    array_len = len(g_input)
    number_of_blocks = (array_len + (
        number_of_threads * 2 - 1)) // (number_of_threads * 2)

    shared_buffer_size = (number_of_threads * 2)

    block_summary = numba.cuda.device_array(number_of_blocks)
    gpu_out = numba.cuda.device_array_like(g_input)
    cumsum_kernel[(number_of_blocks,),
                  (number_of_threads,),
                  0,
                  shared_buffer_size * 8](g_input,
                                          gpu_out,
                                          block_summary,
                                          array_len)
    if (number_of_blocks == 1):
        return gpu_out
    else:
        block_sum = cumsum(block_summary)
        correct_kernel[(number_of_blocks,),
                       (number_of_threads,)](gpu_out,
                                             block_sum,
                                             array_len)
        return gpu_out
