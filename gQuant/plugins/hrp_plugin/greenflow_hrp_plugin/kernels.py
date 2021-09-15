"""
 ////////////////////////////////////////////////////////////////////////////
 //
 // Copyright (C) NVIDIA Corporation.  All rights reserved.
 //
 // NVIDIA Sample Code
 //
 // Please refer to the NVIDIA end user license agreement (EULA) associated
 // with this source code for terms and conditions that govern your use of
 // this software. Any use, reproduction, disclosure, or distribution of
 // this software and related documentation outside the terms of the EULA
 // is strictly prohibited.
 //
 ////////////////////////////////////////////////////////////////////////////
"""

from numba import cuda
import numba
import cupy
import math

MAX_ASSETS = 32
MAX_YEARS = 24
PARENT_MAX_ASSETS = 2 * MAX_ASSETS - 1
SUM_LEN = 256 * 32
# MAX_SHARE = 256 * MAX_YEARS * 4


@cuda.jit
def boot_strap(result, ref, block_size, num_positions, positions):
    sample, assets, length = result.shape
    i = cuda.threadIdx.x
    sample_id = cuda.blockIdx.x // num_positions
    position_id = cuda.blockIdx.x % num_positions
    sample_at = positions[cuda.blockIdx.x]
    for k in range(i, block_size*assets, cuda.blockDim.x):
        asset_id = k // block_size
        loc = k % block_size
        if (position_id * block_size + loc + 1 < length):
            result[sample_id, asset_id, position_id * block_size +
                   loc + 1] = ref[asset_id,  sample_at + loc]


@cuda.jit(device=True)
def gpu_sum(array):
    i = cuda.threadIdx.x
    total_len = SUM_LEN
    length = total_len
    while length > 0:
        length = length // 2
        for k in range(i, length, cuda.blockDim.x):
            if k+length < total_len:
                array[k] += array[k + length]
        cuda.syncthreads()


@cuda.jit
def compute_cov(means, cov, distance, returns, months_starts, num_months,
                assets, time_len, window):
    """
    means of size [sample, months, assets]
    num_months should be 60 - 12, as the windows size is one year 12 months
    """
    shared = cuda.shared.array(shape=0, dtype=numba.float32)
    shared_buffer_size = shared.size
    i = cuda.threadIdx.x
    sample_id = cuda.blockIdx.x // num_months
    step_id = cuda.blockIdx.x % num_months
    start_id = months_starts[step_id]
    end_id = months_starts[
        step_id +
        window] if step_id + window < months_starts.size else time_len
    for a in range(assets):
        # copy asset return to shared
        for k in range(i, shared_buffer_size, cuda.blockDim.x):
            shared[k] = 0
        cuda.syncthreads()
        for k in range(i + start_id, end_id, cuda.blockDim.x):
            shared[k - start_id] = returns[sample_id, a, k]
        cuda.syncthreads()
        gpu_sum(shared)
        if i == 0:
            means[sample_id, step_id, a] = shared[0] / (end_id - start_id)
        cuda.syncthreads()
    for a in range(assets):
        for b in range(a, assets):
            # copy asset return to shared
            for k in range(i, shared_buffer_size, cuda.blockDim.x):
                shared[k] = 0
            cuda.syncthreads()
            mean_a = means[sample_id, step_id, a]
            mean_b = means[sample_id, step_id, b]
            for k in range(i + start_id, end_id, cuda.blockDim.x):
                shared[k - start_id] = (returns[sample_id, a, k] - mean_a) * (
                    returns[sample_id, b, k] - mean_b)
            cuda.syncthreads()
            gpu_sum(shared)
            if i == 0:
                cov[sample_id, step_id, a, b] = shared[0] / (end_id - start_id)
                cov[sample_id, step_id, b, a] = shared[0] / (end_id - start_id)
            cuda.syncthreads()
    # compute distance
    for k in range(i, assets*assets, cuda.blockDim.x):
        a = k // assets
        b = k % assets
        if b > a:
            var_a = cov[sample_id, step_id, a, a]
            var_b = cov[sample_id, step_id, b, b]
            cov_ab = cov[sample_id, step_id, a, b]
            dis_ab = math.sqrt((1.0 - cov_ab / math.sqrt(var_a * var_b)) / 2.0)
            offset = (2 * assets - 1 - a) * a // 2 + (b - a - 1)
            shared[offset] = dis_ab
            # distance[sample_id, step_id, offset] = dis_ab
    cuda.syncthreads()

    # compute distance of the distance
    for k in range(i, assets*assets, cuda.blockDim.x):
        a = k // assets
        b = k % assets
        if b > a:
            summ = 0.0
            for col_id in range(assets):
                if col_id > a:
                    offset_a = (2 * assets - 1 - a) * a // 2 + (col_id - a - 1)
                    val_a = shared[offset_a]
                elif col_id < a:
                    offset_a = (2 * assets - 1 - col_id) * col_id // 2 + (
                        a - col_id - 1)
                    val_a = shared[offset_a]
                else:
                    val_a = 0.0
                if col_id > b:
                    offset_b = (2 * assets - 1 - b) * b // 2 + (col_id - b - 1)
                    val_b = shared[offset_b]
                elif col_id < b:
                    offset_b = (2 * assets - 1 - col_id) * col_id // 2 + (
                        b - col_id - 1)
                    val_b = shared[offset_b]
                else:
                    val_b = 0.0
                summ += (val_a - val_b) * (val_a - val_b)
            offset = (2 * assets - 1 - a) * a // 2 + (b - a - 1)
            distance[sample_id, step_id, offset] = math.sqrt(summ)


@cuda.jit
def leverage_for_target_vol(leverage, returns, months_starts, num_months,
                            window, long_window,
                            short_window, target_vol):
    """
    each block calculate for one rebalancing month,
    leverage of shape [sample, months]
    returns of shape [sample, time_len]
    num_months should be 60 - 12, as the windows size is one year 12 months
    """
    # shared = cuda.shared.array(MAX_SHARE, dtype=numba.float64)
    shared = cuda.shared.array(shape=0, dtype=numba.float32)
    total_samples, time_len = returns.shape
    # means = cuda.shared.array(1, dtype=numba.float64)
    # sds = cuda.shared.array(2, dtype=numba.float64)
    # means = shared[-1:]
    # sds = shared[-3:-1]

    annual_const = math.sqrt(252.)
    shared_buffer_size = shared.size
    i = cuda.threadIdx.x
    sample_id = cuda.blockIdx.x // num_months
    step_id = cuda.blockIdx.x % num_months
    start_id = months_starts[step_id]
    end_id = months_starts[
        step_id +
        window] if step_id + window < months_starts.size else time_len

    # calculate the means for the long window
    start_id = end_id - long_window
    # copy asset return to shared
    for k in range(i, shared_buffer_size, cuda.blockDim.x):
        shared[k] = 0
    cuda.syncthreads()
    for k in range(i + start_id, end_id, cuda.blockDim.x):
        shared[k - start_id] = returns[sample_id, k]
    cuda.syncthreads()

    gpu_sum(shared)
    cuda.syncthreads()
    means = shared[0] / (end_id - start_id)

    # calculate the std for the long window
    # copy asset return to shared
    for k in range(i, shared_buffer_size, cuda.blockDim.x):
        shared[k] = 0
    cuda.syncthreads()
    for k in range(i + start_id, end_id, cuda.blockDim.x):
        shared[k - start_id] = (returns[sample_id, k] -
                                means) * (returns[sample_id, k] - means)

    cuda.syncthreads()
    gpu_sum(shared)

    sd_long = math.sqrt(shared[0] / (end_id - start_id))

    # calculate the means for the short window
    start_id = end_id - short_window
    # copy asset return to shared
    for k in range(i, shared_buffer_size, cuda.blockDim.x):
        shared[k] = 0
    cuda.syncthreads()
    for k in range(i + start_id, end_id, cuda.blockDim.x):
        shared[k - start_id] = returns[sample_id, k]
    cuda.syncthreads()
    gpu_sum(shared)

    cuda.syncthreads()
    means = shared[0] / (end_id - start_id)
    cuda.syncthreads()

    # calculate the std for the short window
    for k in range(i, shared_buffer_size, cuda.blockDim.x):
        shared[k] = 0
    cuda.syncthreads()
    for k in range(i + start_id, end_id, cuda.blockDim.x):
        shared[k - start_id] = (returns[sample_id, k] - means) * (
            returns[sample_id, k] - means)
    cuda.syncthreads()
    gpu_sum(shared)

    sd_short = math.sqrt(shared[0] / (end_id - start_id))
    if i == 0:
        lev = target_vol / (max(sd_short, sd_long)*annual_const)
        leverage[sample_id, step_id] = lev


@cuda.jit(device=True)
def find(x, parent):
    p = x

    while parent[x] != x:
        x = parent[x]

    while parent[p] != x:
        p, parent[p] = parent[p], x
    return x


@cuda.jit(device=True)
def label(Z, n, parent):
    """Correctly label clusters in unsorted dendrogram."""
    next_label = n
    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = find(x, parent), find(y, parent)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        parent[x_root] = next_label
        parent[y_root] = next_label
        next_label += 1


@cuda.jit(device=True)
def mergeSort(a, L, R):
    current_size = 1
    # Outer loop for traversing Each
    # sub array of current_size
    while current_size < len(a):
        left = 0
        # Inner loop for merge call
        # in a sub array
        # Each complete Iteration sorts
        # the iterating sub array
        while left < len(a)-1:
            # mid index = left index of
            # sub array + current sub
            # array size - 1
            mid = min((left + current_size - 1), (len(a)-1))
            # (False result,True result)
            # [Condition] Can use current_size
            # if 2 * current_size < len(a)-1
            # else len(a)-1
            if 2 * current_size + left - 1 > len(a)-1:
                right = len(a) - 1
            else:
                right = 2 * current_size + left - 1
            # Merge call for each sub array
            merge(a, left, mid, right, L, R)
            left = left + current_size*2
        # Increasing sub array size by
        # multiple of 2
        current_size = 2 * current_size


@cuda.jit(device=True)
def merge(a, ll, m, r, L, R):
    n1 = m - ll + 1
    n2 = r - m
    L[:, :] = 0
    R[:, :] = 0
    for i in range(0, n1):
        L[i, 0] = a[ll + i, 0]
        L[i, 1] = a[ll + i, 1]
        L[i, 2] = a[ll + i, 2]
    for i in range(0, n2):
        R[i, 0] = a[m + i + 1, 0]
        R[i, 1] = a[m + i + 1, 1]
        R[i, 2] = a[m + i + 1, 2]

    i, j, k = 0, 0, ll
    while i < n1 and j < n2:
        if L[i, 2] > R[j, 2]:
            a[k, 0] = R[j, 0]
            a[k, 1] = R[j, 1]
            a[k, 2] = R[j, 2]
            j += 1
        else:
            a[k, 0] = L[i, 0]
            a[k, 1] = L[i, 1]
            a[k, 2] = L[i, 2]
            i += 1
        k += 1

    while i < n1:
        a[k, 0] = L[i, 0]
        a[k, 1] = L[i, 1]
        a[k, 2] = L[i, 2]
        i += 1
        k += 1

    while j < n2:
        a[k, 0] = R[j, 0]
        a[k, 1] = R[j, 1]
        a[k, 2] = R[j, 2]
        j += 1
        k += 1


@cuda.jit(device=True)
def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - (i * (i + 1) // 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) // 2) + (i - j - 1)


@cuda.jit(device=True)
def my_seriation(Z, N, stack, result):
    """Returns the order implied by a hierarchical tree (dendrogram).

       :param Z: A hierarchical tree (dendrogram).
       :param N: The number of points given to the clustering process.
       :param cur_index: The position in the tree for the recursive traversal.

       :return: The order implied by the hierarchical tree Z.
    """
    o_point = -1
    stack_point = 0
    stack[0] = N + N - 2

    while stack_point >= 0:
        v = stack[stack_point]
        stack_point -= 1
        left = int(Z[v - N, 0])
        right = int(Z[v - N, 1])

        if right >= N:
            stack_point += 1
            stack[stack_point] = right

        if left >= N:
            stack_point += 1
            stack[stack_point] = left

        if left < N:
            o_point += 1
            result[o_point] = left

        if right < N:
            o_point += 1
            result[o_point] = right
    return result


@cuda.jit
def single_linkage(output, orders, dists, num_months, n):
    """
    dists is shape [sample, months, distance]
    output is of shape [sample, months, n-1, 3]
    """
    large = 1e200
    merged = cuda.shared.array(MAX_ASSETS, dtype=numba.int64)
    merged[:] = 0
    D = cuda.shared.array(MAX_ASSETS, dtype=numba.float64)
    D[:] = large
    L = cuda.shared.array(shape=(MAX_ASSETS, 3), dtype=numba.float64)
    R = cuda.shared.array(shape=(MAX_ASSETS, 3), dtype=numba.float64)
    parent = cuda.shared.array(PARENT_MAX_ASSETS, dtype=numba.int64)
    for k in range(PARENT_MAX_ASSETS):
        parent[k] = k
    stack = cuda.shared.array(MAX_ASSETS, dtype=numba.int64)
    sample_id = cuda.blockIdx.x // num_months
    step_id = cuda.blockIdx.x % num_months
    x = 0
    for k in range(n - 1):
        current_min = large
        merged[x] = 1
        for i in range(n):
            if merged[i] == 1:
                continue
            dis_id = int(condensed_index(n, x, i))

            dist = dists[sample_id, step_id, dis_id]
            # print(k, i, dis_id, dist, D[i])
            if D[i] > dist:
                D[i] = dist

            if D[i] < current_min:
                y = i
                current_min = D[i]

        output[sample_id, step_id, k, 0] = x
        output[sample_id, step_id, k, 1] = y
        output[sample_id, step_id, k, 2] = current_min
        x = y
    # # Sort Z by cluster distances.
    mergeSort(output[sample_id, step_id], L, R)
    # # Find correct cluster labels and compute cluster sizes inplace.
    label(output[sample_id, step_id], n, parent)
    my_seriation(output[sample_id, step_id], n,
                 stack, orders[sample_id, step_id])


@cuda.jit
def HRP_weights(weights, covariances, res_order, N, num_months):
    """
    covariances, [samples, number, N, N]
    res_order, [sample, number, N]
    """
    start_pos = cuda.shared.array(MAX_ASSETS, dtype=numba.int64)
    end_pos = cuda.shared.array(MAX_ASSETS, dtype=numba.int64)
    old_start_pos = cuda.shared.array(MAX_ASSETS, dtype=numba.int64)
    old_end_pos = cuda.shared.array(MAX_ASSETS, dtype=numba.int64)
    parity_w = cuda.shared.array(MAX_ASSETS, dtype=numba.float64)

    sample_id = cuda.blockIdx.x // num_months
    step_id = cuda.blockIdx.x % num_months

    cluster_num = 1
    old_cluster_num = 1
    start_pos[0] = 0
    end_pos[0] = N
    old_start_pos[0] = 0
    old_end_pos[0] = N

    while cluster_num > 0:
        cluster_num = 0
        for i in range(old_cluster_num):
            start = old_start_pos[i]
            end = old_end_pos[i]
            half = (end - start) // 2
            if half > 0:
                start_pos[cluster_num] = start
                end_pos[cluster_num] = start + half
                cluster_num += 1
            if half > 0:
                start_pos[cluster_num] = start + half
                end_pos[cluster_num] = end
                cluster_num += 1
        for subcluster in range(0, cluster_num, 2):
            left_s = start_pos[subcluster]
            left_e = end_pos[subcluster]
            right_s = start_pos[subcluster+1]
            right_e = end_pos[subcluster+1]
            summ = 0.0
            for i in range(left_s, left_e):
                idd = res_order[sample_id, step_id, i]
                parity_w[i - left_s] = 1.0 / \
                    covariances[sample_id, step_id, idd, idd]
                # print('parity', i,  parity_w[i - left_s])
                summ += parity_w[i - left_s]

            for i in range(left_s, left_e):
                parity_w[i - left_s] *= 1.0 / summ

            summ = 0.0
            for i in range(left_s, left_e):
                idd_x = res_order[sample_id, step_id, i]
                for j in range(left_s, left_e):
                    idd_y = res_order[sample_id, step_id, j]
                    summ += parity_w[i - left_s]*parity_w[j - left_s] * \
                        covariances[sample_id, step_id, idd_x, idd_y]
            left_cluster_var = summ

            summ = 0.0
            for i in range(right_s, right_e):
                idd = res_order[sample_id, step_id, i]
                parity_w[i - right_s] = 1.0 / \
                    covariances[sample_id, step_id, idd, idd]
                summ += parity_w[i - right_s]

            for i in range(right_s, right_e):
                parity_w[i - right_s] *= 1.0 / summ

            summ = 0.0
            for i in range(right_s, right_e):
                idd_x = res_order[sample_id, step_id, i]
                for j in range(right_s, right_e):
                    idd_y = res_order[sample_id, step_id, j]
                    summ += parity_w[i - right_s]*parity_w[j - right_s] * \
                        covariances[sample_id, step_id, idd_x, idd_y]
            right_cluster_var = summ

            alloc_factor = 1 - left_cluster_var / \
                (left_cluster_var + right_cluster_var)

            for i in range(left_s, left_e):
                idd = res_order[sample_id, step_id, i]
                weights[sample_id, step_id, idd] *= alloc_factor
            for i in range(right_s, right_e):
                idd = res_order[sample_id, step_id, i]
                weights[sample_id, step_id, idd] *= 1 - alloc_factor
        for i in range(cluster_num):
            old_start_pos[i] = start_pos[i]
            old_end_pos[i] = end_pos[i]
        old_cluster_num = cluster_num


@cuda.jit
def drawdown_kernel(drawdown, returns, months_starts, window):
    """
    returns, [samples, assets, length]
    drawdown, [smaples, months, assets]
    num_months should be 60 - 12, as the windows size is one year 12 months
    """
    # shared = cuda.shared.array(shape=0, dtype=numba.float64)
    # shared_buffer_size = shared.size
    total_samples, assets, time_len = returns.shape
    _, num_months, _ = drawdown.shape
    i = cuda.threadIdx.x
    sample_id = cuda.blockIdx.x // num_months
    step_id = cuda.blockIdx.x % num_months
    start_id = months_starts[step_id]
    end_id = months_starts[
        step_id +
        window] if step_id + window < months_starts.size else time_len
    for a in range(i, assets, cuda.blockDim.x):
        cumsum = 0.0
        currentMax = 1.0
        minDrawDown = 100.0

        for k in range(start_id, end_id):
            cumsum += returns[sample_id, a, k]
            value = math.exp(cumsum)
            if value > currentMax:
                currentMax = value
            currDrawdown = value / currentMax - 1.0
            if currDrawdown < minDrawDown:
                minDrawDown = currDrawdown
        drawdown[sample_id, step_id, a] = -minDrawDown


def get_drawdown(log_return, total_samples, negative=False, window=12):
    first_sample = log_return['sample_id'].min().item()
    all_dates = log_return[first_sample == log_return['sample_id']]['date']
    all_dates = all_dates.reset_index(drop=True)
    months_start = _get_month_start_pos(all_dates)
    log_return_ma = _get_log_return_matrix(total_samples, log_return)
    if negative:
        log_return_ma = -1.0 * log_return_ma
    _, assets, timelen = log_return_ma.shape
    number_of_threads = 128
    num_months = len(months_start) - window
    if num_months == 0:  # use all the months to compute
        num_months = 1
    number_of_blocks = num_months * total_samples
    drawdown = cupy.zeros((total_samples, num_months, assets))
    drawdown_kernel[(number_of_blocks, ),
                    (number_of_threads, )](drawdown, log_return_ma,
                                           months_start, window)
    return drawdown, all_dates


def get_drawdown_metric(log_return, total_samples):
    first_sample = log_return['sample_id'].min().item()
    all_dates = log_return[first_sample == log_return['sample_id']]['date']
    all_dates = all_dates.reset_index(drop=True)
    months_start = _get_month_start_pos(all_dates)
    # log_return_ma = _get_log_return_matrix(total_samples, log_return)
    port_return_ma = log_return['portfolio'].values.reshape(
        total_samples, 1, -1)
    _, assets, timelen = port_return_ma.shape
    number_of_threads = 128
    window = len(months_start)
    num_months = len(months_start) - window
    if num_months == 0:  # use all the months to compute
        num_months = 1
    number_of_blocks = num_months * total_samples
    drawdown = cupy.zeros((total_samples, num_months, assets))
    drawdown_kernel[(number_of_blocks, ),
                    (number_of_threads, )](drawdown, port_return_ma,
                                           months_start, window)
    return drawdown, all_dates


def get_weights(total_samples, cov, orders, num_months, assets):

    number_of_threads = 1

    number_of_blocks = num_months * total_samples

    weights = cupy.ones((total_samples, num_months, assets))

    HRP_weights[(number_of_blocks,), (number_of_threads,)](
        weights,
        cov,
        orders,
        assets,
        num_months)
    return weights


def get_orders(total_samples, num_months, assets, distance):
    number_of_threads = 1
    number_of_blocks = num_months * total_samples

    output = cupy.zeros((total_samples, num_months, assets-1, 3))
    orders = cupy.zeros((total_samples, num_months, assets), dtype=cupy.int64)
    single_linkage[(number_of_blocks,), (number_of_threads,)](
        output,
        orders,
        distance,
        num_months, assets)
    return orders


def run_bootstrap(v, number_samples=2, block_size=60, number_of_threads=256):
    """
    @v, stock price matrix. [time, stocks]
    @number_samples, number of samples
    @block_size, sample block size
    """
    length, assets = v.shape  # get the time length and the number of assets,
    init_prices = v[0, :].reshape(1, -1, 1)  # initial prices for all assets
    v = cupy.log(v)
    # compute the price difference, dimension of [length -1, assets]
    ref = cupy.diff(v, axis=0)
    # output results
    output = cupy.zeros((number_samples, assets, length))
    # sample starting position, exclusive
    sample_range = length - block_size
    # number of positions to sample to cover the whole seq length
    num_positions = (length - 2) // block_size + 1
    sample_positions = cupy.random.randint(
        0, sample_range,
        num_positions * number_samples)  # compute random starting posistion
    number_of_blocks = len(sample_positions)
    boot_strap[(number_of_blocks,), (number_of_threads,)](
        output,
        ref.T,
        block_size,
        num_positions,
        sample_positions)
    # reshape the results [number_samples, number assets, time]
    # output = output.reshape(number_samples, assets, length)
    # convert it into prices
    return (cupy.exp(output.cumsum(axis=2)) * init_prices)


def _get_month_start_pos(all_dates):
    months_id = all_dates.dt.year*12 + (all_dates.dt.month-1)
    months_id = months_id - months_id.min()
    # months_id = months_id[1:]
    month_start = months_id - months_id.shift(1)
    month_start[0] = 1
    months_start = cupy.where((month_start == 1).values)[0]
    # print('month start position', months_start)
    return months_start


def _get_log_return_matrix(total_samples, log_return):
    col = list(log_return.columns)
    col.remove('date')
    col.remove('sample_id')
    col.remove('year')
    col.remove('month')
    log_return_ma = log_return[col].values
    log_return_ma = log_return_ma.reshape(total_samples, -1, len(col))
    log_return_ma = log_return_ma.transpose((0, 2, 1))
    # sample #, assets dim, time length
    return log_return_ma


def compute_cov_distance(total_samples,
                         log_return,
                         window=12):
    first_sample = log_return['sample_id'].min().item()
    all_dates = log_return[first_sample == log_return['sample_id']]['date']
    all_dates = all_dates.reset_index(drop=True)
    months_start = _get_month_start_pos(all_dates)
    log_return_ma = _get_log_return_matrix(total_samples, log_return)
    _, assets, timelen = log_return_ma.shape
    number_of_threads = 256
    num_months = len(months_start) - window
    # print('num', num_months, len(months_start), window)
    if num_months == 0:  # this case, use all the data to compute
        num_months = 1
    number_of_blocks = num_months * total_samples
    means = cupy.zeros((total_samples, num_months, assets))
    cov = cupy.zeros((total_samples, num_months, assets, assets))
    distance = cupy.zeros(
        (total_samples, num_months, (assets - 1) * assets // 2))

    compute_cov[(number_of_blocks, ), (number_of_threads, ), 0,
                256 * MAX_YEARS * 8](means, cov, distance, log_return_ma,
                                     months_start, num_months, assets, timelen,
                                     window)
    return means, cov, distance, all_dates


def compute_leverage(total_samples,
                     log_return,
                     long_window=59,
                     short_window=19,
                     target_vol=0.05):
    first_sample = log_return['sample_id'].min().item()
    all_dates = log_return[first_sample == log_return['sample_id']]['date']
    all_dates = all_dates.reset_index(drop=True)
    months_start = _get_month_start_pos(all_dates)
    for window in range(len(months_start)):
        if (months_start[window] - long_window) > 0:
            break
    port_return_ma = log_return['portfolio'].values.reshape(total_samples, -1)
    number_of_threads = 256
    num_months = len(months_start) - window
    if num_months == 0:  # this case, use all the data to compute
        num_months = 1
    number_of_blocks = num_months * total_samples
    leverage = cupy.zeros((total_samples, num_months))
    leverage_for_target_vol[(number_of_blocks, ), (number_of_threads, ), 0,
                            256 * MAX_YEARS * 8](leverage, port_return_ma,
                                                 months_start, num_months,
                                                 window, long_window,
                                                 short_window, target_vol)
    return leverage, all_dates, window
