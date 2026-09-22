# -*- coding: utf-8 -*-

"""
CuPy cupyx.jit backend for batched cluster distance calculations (the 'cuda' stage-4 backend's
default kernel implementation - see geometry.calc_distance_batch_kernel, which prepares the input
arrays and launches path_pair_node_distances / finalize_cluster_pair_distances below).

Written in Python via cupyx.jit.rawkernel so the CUDA source stays inspectable/editable without a
separate C toolchain; a functionally-identical hand-written CUDA C backend exists as
_cuda_distance_rawkernel.py (select it via TRANSPORT_TOOLS_CUDA_KERNEL=raw, see
get_distance_kernels() below) for the last bit of performance margin - see
benchmarks/cuda_distance_backends_benchmark_en.md for the measured difference between the two.

All node data is a flat array with 7 columns per node - x, y, z, layer_id, is_terminal, radius,
rmsf (same layout as geometry._prepare_compiled_path_sets/_prepare_kernel_path_sets) - addressed
here as nodes[node_id * 7 + column]. Its element dtype is float32 by default, or float64 when
TRANSPORT_TOOLS_CUDA_PRECISION=float64 (see resolve_cuda_precision()/get_distance_kernels() below) -
the caller (geometry.calc_distance_batch_kernel) uploads 'nodes' in whichever dtype the selected
kernel was compiled for.
"""

from __future__ import annotations

import os


_JIT_DISTANCE_KERNELS = {}  # cache keyed by precision only - block size is a launch-time argument now
                            # (dynamic shared memory, see get_jit_distance_kernels), not baked into
                            # the compiled kernel, so one compiled pair serves every block size


def resolve_cuda_precision() -> str:
    """
    Resolve the CUDA floating-point precision from TRANSPORT_TOOLS_CUDA_PRECISION - 'float32' by
    default, or 'float64' when the caller explicitly needs CPU/NumPy-matching accuracy (see
    get_jit_distance_kernels for the accuracy/throughput trade-off). This is the single source of
    truth both get_distance_kernels() below (which precision to compile the kernels for) and
    geometry.calc_distance_batch_kernel (which dtype to upload its arrays in) resolve from, so the
    two can never disagree about which dtype is in play for a given run.
    :return: 'float32' or 'float64'
    """

    precision = os.environ.get("TRANSPORT_TOOLS_CUDA_PRECISION", "float32").strip().lower()
    if precision not in ("float32", "float64"):
        raise ValueError(f"Unsupported TRANSPORT_TOOLS_CUDA_PRECISION={precision!r}; use 'float32' or 'float64'")
    return precision


def get_jit_distance_kernels(cp, precision="float32"):
    """
    Compile (or return from cache) the pair of cupyx.jit kernels for the given floating-point
    precision: path_pair_node_distances (per-effective-path-pair task) and
    finalize_cluster_pair_distances (per-cluster-pair aggregation) - see
    geometry.calc_distance_batch_kernel for how they are launched together. Only precision is baked
    into the compiled kernel (through every arithmetic literal below); the CUDA block size is a
    launch-time argument instead (both kernels' reduction buffer is declared with
    jit.shared_memory(float_type, None) - dynamic/extern shared memory, sized in bytes via the
    launch's shared_mem= argument - rather than a size baked in at compile time), so one compiled
    kernel pair serves every block size geometry._bucket_by_block_size produces, and a distinct
    kernel is only compiled per precision requested (at most two: float32, float64).

    precision='float32' (the default - see resolve_cuda_precision()) trades numerical accuracy for
    throughput: many consumer/workstation-class GPUs (this one included - see
    benchmarks/distance_backend_report.tex) cap float64 arithmetic throughput at a small fraction of
    float32's, and the node-distance computation here (surface_distance) is arithmetic dominated. It
    changes what is computed, not just how - the CPU/pure-Python reference backends stay float64
    always, so CUDA results are not bit-exact against them at the default precision (validated to
    stay within the pipeline's own reporting precision, not bit-exactness - see
    test_stage4_cuda_stress.py). Pass TRANSPORT_TOOLS_CUDA_PRECISION=float64 to match the CPU
    backends exactly instead, at reduced throughput on precision-crippled hardware.
    :param cp: the imported cupy module (only used for the device-side dtypes it exposes)
    :param precision: 'float32' (default) or 'float64'
    :return: (path_pair_node_distances, finalize_cluster_pair_distances) compiled kernel functions
    """

    global _JIT_DISTANCE_KERNELS
    if precision in _JIT_DISTANCE_KERNELS:
        return _JIT_DISTANCE_KERNELS[precision]
    if precision not in ("float32", "float64"):
        raise ValueError(f"Unsupported precision={precision!r}; use 'float32' or 'float64'")

    import cupy
    import cupyx.jit as jit

    # every device-side literal below is constructed through this instead of a hardcoded
    # cupy.float64(...), so the same kernel source compiles to either precision
    float_type = cupy.float32 if precision == "float32" else cupy.float64

    @jit.rawkernel(device=True)
    def adjacent(query_layer, candidate_layer, query_last_layer, candidate_last_layer, query_first_terminal):
        # true when a query node (on one path) is close enough in "layer" (radial shell around
        # the tunnel starting point) to a candidate node (on the other path) to be treated as a
        # valid surface-distance correspondence; mirrors the adjacency rule used by the pure-Python
        # LayeredPathSet._compute_distances and the compiled CPU/RawKernel backends
        return (
            cupy.fabs(query_layer - candidate_layer) <= 1.0
            or candidate_layer >= query_first_terminal
            or (query_layer == query_last_layer and candidate_layer > query_layer)
            or (
                query_layer > candidate_last_layer
                and (
                    candidate_layer == candidate_last_layer
                    or candidate_layer == candidate_last_layer - 1.0
                )
            )
            or candidate_layer < 0.0
        )

    @jit.rawkernel(device=True)
    def surface_distance(nodes, node_a, node_b):
        # Euclidean center-to-center distance minus both nodes' radii (columns 0-2 and 5 of the
        # 7-column node layout - see module docstring), clamped at 0 for overlapping/touching nodes
        base_a = node_a * 7
        base_b = node_b * 7
        dx = nodes[base_a + 0] - nodes[base_b + 0]
        dy = nodes[base_a + 1] - nodes[base_b + 1]
        dz = nodes[base_a + 2] - nodes[base_b + 2]
        distance = cupy.sqrt(dx * dx + dy * dy + dz * dz) - nodes[base_a + 5] - nodes[base_b + 5]
        if distance < 0.0:
            distance = 0.0
        return distance

    @jit.rawkernel()
    def path_pair_node_distances(
        nodes,
        path_nodes,
        effective_path_offsets,
        effective_path_lengths,
        last_layers,
        first_terminal_layers,
        task_effective_a,
        task_effective_b,
        task_cluster_a,
        task_cluster_b,
        num_tasks,
        task_values,
    ):
        # one CUDA block per task (= one effective-path-a/effective-path-b pair, see
        # geometry._prepare_kernel_path_pair_tasks); threads within the block stride over the
        # task's length_a + length_b nodes, each accumulating a local sum, then the block reduces
        # to task_values[task_id] via the shared-memory tree reduction below. A task with no valid
        # node correspondence at all writes NaN into task_values[task_id] instead of using a
        # separate invalid-flag array - finalize_cluster_pair_distances relies on this (NaN
        # propagates through its own reduction the same way, see there)
        task_id = jit.blockIdx.x
        thread_id = jit.threadIdx.x
        block_size = jit.blockDim.x
        if task_id >= num_tasks:
            return

        cluster_a = task_cluster_a[task_id]
        cluster_b = task_cluster_b[task_id]
        effective_path_a = task_effective_a[task_id]
        effective_path_b = task_effective_b[task_id]
        length_a = effective_path_lengths[effective_path_a]
        length_b = effective_path_lengths[effective_path_b]
        path_offset_a = effective_path_offsets[effective_path_a]
        path_offset_b = effective_path_offsets[effective_path_b]

        local_sum = float_type(0.0)
        total_length = length_a + length_b

        # threads stride across a virtual [0, length_a + length_b) work range: indices below
        # length_a process one node of path A each (find its closest adjacent node in B, below),
        # indices at/above length_a process one node of path B each (the 'else' branch further
        # down) - together this is a symmetric average over both paths' nodes, matching the
        # reference LayeredPathSet.avg_distance2path_set
        work = thread_id
        while work < total_length:
            if work < length_a:
                node_a = path_nodes[path_offset_a + work]
                layer_a = nodes[node_a * 7 + 3]
                if layer_a >= 0.0:
                    minimum = float_type(1.0e30)
                    node_index_b = 0
                    while node_index_b < length_b:
                        node_b = path_nodes[path_offset_b + node_index_b]
                        layer_b = nodes[node_b * 7 + 3]
                        if adjacent(
                            layer_a,
                            layer_b,
                            last_layers[cluster_a],
                            last_layers[cluster_b],
                            first_terminal_layers[cluster_a],
                        ):
                            distance = surface_distance(nodes, node_a, node_b)
                            if distance < minimum:
                                minimum = distance
                        node_index_b += 1
                    if minimum == 1.0e30:
                        # poison this thread's running sum with NaN instead of a separate 'invalid'
                        # flag/shared array: NaN propagates through every += below and through the
                        # tree reduction's sums[thread_id] += sums[thread_id + stride] (IEEE 754
                        # NaN + anything = NaN), so sums[0] ends up NaN iff any contributing thread
                        # ever hit this branch - exactly the old invalid[0] != 0 condition, with one
                        # shared array instead of two (see the reduction below)
                        local_sum = cupy.nan
                    else:
                        local_sum += minimum
            else:
                node_index_b = work - length_a
                node_b = path_nodes[path_offset_b + node_index_b]
                layer_b = nodes[node_b * 7 + 3]
                if layer_b >= 0.0:
                    # adjacent() is asymmetric (its arguments are a "query" vs a "candidate" side);
                    # first probe under A's perspective (same as the A-node pass above) - if no
                    # A-node qualifies that way, fall back to B's own perspective below so this
                    # B-node is not spuriously marked invalid just because the asymmetric test
                    # was evaluated from the "wrong" side
                    reverse_fallback = True
                    node_index_a = 0
                    while node_index_a < length_a:
                        node_a = path_nodes[path_offset_a + node_index_a]
                        layer_a = nodes[node_a * 7 + 3]
                        if adjacent(
                            layer_a,
                            layer_b,
                            last_layers[cluster_a],
                            last_layers[cluster_b],
                            first_terminal_layers[cluster_a],
                        ):
                            reverse_fallback = False
                            break
                        node_index_a += 1

                    minimum = float_type(1.0e30)
                    node_index_a = 0
                    while node_index_a < length_a:
                        node_a = path_nodes[path_offset_a + node_index_a]
                        layer_a = nodes[node_a * 7 + 3]
                        if reverse_fallback:
                            is_adjacent = adjacent(
                                layer_b,
                                layer_a,
                                last_layers[cluster_b],
                                last_layers[cluster_a],
                                first_terminal_layers[cluster_b],
                            )
                        else:
                            is_adjacent = adjacent(
                                layer_a,
                                layer_b,
                                last_layers[cluster_a],
                                last_layers[cluster_b],
                                first_terminal_layers[cluster_a],
                            )
                        if is_adjacent:
                            distance = surface_distance(nodes, node_a, node_b)
                            if distance < minimum:
                                minimum = distance
                        node_index_a += 1
                    if minimum == 1.0e30:
                        local_sum = cupy.nan  # see the A-side branch above for why
                    else:
                        local_sum += minimum
            work += block_size

        # shared-memory tree reduction: each thread parks its local (possibly NaN-poisoned) partial
        # sum, then pairs of threads combine in halving strides until index 0 holds the block total
        # - NaN + anything = NaN (IEEE 754), so this single addition-only reduction both sums the
        # valid contributions and propagates "any thread was invalid" to sums[0] in one pass, with
        # no separate invalid-flag array needed. Dynamic/extern shared memory (size=None) instead of
        # a compile-time-fixed size: its byte size is supplied by the launch (see
        # geometry.calc_distance_batch_kernel), so this same compiled kernel serves every block size
        # geometry._bucket_by_block_size produces instead of needing one compiled variant per size.
        # Requires block_size to be a power of two (guaranteed by _bucket_by_block_size's own
        # power-of-two rounding).
        sums = jit.shared_memory(float_type, None)
        sums[thread_id] = local_sum
        jit.syncthreads()

        stride = block_size // 2
        while stride > 0:
            if thread_id < stride:
                sums[thread_id] += sums[thread_id + stride]
            jit.syncthreads()
            stride = stride // 2

        if thread_id == 0:
            # only thread 0 (which now holds the full block reduction) writes the task's result;
            # denominator mirrors the reference implementation's "average over nodes minus the two
            # excluded starting-point nodes" normalisation
            denominator = length_a + length_b - 2
            if cupy.isnan(sums[0]) or denominator <= 0:
                task_values[task_id] = cupy.nan
            else:
                task_values[task_id] = sums[0] / denominator

    @jit.rawkernel()
    def finalize_cluster_pair_distances(
        task_values,
        pair_task_start,
        pair_task_end,
        exact_pairs,
        misaligned_pairs,
        distance_cutoff,
        num_pairs,
        output,
    ):
        # one CUDA block per requested cluster pair, averaging that pair's task_values (written by
        # path_pair_node_distances) over pair_task_start[pair_id]:pair_task_end[pair_id] - i.e.
        # over every effective-path-a/effective-path-b combination for that cluster pair. Taken as
        # two independent per-pair arrays (rather than one cumulative offsets array indexed by
        # pair_id and pair_id+1) so geometry.calc_distance_batch_kernel can freely reorder pairs
        # into size buckets without losing either endpoint of a pair's own task range.
        pair_id = jit.blockIdx.x
        thread_id = jit.threadIdx.x
        block_size = jit.blockDim.x
        if pair_id >= num_pairs:
            return

        if misaligned_pairs[pair_id] != 0:
            # short-circuit: misaligned pairs never got any tasks assigned by
            # geometry._prepare_kernel_path_pair_tasks, so there is nothing to average here
            if thread_id == 0:
                output[pair_id] = 999.0
            return

        start = pair_task_start[pair_id]
        end = pair_task_end[pair_id]
        num_path_pairs = end - start
        local_sum = float_type(0.0)

        # task_values[task_id] is already NaN for any task path_pair_node_distances could not
        # resolve, so summing it directly NaN-poisons local_sum (and, through the same reduction
        # trick as above, sums[0]) exactly when any task in this pair's range was invalid - no
        # separate invalid-flag array needed here either
        task_id = start + thread_id
        while task_id < end:
            local_sum += task_values[task_id]
            task_id += block_size

        sums = jit.shared_memory(float_type, None)
        sums[thread_id] = local_sum
        jit.syncthreads()

        stride = block_size // 2
        while stride > 0:
            if thread_id < stride:
                sums[thread_id] += sums[thread_id + stride]
            jit.syncthreads()
            stride = stride // 2

        if thread_id == 0:
            # non-exact (early-terminated) pairs get cut off at the same '3x cutoff' distance-sum
            # threshold as the compiled CPU backend (geometry.calc_distance_batch_compiled /
            # _distance_kernel.c's compute_pair) - anything past it is reported as the 999.0 "far"
            # sentinel instead of the true average, since exact_pairs=False callers only care
            # whether a pair is within range, not its precise distance beyond it
            if cupy.isnan(sums[0]) or num_path_pairs <= 0:
                output[pair_id] = cupy.nan
            elif exact_pairs[pair_id] == 0 and sums[0] > 3.0 * distance_cutoff * num_path_pairs:
                output[pair_id] = 999.0
            else:
                output[pair_id] = sums[0] / num_path_pairs

    _JIT_DISTANCE_KERNELS[precision] = (
        path_pair_node_distances,
        finalize_cluster_pair_distances,
    )
    return _JIT_DISTANCE_KERNELS[precision]


def get_distance_kernels(cp):
    """
    Return the (path_pair_node_distances, finalize_cluster_pair_distances) kernel pair for the
    selected CUDA implementation - the Python-readable cupyx.jit backend defined above (default),
    or the equivalent hand-written CUDA C RawKernel backend, selected by setting the
    TRANSPORT_TOOLS_CUDA_KERNEL environment variable to 'jit' (default) or 'raw'. The floating-point
    precision used internally is controlled independently via resolve_cuda_precision() ('float32'
    default, or 'float64' - see get_jit_distance_kernels for the accuracy/throughput trade-off; not
    yet wired through to the 'raw' RawKernel backend, which does not currently exist in this repo -
    if restored, it should adopt the same dynamic-shared-memory design as get_jit_distance_kernels
    so it is also block-size-independent at compile time). This is the single entry point
    geometry.py's _get_cuda_distance_kernels() calls.
    :param cp: the imported cupy module, forwarded to whichever backend is selected
    :return: (path_pair_node_distances, finalize_cluster_pair_distances) kernel functions
    """

    backend = os.environ.get("TRANSPORT_TOOLS_CUDA_KERNEL", "jit").strip().lower()
    if backend == "raw":
        from transport_tools.libs import _cuda_distance_rawkernel
        return _cuda_distance_rawkernel.get_distance_kernels(cp)
    if backend != "jit":
        raise ValueError(f"Unsupported TRANSPORT_TOOLS_CUDA_KERNEL={backend!r}; use 'jit' or 'raw'")
    return get_jit_distance_kernels(cp, resolve_cuda_precision())


def get_distance_kernel(cp):
    """Return the first selected CUDA kernel for backward-compatible imports."""

    return get_distance_kernels(cp)[0]
