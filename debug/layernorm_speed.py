import torch
import triton
import triton.language as tl
import tlib
import tlib.nn as tlnn


def is_hip() -> bool:
    return torch.version.hip is not None


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _layer_norm_forward_kernel_tlib(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    Y_row, rstd, mean = tlnn.layer_norm(X_row, mask=mask, weight=W_row, bias=B_row, eps=eps, return_rstd_mean=True)

    tl.store(Mean_ptr + col_offsets, mean, mask=mask)
    tl.store(RSTD_ptr + col_offsets, rstd, mask=mask)
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    Xmm = tl.where(mask, X_row - mean, 0)
    var = tl.sum(Xmm * Xmm, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    tl.store(Mean_ptr + col_offsets, mean, mask=mask)
    tl.store(RSTD_ptr + col_offsets, rstd, mask=mask)

    Y_row = Xmm * rstd * W_row + B_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


def layer_norm_forward(X, W, B, eps, use_tlib=False):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        kernel_args["grf_mode"] = "large"

    if use_tlib:
        _layer_norm_forward_kernel_tlib[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0),
            B,
            B.stride(0),
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    else:
        _layer_norm_forward_kernel[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0),
            B,
            B.stride(0),
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps


DEVICE = torch.device("cuda")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(2, 14, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["liger", "liger-tlib"],  # Possible values for `line_arg`.
        line_names=["liger", "liger-tlib"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="layernorm-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={"batch": 8},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(batch, size, provider):
    x = torch.rand(batch, size, device=DEVICE, dtype=torch.float32)
    w = torch.rand(size, device=DEVICE, dtype=torch.float32)
    b = torch.rand(size, device=DEVICE, dtype=torch.float32)
    eps = 1e-05
    quantiles = [0.5, 0.2, 0.8]
    if provider == "liger":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer_norm_forward(x, w, b, eps, use_tlib=False), quantiles=quantiles
        )
    if provider == "liger-tlib":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer_norm_forward(x, w, b, eps, use_tlib=True), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
