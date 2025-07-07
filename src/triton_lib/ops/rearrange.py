import triton
import triton.language as tl
import triton.language.core as tlc

import triton_lib as tlib

from typing import Union


@tl.constexpr_function
@tlib.lru_cache
def parse(
    description: str,
    x_shape: tlc.tuple | None,
    y_shape: tlc.tuple | None,
    z_shape: tlc.tuple | None,
    cse: bool,
) -> tuple[tl.constexpr, tl.constexpr]:
    tensor_shapes = [
        shape for shape in [x_shape, y_shape, z_shape] if shape is not None
    ]
    description, parameters = tlib.ops.util._clean_description(description)
    signature = tlib.expr.CallSignature(text=description, parameters=parameters)

    op = tlib.expr.stage1.parse_op(description)
    for expr in op.all():
        if isinstance(expr, tlib.expr.stage1.Marker):
            raise tlib.SyntaxError(
                description,
                signature.get_pos_for_brackets(list(op.all())),
                "Brackets are not allowed in this function.",
            )

    if len(op[0]) != len(tensor_shapes):
        raise ValueError(
            f"Expected {len(op[0])} input tensors, but got {len(tensor_shapes)}"
        )

    exprs = tlib.expr.solve(
        tlib.expr.input_equations(op[0], tensor_shapes)
        + tlib.expr.output_equations(op[1])
        + tlib.expr.constraint_equations(parameters),
        cse=cse,
        signature=signature,
    )[: len(op[0]) + len(op[1])]
    exprs_in, exprs_out = exprs[: len(op[0])], exprs[len(op[0]) :]
    return tlib.ops.util._wrap_triton_constexpr(exprs_in, exprs_out)


@tl.constexpr_function
@tlib.jit(
    trace=lambda t, c: lambda exprs, x, y, z: c(
        exprs, *(t(arg) if arg is not None else None for arg in [x, y, z])
    )
)
def rearrange_stage3(out, x, y, z, backend=None):
    tensors_in = [v for v in [x, y, z] if v is not None]
    exprs_in, exprs_out = tlib.ops.util._unwrap_triton_constexpr(*out)

    if len(exprs_in) != len(tensors_in):
        raise ValueError(
            f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}"
        )
    if any(
        isinstance(expr, tlib.expr.stage3.Marker)
        for root in list(exprs_in) + list(exprs_out)
        for expr in root.all()
    ):
        raise ValueError(f"Marker '{expr}' is not allowed")

    tensors_in = [
        tlib.tracer.call_factory(tensor, expr.shape, name="embedding", init="rearrange")
        for tensor, expr in zip(tensors_in, exprs_in)
    ]
    # tensors_in = backend.all_to_tensor(tensors_in, convert_scalars=True)

    exprs_in, tensors_in = tlib.ops.util.flatten(exprs_in, tensors_in, backend=backend)
    exprs_out_flat = tlib.ops.util.flatten(exprs_out)
    assert all(tlib.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert all(tlib.expr.stage3.is_flat(expr) for expr in exprs_out_flat)
    if len(exprs_in) != len(exprs_out_flat):
        raise ValueError(
            f"Got different number of input ({len(exprs_in)}) and output expressions "
            f"({len(exprs_out_flat)}) (after flattening)"
        )  # TODO:

    # Order inputs to align with output expressions
    indices = tlib.ops.util.assignment(exprs_in, exprs_out_flat)
    exprs_in = [exprs_in[i] for i in indices]
    tensors_in = [tensors_in[i] for i in indices]

    # Transpose and broadcast missing output dimensions
    tensors = [
        tlib.ops.util.transpose_broadcast(expr_in, tensor, expr_out, backend=backend)[0]
        for expr_in, tensor, expr_out in zip(exprs_in, tensors_in, exprs_out_flat)
    ]

    # Unflatten output expressions
    tensors = tlib.ops.util.unflatten(
        exprs_out_flat, tensors, exprs_out, backend=backend
    )

    return tensors


@triton.jit
def rearrange(
    description: tl.constexpr,
    x: tl.tensor,
    y: tl.tensor | None = None,
    z: tl.tensor | None = None,
    cse: tl.constexpr = True,
) -> Union[tl.tensor, tuple[tl.tensor]]:
    out: tl.constexpr = parse(
        description,
        tlib.tracer.get_shape(x),
        tlib.tracer.get_shape(y),
        tlib.tracer.get_shape(z),
        cse=cse,
    )

    func: tl.constexpr = rearrange_stage3(out, x, y, z)
    if tl.constexpr(y is None and z is None):
        return func(x)[0]
    elif tl.constexpr(z is None):
        return func(x, y)
    else:
        return func(x, y, z)
