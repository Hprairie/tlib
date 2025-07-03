import triton
import triton.language as tl
import triton.language.core as tlc

import triton_lib as tlib

from typing import Union


@tl.constexpr_function
def parse(
    description: str,
    x_shape: tlc.tuple | None,
    y_shape: tlc.tuple | None,
    z_shape: tlc.tuple | None,
    cse: bool,
)-> tuple[tl.constexpr, tl.constexpr]:
    tensor_shapes = [shape for shape in [x_shape, y_shape, z_shape] if shape is not None]
    description, parameters = tlib.ops.util._clean_description(
        description
    )
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
        raise ValueError(f"Expected {len(op[0])} input tensors, but got {len(tensor_shapes)}")

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
@tlib.ji
def rearrange_stag3(out, x, y, z):
    exprs_in, exprs_out = tlib.ops.util._unwrap_triton_constexpr(*out)
    tensors_in = [tensor for tensor in [x, y, z] if tensor is not None]

    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
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

    import pdb; pdb.set_trace()
    zz = x + y


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
        cse=cse
    )
    x, y, z = rearrange_stag3(out, x, y, z)(x,y,z)
    # if tl.constexpr(y is None and z is None):
    #     return x
    # elif tl.constexpr(z is None):
    #     return x, y
    # else:
    #     return x, y, z