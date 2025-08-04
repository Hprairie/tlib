import triton
import triton.language as tl
import tlib
from functools import partial
from . import util
import numpy as np
from typing import Union
import numpy.typing as npt


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, backend="triton": c(exprs))
def arange_stage3(exprs, backend):
    expr_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    if isinstance(backend, str):
        backend = tlib.backend.get(backend)
    for expr in expr_in.all():
        if isinstance(expr, tlib.expr.stage3.Marker):
            raise ValueError("Marker in input expression not allowed")
    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, tlib.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")

    marked_axes = [
        expr for expr in expr_out.all() if isinstance(expr, tlib.expr.stage3.Axis) and tlib.expr.stage3.is_marked(expr)
    ]
    if len(marked_axes) > 1:
        raise ValueError(f"Expected at most one marked axis, got {len(marked_axes)}")
    ndim = marked_axes[0].value if len(marked_axes) == 1 else 1

    expr_in = util.flatten([expr_in])[0]
    expr_out_flat = util.flatten([expr_out])[0]

    def replace(expr):
        if isinstance(expr, tlib.expr.stage3.Axis) and tlib.expr.stage3.is_marked(expr):
            expr = tlib.expr.stage3.Concatenation([tlib.expr.stage3.Axis(None, 1) for _ in range(ndim)])
            expr = tlib.expr.stage3.Composition(expr)
            return expr

    expr_out_flat_withconcat = tlib.expr.stage3.replace(expr_out_flat, replace)
    expr_out_flat_withconcat = tlib.expr.stage3.demark(expr_out_flat_withconcat)

    arange_tensors = []
    for i, axis in enumerate(expr_in):
        stride = 1
        for j in range(i + 1, len(expr_in)):
            stride *= expr_in[j].value
        arange_tensors.append(backend.arange(axis.value, stride))

    # Broadcast tensors across different axes and sum them
    tensor = None
    for i, arange_tensor in enumerate(arange_tensors):
        axes_to_expand = tuple(j for j in range(len(expr_in)) if j != i)
        broadcasted_tensor = backend.expand_dims(arange_tensor, axes_to_expand)
        if tensor is None:
            tensor = broadcasted_tensor
        else:
            tensor = tensor + broadcasted_tensor
    return tensor


@tl.constexpr_function
def parse(description: str, parameters: dict, cse: bool):
    description, parameters = tlib.ops.util._clean_description(description, parameters)
    assert "->" not in description, "tlib.arange should not specify an output expression"
    assert "[" not in description and "]" not in description, "tlib.arange should not specify a application axis `[ ]`"
    description = description + f" [{len(description.split(" "))}]"

    op = tlib.expr.stage1.parse_op(description)

    # Implicitly determine input expression
    if len(op) == 1:
        op = tlib.expr.stage1.Op(
            [
                tlib.expr.stage1.Args([tlib.expr.stage1.get_unmarked(op[0][0])]),
                op[0],
            ],
        )

    if len(op[0]) != 1:
        raise ValueError(f"Expected 1 input expression, but got {len(op[0])}")
    if len(op[1]) != 1:
        raise ValueError(f"Expected 1 output expression, but got {len(op[1])}")

    marked_expr_out = tlib.expr.stage1.Composition(tlib.expr.stage1.get_marked(op[1][0]))

    def after_stage2(exprs1, exprs2):
        expr_out = exprs1[1]
        out_axes = [
            expr
            for expr in expr_out.all()
            if isinstance(expr, (tlib.expr.stage2.NamedAxis, tlib.expr.stage2.UnnamedAxis))
        ]
        marked_out_axes = [expr for expr in out_axes if tlib.expr.stage2.is_marked(expr)]
        if len(marked_out_axes) > 1:
            raise ValueError(f"Expected at most one marked axis, got {len(marked_out_axes)}")
        ndim = len(out_axes) - len(marked_out_axes)
        return [tlib.expr.Equation(marked_expr_out, np.asarray([ndim]))]

    signature = tlib.expr.CallSignature(text=description, parameters=parameters)
    expr_in, expr_out = tlib.expr.solve(
        tlib.expr.input_equations(op[0])
        + tlib.expr.output_equations(op[1])
        + tlib.expr.constraint_equations(parameters),
        cse=cse,
        after_stage2=after_stage2,
        signature=signature,
    )[:2]

    return tlib.ops.util._wrap_triton_constexpr(expr_in, expr_out)


@triton.jit
def arange(
    description: tl.constexpr,
    parameters: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    exprs: tl.constexpr = parse(description, parameters, cse=cse)
    func: tl.constexpr = arange_stage3(exprs, backend="triton")
    return func()
