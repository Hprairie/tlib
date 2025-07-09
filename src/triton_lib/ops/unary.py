import triton
import triton.language as tl

import triton_lib as tlib
from triton_lib.ops.vmap_with_axis import vmap_with_axis_stage3


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, tensor_in, op, mask, backend=None: c(exprs, t(tensor_in), op, t(mask)))
def unary_stage3_mask(exprs, tensor_in, op, mask, backend=None):
    expr_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    tensors_out, _ = vmap_with_axis_stage3(expr_in, [tensor_in], expr_out, op, mask, backend=backend)
    return tensors_out[0]


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, tensor_in, op, backend=None: c(exprs, t(tensor_in), op))
def unary_stage3(exprs, tensor_in, op, backend=None):
    expr_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    tensors_out, _ = vmap_with_axis_stage3(expr_in, [tensor_in], expr_out, op, backend=backend)
    return tensors_out[0]


@tl.constexpr_function
@tlib.lru_cache
def parse(description, tensor_shapes, cse=True):
    tensor_shapes = [tensor_shapes]
    description, parameters = tlib.ops.util._clean_description(description)
    signature = tlib.expr.CallSignature(text=description, parameters=parameters)

    op = tlib.expr.stage1.parse_op(description)
    for expr in op.all():
        if isinstance(expr, tlib.expr.stage1.Concatenation):
            raise tlib.SyntaxError(
                description,
                signature.get_pos_for_concatenations(list(op.all())),
                "Concatenations are not allowed in this function.",
            )

    # Implicitly determine output expression
    if len(op) == 1:
        op = tlib.expr.stage1.Op(
            [
                op[0],
                op[0].__deepcopy__(),
            ]
        )

    if len(op[0]) != len(tensor_shapes):
        raise ValueError(f"Expected {len(op[0])} input tensors, but got {len(tensor_shapes)}")

    exprs = tlib.expr.solve(
        tlib.expr.input_equations(op[0], tensor_shapes)
        + tlib.expr.output_equations(op[1])
        + tlib.expr.constraint_equations(parameters),
        cse=cse,
        cse_concat=False,
        signature=signature,
    )[: len(op[0]) + len(op[1])]
    exprs_in, exprs_out = exprs[: len(op[0])], exprs[len(op[0]) :]

    return tlib.ops.util._wrap_triton_constexpr(exprs_in, exprs_out)


@triton.jit
def unary(
    description: tl.constexpr,
    tensor: tl.tensor,
    op: tl.constexpr,
    mask: tl.tensor | None = None,
    reverse: tl.constexpr | None = None,  # For non commutative operations
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Applies a function to the marked axes of the input tensors by passing the ``axis``
    argument and relying on implicit broadcasting rules.

    The function ``op`` must accept input tensors and an ``axis`` argument specifying the
    indices of the axes along which the operation is applied. When the function is applied on
    scalars, the ``axis`` argument is not passed. For multiple input tensors, the function
    must follow
    `Numpy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

    Args:
        description: Description string for the operation in tlib notation.
        tensors: Input tensors or tensor factories matching the description string.
        op: Backend operation. Is called with ``op(tensor, axis=...)``. If ``op`` is a string,
            retrieves the attribute of ``backend`` with the same name.
        kwargs: Additional keyword arguments that are passed to ``op``.
        backend: Backend to use for all operations. If None, determines the backend from the input
            tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the
            result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Reverse order of elements along an axis:

        >>> x = np.random.uniform(size=(16, 20))
        >>> tlib.vmap_with_axis("a [b] -> a [b]", x, op=np.flip).shape
        (16, 20)

        Roll elements along two axes:

        >>> x = np.random.uniform(size=(16, 20))
        >>> tlib.vmap_with_axis(
        ...     "a ([b c]) -> a ([b c])",
        ...     x,
        ...     op=partial(np.roll, shift=(2, 2)),
        ...     b=2,
        ... ).shape
        (16, 20)

        Compute sum along axis:

        >>> x = np.random.uniform(size=(16, 20))
        >>> tlib.vmap_with_axis("a ([b] c) -> c a", x, op=np.sum, b=2).shape
        (16, 20)
    """
    reprs: tl.constexpr = parse(description, tlib.tracer.get_shape(tensor), cse=cse)
    if tl.constexpr(mask is not None):
        tensor = unary_stage3_mask(reprs, tensor, op=op, mask=mask)(tensor, mask)
    else:
        tensor = unary_stage3(reprs, tensor, op=op)(tensor)
    return tensor
