import triton
import triton.language as tl
import tlib

_any = any  # Is overwritten below


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, tensor_in, op, mask, backend=None: c(exprs, t(tensor_in), op, t(mask)))
def reduce_stage3_mask(exprs, tensor_in, op, mask, backend=None):
    expr_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, tlib.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    tensors_out, exprs_out = tlib.ops.vmap_with_axis_stage3(
        [expr_in], [tensor_in], [expr_out], op, mask, backend=backend
    )
    return tensors_out[0]


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, tensor_in, op, backend=None: c(exprs, t(tensor_in), op))
def reduce_stage3(exprs, tensor_in, op, backend=None):
    expr_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, tlib.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    tensors_out, exprs_out = tlib.ops.vmap_with_axis_stage3([expr_in], [tensor_in], [expr_out], op, backend=backend)
    return tensors_out[0]


@tl.constexpr_function
def parse(description, tensor_shape, keepdims=None, cse=True):
    description, parameters = tlib.ops.util._clean_description(description, None)
    signature = tlib.expr.CallSignature(text=description, parameters=parameters)

    op = tlib.expr.stage1.parse_op(description)
    for expr in op.all():
        if isinstance(expr, tlib.expr.stage1.Concatenation):
            raise tlib.SyntaxError(
                description,
                signature.get_pos_for_concatenations(list(op.all())),
                "Concatenations are not allowed in this function.",
            )

    if len(op) == 1:
        expr_in = tlib.expr.solve(
            tlib.expr.input_equations(op[0], [tensor_shape]) + tlib.expr.constraint_equations(parameters),
            cse=cse,
            cse_in_markers=True,
            signature=signature,
        )[0]

        if not _any(isinstance(expr, tlib.expr.stage3.Marker) for expr in expr_in.all()):
            raise ValueError("No axes are marked for reduction")

        # Determine output expressions by removing markers from input expressions
        def replace(expr):
            if isinstance(expr, tlib.expr.stage3.Marker):
                if keepdims:
                    return [tlib.expr.stage3.Axis(None, 1)]
                else:
                    return []

        expr_out = tlib.expr.stage3.replace(expr_in, replace)

    else:
        if keepdims is not None:
            raise ValueError("keepdims cannot be given when using '->'")

        if len(op[0]) != 1:
            raise ValueError(f"Expected 1 input expression, but got {len(op[0])}")
        if len(op[1]) != 1:
            raise ValueError(f"Expected 1 output expression, but got {len(op[1])}")

        expr_in, expr_out = tlib.expr.solve(
            tlib.expr.input_equations(op[0], [tensor_shape])
            + tlib.expr.output_equations(op[1])
            + tlib.expr.constraint_equations(parameters),
            cse=cse,
            cse_in_markers=True,
            signature=signature,
        )[:2]

        # If no axes are marked for reduction in expr_in, mark all axes that
        # don't appear in expr_out
        if not _any(tlib.expr.stage3.is_marked(expr) for expr in expr_in.all()):
            axes_names_out = {axis.name for axis in expr_out.all() if isinstance(axis, tlib.expr.stage3.Axis)}
            expr_in = tlib.expr.stage3.mark(
                expr_in,
                lambda expr: isinstance(expr, tlib.expr.stage3.Axis) and expr.name not in axes_names_out,
            )

    return tlib.ops.util._wrap_triton_constexpr(expr_in, expr_out)


@triton.jit
def reduce(
    description: tl.constexpr,
    tensor: tl.tensor,
    op: tl.constexpr,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Applies a reduction operation on the given tensors.

    The operation reduces all marked axes in the input to a single scalar. It supports
    the following shorthand notation:

    * When no brackets are found, brackets are placed implicitly around all axes that do not
      appear in the output.

      Example: ``a b c -> a c`` resolves to ``a [b] c -> a c``.

    * When no output is given, it is determined implicitly by removing marked subexpressions
      from the input.

      Example: ``a [b] c`` resolves to ``a [b] c -> a c``.

    Args:
        description: Description string for the operation in tlib notation.
        tensor: Input tensor or tensor factory matching the description string.
        op: Backend reduction operation. Is called with ``op(tensor, axis=...)``. If ``op`` is
            a string, retrieves the attribute of ``backend`` with the same name.
        keepdims: Whether to replace marked expressions with 1s instead of dropping them. Must
            be None when ``description`` already contains an output expression. Defaults to None.
        backend: Backend to use for all operations. If None, determines the backend from the
            input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of
            computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the reduction operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Compute mean along rows of a matrix:

        >>> x = np.random.uniform(size=(16, 20))
        >>> tlib.mean("a b -> b", x).shape
        (20,)
        >>> tlib.mean("[a] b -> b", x).shape
        (20,)
        >>> tlib.mean("[a] b", x).shape
        (20,)

        Compute sum along rows of a matrix and broadcast to the original shape:

        >>> x = np.random.uniform(size=(16, 20))
        >>> tlib.sum("[a] b -> a b", x).shape
        (16, 20,)

        Sum pooling with kernel size 2:

        >>> x = np.random.uniform(size=(4, 16, 16, 3))
        >>> tlib.sum("b (s [s2])... c", x, s2=2).shape
        (4, 8, 8, 3)

        Compute variance per channel over an image:

        >>> x = np.random.uniform(size=(256, 256, 3))
        >>> tlib.var("[...] c", x).shape
        (3,)
    """
    reprs: tl.constexpr = parse(
        description,
        tlib.tracer.get_shape(tensor),
        keepdims=keepdims,
        cse=cse,
    )
    if tl.constexpr(mask is not None):
        tensor = reduce_stage3_mask(reprs, tensor, op=op, mask=mask)(tensor, mask)
    else:
        tensor = reduce_stage3(reprs, tensor, op=op)(tensor)
    return tensor


@triton.jit
def sum(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="sum"``"""
    return reduce(
        description,
        tensor,
        op="sum",
        mask=mask,
        keepdims=keepdims,
        cse=cse,
    )


@triton.jit
def mean(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="mean"``"""
    return reduce(
        description,
        tensor,
        op="mean",
        mask=mask,
        keepdims=keepdims,
        cse=cse,
    )


@triton.jit
def var(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="var"``"""
    return reduce(
        description,
        tensor,
        op="var",
        mask=mask,
        keepdims=keepdims,
        cse=cse,
    )


@triton.jit
def std(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="std"``"""
    return reduce(
        description,
        tensor,
        op="std",
        mask=mask,
        keepdims=keepdims,
        cse=cse,
    )


@triton.jit
def prod(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="prod"``"""
    return reduce(
        description,
        tensor,
        op="prod",
        mask=mask,
        keepdims=keepdims,
        cse=cse,
    )


@triton.jit
def count_nonzero(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="count_nonzero"``"""
    return reduce(
        description,
        tensor,
        op="count_nonzero",
        mask=mask,
        keepdims=keepdims,
        cse=cse,
    )


@triton.jit
def any(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="any"``"""
    return reduce(description, tensor, op="any", mask=mask, keepdims=keepdims, cse=cse)


@triton.jit
def all(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="all"``"""
    return reduce(description, tensor, op="all", mask=mask, keepdims=keepdims, cse=cse)


@triton.jit
def max(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="max"``"""
    return reduce(description, tensor, op="max", mask=mask, keepdims=keepdims, cse=cse)


@triton.jit
def min(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="min"``"""
    return reduce(description, tensor, op="min", mask=mask, keepdims=keepdims, cse=cse)


@triton.jit
def argmax(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="argmax"``"""
    return reduce(description, tensor, op="argmax", mask=mask, keepdims=keepdims, cse=cse)


@triton.jit
def argmin(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="argmin"``"""
    return reduce(description, tensor, op="argmin", mask=mask, keepdims=keepdims, cse=cse)


@triton.jit
def logsumexp(
    description: tl.constexpr,
    tensor: tl.tensor,
    mask: tl.tensor | None = None,
    keepdims: tl.constexpr | None = None,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.reduce` with ``op="logsumexp"``"""
    return reduce(description, tensor, op="logsumexp", mask=mask, keepdims=keepdims, cse=cse)
