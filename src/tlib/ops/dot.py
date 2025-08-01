import triton
import triton.language as tl

import tlib
from . import util


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, tensors_in: c(exprs, tuple([t(arg) for arg in tensors_in])))
def dot_stage3(exprs, tensors_in, backend=None):
    # Call tensor factories
    exprs_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    output_axis_names = {a.name for a in expr_out.all() if isinstance(a, tlib.expr.stage3.Axis)}

    def get_fans(idx):
        other_input_axis_names = {
            a.name
            for i, expr_in in enumerate(exprs_in)
            for a in expr_in.all()
            if i != idx and isinstance(a, tlib.expr.stage3.Axis)
        }
        in_axis = []
        out_axis = []
        batch_axis = []
        for i, child in enumerate(exprs_in[idx]):
            any_in_other_input = any(
                isinstance(a, tlib.expr.stage3.Axis) and a.name in other_input_axis_names for a in child.all()
            )
            any_in_output = any(
                isinstance(a, tlib.expr.stage3.Axis) and a.name in output_axis_names for a in child.all()
            )
            if any_in_other_input and not any_in_output:
                in_axis.append(i)
            elif any_in_output and not any_in_other_input:
                out_axis.append(i)
            else:
                batch_axis.append(i)
        return {
            "in_axis": tuple(in_axis),
            "out_axis": tuple(out_axis),
            "batch_axis": tuple(batch_axis),
        }

    tensors_in = [
        tlib.tracer.call_factory(tensor, expr.shape, **get_fans(i), name="weight", init="dot")
        for i, (tensor, expr) in enumerate(zip(tensors_in, exprs_in))
    ]
    tensors_in = backend.all_to_tensor(tensors_in)

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend=backend)
    expr_out_flat = util.flatten([expr_out])[0]
    assert all(tlib.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert tlib.expr.stage3.is_flat(expr_out_flat)

    # Apply einsum
    einsum_variables = {}

    def get_einsum_variable(key):
        if key in einsum_variables:
            return einsum_variables[key]
        else:
            v = chr(ord("a") + len(einsum_variables))
            if ord(v) > ord("z"):
                raise ValueError(f"Only supports up to {ord('z') - ord('a') + 1} unique input axes")
            einsum_variables[key] = v
            return v

    def to_einsum(axes):
        return "".join(get_einsum_variable(a.name) for a in axes)

    input_axis_names = {a.name for expr in exprs_in for a in tlib.expr.stage3.get_axes(expr)}

    einsum_str = (
        ",".join(to_einsum(tlib.expr.stage3.get_axes(expr)) for expr in exprs_in)
        + "->"
        + to_einsum([a for a in tlib.expr.stage3.get_axes(expr_out_flat) if a.name in input_axis_names])
    )

    tensor = backend.einsum(einsum_str, *tensors_in)
    expr = tlib.expr.stage3.List(
        [a.__deepcopy__() for a in tlib.expr.stage3.get_axes(expr_out_flat) if a.name in input_axis_names]
    )

    # Transpose and broadcast missing output dimensions
    tensor = util.transpose_broadcast(expr, tensor, expr_out_flat, backend=backend)[0]

    # Unflatten output expression
    tensor = backend.reshape(tensor, expr_out.shape)

    return tensor


@tl.constexpr_function
def parse(description, tensor_shapes, cse=True):
    description, parameters = tlib.ops.util._clean_description(description)

    signature = tlib.expr.CallSignature(text=description, parameters=parameters)
    op = tlib.expr.stage1.parse_op(description)

    # Implicitly determine second input expression
    if len(op[0]) == 1 and len(tensor_shapes) == 2:
        # TODO: deprecate this
        for root in [op[0][0], op[1][0]]:
            for expr in root.all():
                if (
                    isinstance(expr, tlib.expr.stage1.UnnamedAxis)
                    and expr.value != 1
                    and tlib.expr.stage1.is_marked(expr)
                ):
                    raise ValueError(f"Cannot mark unnamed non-trivial axes, but found {expr}")

        # Create second input expression from ordered list of marked axes
        names = set()
        expr_in2 = []
        for root in [op[0][0], op[1][0]]:
            for expr in root.all():
                if (
                    isinstance(expr, tlib.expr.stage1.NamedAxis)
                    and tlib.expr.stage1.is_marked(expr)
                    and expr.name not in names
                ):
                    names.add(expr.name)

                    # Copy axis
                    expr2 = expr.__deepcopy__()

                    # Apply the same ellipses
                    parent = expr
                    while parent.parent is not None:
                        if isinstance(parent, tlib.expr.stage1.Ellipsis):
                            expr2 = tlib.expr.stage1.Ellipsis(expr2, ellipsis_id=parent.ellipsis_id)
                        parent = parent.parent

                    # Append to second output expression
                    expr_in2.append(expr2)
        expr_in2 = tlib.expr.stage1.List(expr_in2)

        op = tlib.expr.stage1.Op(
            [
                tlib.expr.stage1.Args(
                    [
                        tlib.expr.stage1.demark(op[0][0]),
                        expr_in2,
                    ]
                ),
                tlib.expr.stage1.Args(
                    [
                        tlib.expr.stage1.demark(op[1][0]),
                    ]
                ),
            ]
        )

    if len(op[0]) != len(tensor_shapes):
        raise ValueError(f"Expected {len(op[0])} input tensor(s), but got {len(tensor_shapes)}.")
    if len(op[1]) != 1:
        raise ValueError(f"Expected 1 output expression, but got {len(op[1])}.")

    # Check for invalid expressions
    for expr in op.all():
        if isinstance(expr, tlib.expr.stage1.Concatenation):
            raise tlib.SyntaxError(
                description,
                signature.get_pos_for_concatenations(list(op.all())),
                "Concatenations are not allowed in this function.",
            )
    for expr in op[1].all():
        if isinstance(expr, tlib.expr.stage1.Marker):
            raise tlib.SyntaxError(
                description,
                signature.get_pos_for_brackets(list(op[1].all())),
                "Brackets are not allowed in the output expression of this function.",
            )

    exprs = tlib.expr.solve(
        tlib.expr.input_equations(op[0], tensor_shapes)
        + tlib.expr.output_equations(op[1])
        + tlib.expr.constraint_equations(parameters),
        cse=cse,
        cse_concat=False,
        signature=signature,
    )[: len(op[0]) + 1]
    exprs_in, expr_out = exprs[:-1], exprs[-1]

    # If no axes are marked, mark all axes that are not in the output
    if not any(tlib.expr.stage3.is_marked(expr) for expr_in in exprs_in for expr in expr_in.all()):
        axes_names_out = {axis.name for axis in expr_out.all() if isinstance(axis, tlib.expr.stage3.Axis)}
        exprs_in = [
            tlib.expr.stage3.mark(
                expr_in,
                lambda expr: isinstance(expr, tlib.expr.stage3.Axis) and expr.name not in axes_names_out,
            )
            for expr_in in exprs_in
        ]

    # Check for invalid expressions
    out_axis_names = {a.name for a in expr_out.all() if isinstance(a, tlib.expr.stage3.Axis)}
    invalid_axes = []
    for expr_in in exprs_in:
        for axis in expr_in.all():
            if isinstance(axis, tlib.expr.stage3.Axis):
                is_reduced_axis = axis.name not in out_axis_names
                is_marked = tlib.expr.stage3.is_marked(axis)
                if is_reduced_axis and not is_marked and axis.value != 1:
                    invalid_axes.append(axis)
    if len(invalid_axes) > 0:
        pos = []
        for axis in invalid_axes:
            pos.extend(range(axis.begin_pos, axis.end_pos))
        raise tlib.SyntaxError(
            description,
            pos,
            f"Axis {axis} does not appear in the output expression of tlib.dot and must "
            "therefore be marked with brackets.",
        )

    return tlib.ops.util._wrap_triton_constexpr(exprs_in, expr_out)


@triton.jit
def dot(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Computes a general dot-product of the input tensors.

    The following shorthand notation is supported:

    * When no brackets are found, brackets are placed implicitly around all axes that do not
      appear in the output.

      Example: ``a b, b c -> a c`` expands to ``a [b], [b] c -> a c``

    * When given two input tensors, the expression of the second input is determined implicitly
      from the marked axes in the input and output expression.

      Example: ``a [b] -> a [c]`` expands to ``a b, b c -> a c``

      Axes marked multiple times appear only once in the implicit second input expression.

      Example: ``[a b] -> [a c]`` expands to ``a b, a b c -> a c``

    The function additionally passes the ``in_axes``, ``out_axes`` and ``batch_axes`` arguments
    to tensor factories that can be used to determine the fan-in and fan-out of a neural network
    layer and initialize weights accordingly (see e.g. `jax.nn.initializers.lecun_normal
    <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.lecun_normal.html#jax.nn.initializers.lecun_normal>`_)

    Args:
        description: Description string for the operation in tlib notation.
        tensors: Input tensors or tensor factories matching the description string.
        backend: Backend to use for all operations. If None, determines the backend from the
            input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of computing
            the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the dot-product operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Compute an inner product between two vectors:

        >>> a, b = np.random.uniform(size=(10,)), np.random.uniform(size=(10,))
        >>> tlib.dot("a, a ->", a, b).shape
        ()

        Compute a matrix-vector product:

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> tlib.dot("a b, b -> a", a, b).shape
        (10,)
        >>> tlib.dot("a [b] -> a", a, b).shape
        (10,)
        >>> tlib.dot("a [b->]", a, b).shape
        (10,)

        Compute a vector-matrix product:

        >>> a, b = np.random.uniform(size=(10,)), np.random.uniform(size=(10, 10))
        >>> tlib.dot("a, a b -> b", a, b).shape
        (10,)
        >>> tlib.dot("[a] -> [b]", a, b).shape
        (10,)
        >>> tlib.dot("[a->b]", a, b).shape
        (10,)

        Multiply a tensor with a weight matrix:

        >>> x, w = (
        ...     np.random.uniform(size=(4, 16, 16, 64)),
        ...     np.random.uniform(
        ...         size=(
        ...             64,
        ...             32,
        ...         )
        ...     ),
        ... )
        >>> tlib.dot("b... [c1->c2]", x, w).shape
        (4, 16, 16, 32)
    """
    if tl.constexpr(isinstance(tensors, tl.tensor)):
        tensors = (tensors,)
    tensor_shapes: tl.constexpr = tlib.ops.util.get_shapes(tensors)
    exprs: tl.constexpr = parse(description, tensor_shapes, cse=cse)
    func: tl.constexpr = dot_stage3(exprs, tensors)
    tensor = func(*tensors)
    return tensor
