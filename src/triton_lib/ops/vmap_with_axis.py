import triton
import triton.language as tl

import triton_lib as tlib
from . import util


@tlib.jit(
    trace=lambda t, c: lambda exprs_in, tensors_in, exprs_out, op, mask, kwargs={}, backend=None: c(
        exprs_in,
        [t(x) for x in tensors_in],
        exprs_out,
        op,
        t(mask) if mask is not None else mask,
        kwargs,
    )
)
def vmap_with_axis_stage3(exprs_in, tensors_in, exprs_out, op, mask=None, kwargs=None, backend=None):
    if kwargs is None:
        kwargs = {}
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
    if len(set(exprs_out)) != 1:
        raise ValueError("All output expressions must be the same")
    for root in list(exprs_in) + list(exprs_out):
        for expr in root.all():
            if isinstance(expr, tlib.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    if len(exprs_out) > 1:
        raise ValueError("Only one output tensor allowed")
    if all(tlib.tracer.is_scalar(tensor) for tensor in tensors_in):
        raise ValueError("At least one input tensor must be a non-scalar")  # TODO: support this
    kwargs = {**kwargs}

    # Call tensor factories
    tensors_in = [
        tlib.tracer.call_factory(tensor, expr.shape, backend=backend) for tensor, expr in zip(tensors_in, exprs_in)
    ]
    # tensors_in = backend.all_to_tensor(tensors_in)

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend=backend)
    in_axis_names = {axis.name for expr in exprs_in for axis in expr}

    def is_broadcast_axis(expr):
        return isinstance(expr, tlib.expr.stage3.Axis) and expr.name not in in_axis_names

    exprs_out_flat = util.flatten(exprs_out)
    exprs_out_flat_without_broadcast = [tlib.expr.stage3.remove(expr, is_broadcast_axis) for expr in exprs_out_flat]

    transpose_first = len(exprs_in) > 1

    # Ensure that axis markings are consistent
    def is_vmapped(expr):
        return not tlib.expr.stage3.is_marked(expr)

    vmapped_axis_names = {
        v.name for root in list(exprs_in) + list(exprs_out_flat_without_broadcast) for v in root if is_vmapped(v)
    }
    for root in list(exprs_in) + list(exprs_out_flat_without_broadcast):
        for v in root:
            if (v.name in vmapped_axis_names) != is_vmapped(v):
                raise ValueError(f"Axis {v.name} appears both as vmapped and non-vmapped")

    marked_input_axes = {
        axis.name
        for expr_in in exprs_in
        for axis in expr_in.all()
        if isinstance(axis, tlib.expr.stage3.Axis) and tlib.expr.stage3.is_marked(axis)
    }
    marked_output_axes = {
        axis.name
        for expr_out in exprs_out_flat_without_broadcast
        for axis in expr_out.all()
        if isinstance(axis, tlib.expr.stage3.Axis) and tlib.expr.stage3.is_marked(axis)
    }
    if marked_output_axes.difference(marked_input_axes):
        raise ValueError("Marked output axes must be a subset of marked input axes")

    if transpose_first:
        # Transpose and insert trivial axes
        if marked_input_axes != marked_output_axes:
            raise ValueError("When using multiple input tensors the same axes must be marked in all tensors")
        x = [
            (
                (tensor_in, expr_in)
                if tlib.tracer.is_scalar(tensor_in)
                else util.transpose_broadcast(
                    expr_in,
                    tensor_in,
                    exprs_out_flat_without_broadcast[0],
                    broadcast=False,
                    backend=backend,
                )
            )
            for expr_in, tensor_in in zip(exprs_in, tensors_in)
        ]
        tensors_in = [x[0] for x in x]
        exprs_in = [x[1] for x in x]
        assert len({len(expr) for expr in exprs_in if len(expr) > 0}) == 1
        marked_input_axes = {
            axis.name
            for expr_in in exprs_in
            for axis in expr_in.all()
            if isinstance(axis, tlib.expr.stage3.Axis) and tlib.expr.stage3.is_marked(axis)
        }
        exprs_op_output = exprs_out_flat_without_broadcast
    else:
        assert len(exprs_in) == 1  # TODO: see above
        expr_in = exprs_in[0]

        def to_op_output(expr_out_flat_wb):
            axis_names = {axis.name for axis in expr_out_flat_wb.all() if isinstance(axis, tlib.expr.stage3.Axis)}
            new_axes = []
            for axis in expr_in.all():
                if isinstance(axis, tlib.expr.stage3.Axis) and axis.name in axis_names:
                    if isinstance(axis.parent, tlib.expr.stage3.Marker):
                        axis = axis.parent
                    new_axes.append(axis)
            return tlib.expr.stage3.List.maybe(new_axes)

        exprs_op_output = [to_op_output(expr_out_flat_wb) for expr_out_flat_wb in exprs_out_flat_without_broadcast]

    # Add axis argument
    if transpose_first:
        axis_indices = tuple(
            i for i, axis in enumerate(exprs_out_flat_without_broadcast[0]) if axis.name in marked_input_axes
        )
    else:
        axes_in = [list(expr) for expr in exprs_in]
        axis_indices = tuple(
            i for i in range(len(axes_in[0])) if any(axes_in[i].name in marked_input_axes for axes_in in axes_in)
        )
    if len(axis_indices) > 0:
        kwargs["axis"] = axis_indices if len(axis_indices) > 1 else axis_indices[0]

    # Apply operation
    if isinstance(op, str):
        op = getattr(backend, op)
    elif not isinstance(op, tlib.tracer.Tracer):
        concrete_op = op
        op = lambda *args, **kwargs: tlib.tracer.apply(
            concrete_op,
            args=args,
            kwargs=kwargs,
            output=(
                [tlib.tracer.Tensor(expr.shape) for expr in exprs_op_output]
                if len(exprs_op_output) > 1
                else tlib.tracer.Tensor(exprs_op_output[0].shape)
            ),
        )

    if mask is None:
        tensors_out = op(*tensors_in, **kwargs)
    else:
        tensors_out = op(*tensors_in, mask=mask, **kwargs)

    if not isinstance(tensors_out, (tuple, list)):
        tensors_out = (tensors_out,)
    if len(tensors_out) != len(exprs_out_flat_without_broadcast):
        raise ValueError(
            f"Expected {len(exprs_out_flat_without_broadcast)} output tensor(s), " f"got {len(tensors_out)}"
        )

    # Transpose and broadcast missing output dimensions
    tensors_out = [
        util.transpose_broadcast(expr_in, tensor_out, expr_out, backend=backend)[0]
        for expr_in, tensor_out, expr_out in zip(exprs_op_output, tensors_out, exprs_out_flat)
    ]

    # Unflatten output expressions
    tensors_out = util.unflatten(exprs_out_flat, tensors_out, exprs_out, backend=backend)

    return tensors_out, exprs_out


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, tensor_in, op, mask, backend=None: c(exprs, t(tensor_in), op, t(mask)))
def vmap_stage3_mask(exprs, tensor_in, op, mask, backend=None):
    expr_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, tlib.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    tensors_out, _ = vmap_with_axis_stage3([expr_in], [tensor_in], [expr_out], op, mask, backend=backend)
    return tensors_out[0]


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, tensor_in, op, backend=None: c(exprs, t(tensor_in), op))
def vmap_stage3(exprs, tensor_in, op, backend=None):
    expr_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, tlib.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    tensors_out, _ = vmap_with_axis_stage3([expr_in], [tensor_in], [expr_out], op, backend=backend)
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
def vmap_with_axis(
    description: str,
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
    reprs = parse(description, tlib.tracer.get_shape(tensor), cse=cse)
    if tl.constexpr(mask is not None):
        tensor = vmap_stage3_mask(reprs, tensor, op=op, mask=mask)(tensor, mask)
    else:
        tensor = vmap_stage3(reprs, tensor, op=op)(tensor)
    return tensor
