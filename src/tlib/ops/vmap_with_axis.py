import tlib
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
