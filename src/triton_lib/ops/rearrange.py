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
)-> tuple[str, str]:
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

    import pdb; pdb.set_trace()

    return exprs_in, exprs_out


@triton.jit
def rearrange(
    description: str,
    x: tl.tensor,
    y: tl.tensor | None,
    z: tl.tensor | None,
    cse: bool = True,
) -> Union[tl.tensor, tuple[tl.tensor]]:
    """Rearranges the input tensors to match the output expressions.

    Args:
        description: Description string for the operation in tlib notation. Must not contain
            brackets.
        x: Input tensor.
        y: Input tensor.
        z: Input tensor.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.

    Returns:
        The result of the rearrange operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Transpose the row and column axes of a batch of images:

        >>> x = np.random.uniform(size=(4, 64, 48, 3))
        >>> tlib.rearrange("b h w c -> b w h c", x).shape
        (4, 48, 64, 3,)

        Insert new axis (repeats elements along the new axis):

        >>> x = np.random.uniform(size=(10, 10))
        >>> tlib.rearrange("a b -> a c b", x, c=100).shape
        (10, 100, 10,)

        Concatenate two tensors along the first axis:

        >>> a, b = (
        ...     np.random.uniform(size=(10, 10)),
        ...     np.random.uniform(size=(20, 10)),
        ... )
        >>> tlib.rearrange("a b, c b -> (a + c) b", a, b).shape
        (30, 10,)

        Split a tensor:

        >>> x = np.random.uniform(size=(10, 2))
        >>> a, b = tlib.rearrange("a (1 + 1) -> a, a", x)
        >>> a.shape, b.shape
        ((10,), (10,))

        Swap the first and last third of a tensor along a given axis:

        >>> x = np.arange(6)
        >>> tlib.rearrange("(b + c + d) -> (d + c + b)", x, b=2, c=2)
        array([4, 5, 2, 3, 0, 1])
    """
    exprs_in, exprs_out = parse(
        description,
        tlib.tracer.get_shape(x),
        tlib.tracer.get_shape(y),
        tlib.tracer.get_shape(z),
        cse=cse
    )
    # x, y, z, exprs_out = rearrange_stage3(exprs_in, tensors, exprs_out)
    if tl.constexpr(y is None and z is None):
        return x
    elif tl.constexpr(z is None):
        return x, y
    else:
        return x, y, z


rearrange.parse = parse