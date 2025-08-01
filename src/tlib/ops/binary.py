import triton
import tlib
import triton.language as tl

from . import util
from typing import Union


@tl.constexpr_function
@tlib.jit(trace=lambda t, c: lambda exprs, tensors_in, op: c(exprs, tuple([t(arg) for arg in tensors_in]), op))
def binary_stage3(exprs, tensors_in, op, backend=None):
    exprs_in, expr_out = tlib.ops.util._unwrap_triton_constexpr(*exprs)
    for root in list(exprs_in) + [expr_out]:
        for expr in root.all():
            if isinstance(expr, tlib.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")

    assert not any(tlib.expr.stage3.is_marked(expr) for root in exprs_in for expr in root.all())
    assert not any(tlib.expr.stage3.is_marked(expr) for expr in expr_out.all())

    # Call tensor factories
    def get_name(s):
        if s == "add":
            return "bias"
        elif s == "multiply":
            return "scale"
        else:
            return s

    tensors_in = [
        tlib.tracer.call_factory(
            tensor,
            expr.shape,
            name=get_name(util._op_to_str(op)),
            init=util._op_to_str(op),
        )
        for tensor, expr in zip(tensors_in, exprs_in)
    ]
    tensors_in = backend.all_to_tensor(tensors_in)

    tensors_out, exprs_out = tlib.vmap_with_axis_stage3(exprs_in, tensors_in, [expr_out], op, backend=backend)
    assert len(tensors_out) == 1 and len(exprs_out) == 1
    return tensors_out[0]


@tl.constexpr_function
def parse(description, tensor_shapes, cse=True):
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

    # Add second input expression from marked subexpressions
    if len(op[0]) == 1 and len(tensor_shapes) == 2:
        # TODO: deprecate this
        op = tlib.expr.stage1.Op(
            [
                tlib.expr.stage1.Args(
                    [
                        tlib.expr.stage1.demark(op[0][0]),
                        tlib.expr.stage1.get_marked(op[0][0]),
                    ]
                ),
            ]
            + list(op[1:])
        )

    for expr in op.all():
        if isinstance(expr, tlib.expr.stage1.Marker):
            raise tlib.SyntaxError(
                description,
                signature.get_pos_for_brackets(list(op.all())),
                "Brackets are not allowed in this function.",
            )

    # Implicitly determine output expression
    if len(op) == 1:
        # Use one of the input expression if contains the axis names of
        # all others and if this choice is unique
        input_args = op[0]
        in_axis_names = [
            {expr.name for expr in root.all() if isinstance(expr, tlib.expr.stage1.NamedAxis)} for root in input_args
        ]

        valid_parents = set()
        for i, parent in enumerate(in_axis_names):
            for j, child in enumerate(in_axis_names):
                if i != j and not child.issubset(parent):
                    break
            else:
                # Found valid parent
                valid_parents.add(input_args[i])

        if len(valid_parents) != 1:
            raise ValueError(f"Could not implicitly determine the output expression for op '{op}'")
        expr_out = next(iter(valid_parents)).__deepcopy__()
        op = tlib.expr.stage1.Op([op[0], tlib.expr.stage1.Args([expr_out])])

    if len(op[0]) != len(tensor_shapes):
        raise ValueError(f"Expected {len(op[0])} input tensors, but got {len(tensor_shapes)}")
    if len(op[1]) != 1:
        raise ValueError(f"Expected 1 output expression, but got {len(op[1])}")

    exprs = tlib.expr.solve(
        tlib.expr.input_equations(op[0], tensor_shapes)
        + tlib.expr.output_equations(op[1])
        + tlib.expr.constraint_equations(parameters),
        cse=cse,
        cse_concat=False,
        signature=signature,
    )[: len(op[0]) + 1]
    exprs_in, expr_out = exprs[:-1], exprs[-1]

    return tlib.ops.util._wrap_triton_constexpr(exprs_in, expr_out)


@triton.jit
def binary(
    description: tl.constexpr,
    tensors,
    op: tl.constexpr,
    cse: tl.constexpr = True,
    kwargs: tl.constexpr | None = None,
) -> Union[tl.tensor | tuple[tl.tensor]]:
    """Applies an element-by-element operation over the given tensors.

    It supports the following shorthand notation:

    * The output is determined implicitly if one of the input expressions contains the named axes
      of all other inputs and if this choice is unique.

      | Example: ``a b, a`` expands to ``a b, a -> a b``.
      | Example: ``b a, b, a`` expands to ``b a, b, a -> b a``.
      | Example: ``a b, b a`` raises an exception.
      | Example: ``a b, a b`` expands to ``a b, a b -> a b``.

    * Bracket notation can be used when passing two input tensors to indicate that the second
      input is a subexpression of the first.

      Example: ``a [b]`` expands to ``a b, b``.

    Args:
        description: Description string for the operation in tlib notation.
        tensors: Input tensors or tensor factories matching the description string.
        op: Backend elemebt-by-element operation. Must accept the same number of tensors
            as specified in the description string and comply with numpy broadcasting rules.
            If ``op`` is a string, retrieves the attribute of ``backend`` with the same name.
        backend: Backend to use for all operations. If None, determines the backend from
            the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of
            computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the elementwise operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Compute a sum of two vectors:

        >>> a, b = np.random.uniform(size=(10,)), np.random.uniform(size=(10,))
        >>> tlib.elementwise("a, a -> a", a, b, op=np.add).shape
        (10,)

        Add a vector on all columns of a matrix:

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> tlib.add("a b, a -> a b", a, b).shape
        (10, 10,)

        Subtract a vector from all rows of a matrix:

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> tlib.subtract("a b, b -> a b", a, b).shape
        (10, 10,)

        Select from one of two choices according to a boolean mask:

        >>> x, mask = (
        ...     np.random.uniform(size=(10, 10)),
        ...     np.random.uniform(size=(10,)),
        ... )
        >>> tlib.where("a, a b, -> a b", mask, x, 0).shape
        (10, 10,)

        Add a bias onto all channels of a tensor:

        >>> x, w = (
        ...     np.random.uniform(size=(4, 16, 16, 64)),
        ...     np.random.uniform(size=(64,)),
        ... )
        >>> tlib.add("b... [c]", x, w).shape
        (4, 16, 16, 64)
    """
    tensor_shapes: tl.constexpr = tlib.ops.util.get_shapes(tensors)
    reprs: tl.constexpr = parse(description, tensor_shapes, cse=cse)
    func: tl.constexpr = binary_stage3(reprs, tensors, op=op)
    return func(*tensors)


@triton.jit
def add(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="add"``"""
    return binary(
        description,
        tensors,
        op="add",
        cse=cse,
    )


@triton.jit
def subtract(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="subtract"``"""
    return binary(
        description,
        tensors,
        op="subtract",
        cse=cse,
    )


@triton.jit
def multiply(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="multiply"``"""
    return binary(
        description,
        tensors,
        op="multiply",
        cse=cse,
    )


@triton.jit
def true_divide(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="true_divide"``"""
    return binary(
        description,
        tensors,
        op="true_divide",
        cse=cse,
    )


@triton.jit
def floor_divide(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="floor_divide"``"""
    return binary(
        description,
        tensors,
        op="floor_divide",
        cse=cse,
    )


@triton.jit
def divide(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="divide"``"""
    return binary(
        description,
        tensors,
        op="divide",
        cse=cse,
    )


@triton.jit
def logical_and(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="logical_and"``"""
    return binary(
        description,
        tensors,
        op="logical_and",
        cse=cse,
    )


@triton.jit
def logical_or(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="logical_or"``"""
    return binary(
        description,
        tensors,
        op="logical_or",
        cse=cse,
    )


@triton.jit
def where(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="where"``"""
    return binary(
        description,
        tensors,
        op="where",
        cse=cse,
    )


@triton.jit
def less(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="less"``"""
    return binary(
        description,
        tensors,
        op="less",
        cse=cse,
    )


@triton.jit
def less_equal(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="less_equal"``"""
    return binary(
        description,
        tensors,
        op="less_equal",
        cse=cse,
    )


@triton.jit
def greater(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="greater"``"""
    return binary(
        description,
        tensors,
        op="greater",
        cse=cse,
    )


@triton.jit
def greater_equal(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="greater_equal"``"""
    return binary(
        description,
        tensors,
        op="greater_equal",
        cse=cse,
    )


@triton.jit
def equal(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="equal"``"""
    return binary(
        description,
        tensors,
        op="equal",
        cse=cse,
    )


@triton.jit
def not_equal(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="not_equal"``"""
    return binary(
        description,
        tensors,
        op="not_equal",
        cse=cse,
    )


@triton.jit
def maximum(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="maximum"``"""
    return binary(
        description,
        tensors,
        op="maximum",
        cse=cse,
    )


@triton.jit
def minimum(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="minimum"``"""
    return binary(
        description,
        tensors,
        op="minimum",
        cse=cse,
    )


@triton.jit
def kl_div(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="kl_div"``"""
    return binary(
        description,
        tensors,
        op="kl_div",
        cse=cse,
    )


@triton.jit
def cross_entropy(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="cross_entropy"``"""
    return binary(
        description,
        tensors,
        op="cross_entropy",
        cse=cse,
    )


@triton.jit
def mse(
    description: tl.constexpr,
    tensors,
    cse: tl.constexpr = True,
) -> tl.tensor:
    """Specialization of :func:`tlib.binary` with ``op="mse"``"""
    return binary(
        description,
        tensors,
        op="mse",
        cse=cse,
    )
