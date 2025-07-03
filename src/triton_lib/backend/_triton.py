from .base import Backend, associative_binary_to_nary
import triton_lib.tracer as tracer
from triton_lib.tracer.tensor import op
import triton
import triton.language as tl
import triton_lib as tlib
import types
from functools import partial


def create():
    ttl = tracer.import_("triton.language", "tl")

    class numpy(Backend):
        name = "numpy"
        tensor_types = [tl.tensor, int, float, bool]
        _get_tests = staticmethod(_get_tests)

        @staticmethod
        @tlib.trace
        def to_tensor(tensor, shape):
            return tlib.tracer.apply(
                ttl.tensor,
                args=[tensor],
                output=tlib.tracer.Tensor(shape),
            )

        reshape = op.reshape(ttl.reshape)
        transpose = op.transpose(ttl.trans)
        broadcast_to = op.broadcast_to(ttl.broadcast_to)
        # einsum = op.einsum(tnp.einsum)
        arange = op.arange(ttl.arange)

        # stack = op.stack(ttl.stack)
        concatenate = op.concatenate(ttl.cat)

        add = associative_binary_to_nary(op.elementwise(ttl.add))
        subtract = op.elementwise(tnp.subtract)
        multiply = associative_binary_to_nary(op.elementwise(tnp.multiply))
        true_divide = op.elementwise(tnp.true_divide)
        floor_divide = op.elementwise(tnp.floor_divide)
        divide = op.elementwise(tnp.divide)
        logical_and = associative_binary_to_nary(op.elementwise(tnp.logical_and))
        logical_or = associative_binary_to_nary(op.elementwise(tnp.logical_or))
        where = op.elementwise(tnp.where)
        less = op.elementwise(tnp.less)
        less_equal = op.elementwise(tnp.less_equal)
        greater = op.elementwise(tnp.greater)
        greater_equal = op.elementwise(tnp.greater_equal)
        equal = op.elementwise(tnp.equal)
        not_equal = op.elementwise(tnp.not_equal)
        maximum = associative_binary_to_nary(op.elementwise(tnp.maximum))
        minimum = associative_binary_to_nary(op.elementwise(tnp.minimum))

        sum = op.reduce(tnp.sum)
        mean = op.reduce(tnp.mean)
        var = op.reduce(tnp.var)
        std = op.reduce(tnp.std)
        prod = op.reduce(tnp.prod)
        count_nonzero = op.reduce(tnp.count_nonzero)
        any = op.reduce(tnp.any)
        all = op.reduce(tnp.all)
        min = op.reduce(tnp.min)
        max = op.reduce(tnp.max)

        log = op.elementwise(tnp.log)
        exp = op.elementwise(tnp.exp)
        sqrt = op.elementwise(tnp.sqrt)
        square = op.elementwise(tnp.square)

        @staticmethod
        @tlib.trace
        def get_at(tensor, coordinates):
            return tensor[coordinates]

        @staticmethod
        @tlib.trace
        def set_at(tensor, coordinates, updates):
            return tensor.__setitem__(coordinates, updates)

        @staticmethod
        @tlib.trace
        def add_at(tensor, coordinates, updates):
            return tensor.__setitem__(
                coordinates, tensor.__getitem__(coordinates).__iadd__(updates)
            )

        @staticmethod
        @tlib.trace
        def subtract_at(tensor, coordinates, updates):
            return tensor.__setitem__(
                coordinates, tensor.__getitem__(coordinates).__isub__(updates)
            )

        flip = op.keep_shape(tnp.flip)
        roll = op.keep_shape(tnp.roll)

    return numpy()


def _get_tests():
    test = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32": tl.full(shape, value, dtype=dtype),
        # to_tensor=tl.asarray,
        # to_numpy=lambda x: x,
    )
    return [(create(), test)]