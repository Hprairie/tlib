from .base import Backend, associative_binary_to_nary
import tlib.tracer as tracer
from tlib.tracer.tensor import op
import triton
import triton.language as tl
import tlib
import types
from functools import partial


def create():
    ttl = tracer.import_("tlib.functional", "tlf")

    class triton(Backend):
        name = "triton"
        tensor_types = [tl.tensor, int, float, bool]
        _get_tests = staticmethod(_get_tests)

        @staticmethod
        @tlib.trace
        def to_tensor(tensor, shape):
            return tlib.tracer.apply(
                ttl.to_tensor,
                args=[tensor],
                output=tlib.tracer.Tensor(shape),
            )

        reshape = op.reshape(ttl.reshape)
        transpose = op.transpose(ttl.transpose)
        broadcast_to = op.broadcast_to(ttl.broadcast_to)
        arange = op.arange(ttl.arange)
        expand_dims = op.expand_dims(ttl.expand_dims)

        stack = op.stack(ttl.stack)
        concatenate = op.concatenate(ttl.concatenate)

        add = associative_binary_to_nary(op.elementwise(ttl.add))
        subtract = op.elementwise(ttl.subtract)
        multiply = associative_binary_to_nary(op.elementwise(ttl.multiply))
        true_divide = op.elementwise(ttl.true_divide)
        floor_divide = op.elementwise(ttl.floor_divide)
        divide = op.elementwise(ttl.divide)
        logical_and = associative_binary_to_nary(op.elementwise(ttl.logical_and))
        logical_or = associative_binary_to_nary(op.elementwise(ttl.logical_or))
        where = op.elementwise(ttl.where)
        less = op.elementwise(ttl.less)
        less_equal = op.elementwise(ttl.less_equal)
        greater = op.elementwise(ttl.greater)
        greater_equal = op.elementwise(ttl.greater_equal)
        equal = op.elementwise(ttl.equal)
        not_equal = op.elementwise(ttl.not_equal)
        maximum = associative_binary_to_nary(op.elementwise(ttl.maximum))
        minimum = associative_binary_to_nary(op.elementwise(ttl.minimum))
        kl_div = op.elementwise(ttl.kl_div)
        mse = op.elementwise(ttl.mse)
        cross_entropy = op.elementwise(ttl.cross_entropy)

        sum = op.reduce(ttl.sum)
        mean = op.reduce(ttl.mean)
        var = op.reduce(ttl.var)
        std = op.reduce(ttl.std)
        prod = op.reduce(ttl.prod)
        count_nonzero = op.reduce(ttl.count_nonzero)
        any = op.reduce(ttl.any)
        all = op.reduce(ttl.all)
        min = op.reduce(ttl.min)
        max = op.reduce(ttl.max)
        argmin = op.reduce(ttl.argmin)
        argmax = op.reduce(ttl.argmax)
        logsumexp = op.reduce(ttl.logsumexp)

        log = op.elementwise(ttl.log)
        exp = op.elementwise(ttl.exp)
        sqrt = op.elementwise(ttl.sqrt)
        square = op.elementwise(ttl.square)

        # Unary ops
        cumsum = op.keep_shape(ttl.cumsum)
        cumprod = op.keep_shape(ttl.cumprod)
        flip = op.keep_shape(ttl.flip)
        softmax = op.keep_shape(ttl.softmax)
        sort = op.keep_shape(ttl.sort)
        associative_scan = op.keep_shape(ttl.associative_scan)

        # Binary ops

        @staticmethod
        @tlib.trace
        def get_at(tensor, coordinates):
            return tensor[coordinates]

        @staticmethod
        @tlib.trace
        def set_at(tensor, coordinates, updates):
            return tensor.__setitem__(coordinates, updates)

        # @staticmethod
        # @tlib.trace
        # def add_at(tensor, coordinates, updates):
        #     return tensor.__setitem__(coordinates, tensor.__getitem__(coordinates).__iadd__(updates))

        # @staticmethod
        # @tlib.trace
        # def subtract_at(tensor, coordinates, updates):
        #     return tensor.__setitem__(coordinates, tensor.__getitem__(coordinates).__isub__(updates))

        # roll = op.keep_shape(ttl.roll)

    return triton()


def _get_tests():
    test = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32": tl.full(shape, value, dtype=dtype),
        # to_tensor=tl.asarray,
        # to_numpy=lambda x: x,
    )
    return [(create(), test)]
