# -*- coding: UTF-8 -*-
import math

from ctypes import c_float, c_double, c_int64, c_int32, c_bool, c_void_p, byref, pointer
from .diopi_runtime import Sizes, Scalar, Tensor, TensorHandle
from .utils import check_returncode, check_function, squeeze
from . import Dtype, raw_like
from collections import namedtuple
import numpy as np


def broadcast_out_size(size1, size2):
    sizeO = size1 if len(size1) > len(size2) else size2
    length = len(size2) if len(size1) > len(size2) else len(size1)
    idx = -1
    while length > 0:
        assert size1[idx] == size2[idx] or size1[idx] == 1 or size2[idx] == 1,\
            "size1 and size2 must be broadcastable"
        sizeO[idx] = size1[idx] if size2[idx] == 1 else size2[idx]
        idx -= 1
        length -= 1

    return sizeO


def reduce_op_process(input, dim=None, keepdim=False, dtype=None):
    sizeI = list(input.size())
    if dim is None:
        for i in range(0, len(sizeI)):
            sizeI[i] = 1
        dim = []
    elif isinstance(dim, list):
        for i in dim:
            sizeI[i] = 1
    else:
        sizeI[dim] = 1
        dim = [dim]

    if dtype is None:
        dtype = input.get_dtype()

    out = Tensor(sizeI, dtype)
    if not keepdim:
        squeeze(out)
    return dim, out


def fill(tensor, value):
    func = check_function("diopiFill")
    ret = func(tensor.context_handle, tensor.tensor_handle, c_float(value))
    check_returncode(ret)
    return tensor


def ones_like(tensor):
    new_tensor = raw_like(tensor)
    fill(new_tensor, 1)
    return new_tensor


def zeros_like(tensor):
    new_tensor = raw_like(tensor)
    fill(new_tensor, 0)
    return new_tensor


def unary_op(input, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle)

    check_returncode(ret)
    return out


def binary_op(input, other, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle,
                   other.tensor_handle)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, other.tensor_handle)

    check_returncode(ret)
    return out


def binary_op_scalar(input, other, inplace, call, alpha=None, dtype=None) -> Tensor:
    args = "input.context_handle, "
    if dtype is None:
        dtype = input.get_dtype()

    if inplace:
        out = input
    else:
        sizeI = input.size()
        if not isinstance(other, Tensor):
            out = Tensor(sizeI, dtype)
        else:
            sizeO = other.size()
            outsize = broadcast_out_size(list(sizeI), list(sizeO))
            out = Tensor(outsize, dtype)
        args = args + "out.tensor_handle, "

    if not isinstance(other, Tensor):
        call = call + "Scalar"
        other = Scalar(input.get_dtype(), other)
        args = args + "input.tensor_handle, byref(other)"
    else:
        args = args + "input.tensor_handle, other.tensor_handle"\

    if alpha is not None:
        alpha = Scalar(input.get_dtype(), alpha)
        args = args + ", byref(alpha)"

    func = check_function(call)
    ret = eval(f'func({args})')

    check_returncode(ret)
    return out


def softmax(input, dim, dtype=None):
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    if dtype is None:
        dtype = input.get_dtype()
    out = raw_like(input)

    func = check_function('diopiSoftmax')
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim), c_int32(dtype.value))
    check_returncode(ret)
    return out


def relu(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiRelu')


def abs(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiAbs')


def floor(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiFloor')


def sign(input) -> Tensor:
    return unary_op(input, False, 'diopiSign')


def sigmoid(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSigmoid')


def sqrt(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSqrt')


def neg(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiNeg')


def sin(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSin')


def cos(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiCos')


def tanh(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiTanh')


def exp(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiExp')


def log(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog')


def log2(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog2')


def log10(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog10')


def erf(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiErf')


def add(input, other, alpha=1) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiAdd', alpha=alpha)


def sub(input, other, alpha=1.0) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiSub', alpha=alpha)


def eq(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiEq', dtype=Dtype.bool)


def ne(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiNe', dtype=Dtype.bool)


def ge(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiGe', dtype=Dtype.bool)


def gt(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiGt', dtype=Dtype.bool)


def le(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiLe', dtype=Dtype.bool)


def lt(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiLt', dtype=Dtype.bool)


def mul(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiMul')


def div(input, other, rounding_mode=None) -> Tensor:
    call = "diopiDiv"
    args = "input.context_handle, out.tensor_handle, input.tensor_handle, "
    sizeI = input.size()
    rounding_mode = convert_round_mode(rounding_mode)
    if not isinstance(other, Tensor):
        out = Tensor(sizeI, input.get_dtype())
        call = call + "Scalar"
        other = Scalar(input.get_dtype(), other)
        args = args + "byref(other)"
    else:
        sizeO = other.size()
        outsize = broadcast_out_size(list(sizeI), list(sizeO))
        out = Tensor(outsize, input.get_dtype())
        args = args + "other.tensor_handle"

    func = check_function(call)
    ret = eval(f'func({args}, rounding_mode)')

    check_returncode(ret)
    return out


def logical_and(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiBitwiseAnd', dtype=Dtype.bool)


def logical_or(input, other) -> Tensor:
    return binary_op_scalar(input, other, False, 'diopiBitwiseOr', dtype=Dtype.bool)


def leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor:
    negative_slope = byref(Scalar(Dtype.float64, negative_slope))
    if inplace:
        out = input
        func = check_function("diopiLeakyReluInp")
        ret = func(input.context_handle,
                   input.tensor_handle, negative_slope)
    else:
        out = raw_like(input)
        func = check_function("diopiLeakyRelu")
        ret = func(input.context_handle,
                   out.tensor_handle, input.tensor_handle, negative_slope)

    check_returncode(ret)
    return out


def bmm(input, mat2) -> Tensor:
    size1 = list(input.size())
    assert (len(size1) == 3), 'input must be 3d tensor'
    size2 = mat2.size()
    assert (len(size2) == 3), 'mat2 must be 3d tensor'
    assert (size1[0] == size2[0]), 'invalid args'
    assert (size1[2] == size2[1]), 'invalid args'

    size_out = size1
    size_out[2] = size2[2]
    out = Tensor(size_out, input.get_dtype())

    func = check_function("diopiBmm")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, mat2.tensor_handle)
    check_returncode(ret)
    return out


def addcmul(input, tensor1, tensor2, value=1) -> Tensor:
    size1 = list(tensor1.size())
    size2 = list(tensor2.size())
    sizeI = list(input.size())
    sizeO = broadcast_out_size(size1, size2)
    sizeO = broadcast_out_size(sizeI, sizeO)
    out = Tensor(sizeO, input.get_dtype())
    value = byref(Scalar(input.get_dtype(), value))

    func = check_function("diopiAddcmul")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               tensor1.tensor_handle, tensor2.tensor_handle, value)
    check_returncode(ret)
    return out


def matmul(input, other) -> Tensor:
    out = raw_like(input)
    sizeI = list(input.size())
    sizeO = list(other.size())

    # vector x vector
    if len(sizeI) == 1 and len(sizeO) == 1:
        out = Tensor((), input.get_dtype())
    # (batched) matrix x vector
    elif len(sizeO) == 1:
        sizeI[-1] = 1
        out = Tensor(sizeI,  input.get_dtype())
    # pretended matrix x (batched) matrix
    elif len(sizeI) == 1:
        sizeO[-2] = 1
        out = Tensor(sizeO, input.get_dtype())
    # (batched) matrix x (batched) matrix
    else:
        sizeI[-1] = sizeO[-1]
        if len(sizeI) > 3 and len(sizeO) > 2:
            assert sizeI[-3] == sizeO[-3] or sizeI[-3] == 1 or sizeO[-3] == 1,\
                'input and other should be broadcastable'
            sizeI[-3] = sizeI[-3] if sizeI[-3] == 1 else sizeO[-3]
        out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiMatmul")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_returncode(ret)
    return out


def clamp(input, min=None, max=None, inplace=False) -> Tensor:
    assert min is not None or max is not None,\
        "min and max can not be None in the meantime"
    if max is None:
        return clamp_min(input, min, inplace)
    if min is None:
        return clamp_max(input, max, inplace)

    call = "diopiClamp"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(min, Tensor):
        assert (isinstance(max, Tensor)), 'min and max must have same type'
        args += "input.tensor_handle, min.tensor_handle, max.tensor_handle"
    else:
        assert (~isinstance(max, Tensor)), 'min and max must have same type'
        call = call + 'Scalar'
        min = byref(Scalar(input.get_dtype(), min))
        max = byref(Scalar(input.get_dtype(), max))
        args = args + "input.tensor_handle, min, max"

    func = check_function(call)
    ret = eval(f'func({args})')
    check_returncode(ret)
    return out


def clamp_min(input, min, inplace=False) -> Tensor:
    call = "diopiClampMin"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(min, Tensor):
        args = args + "input.tensor_handle, min.tensor_handle"
    else:
        call = call + 'Scalar'
        min = byref(Scalar(input.get_dtype(), min))
        args = args + "input.tensor_handle, min"

    func = check_function(call)
    ret = eval(f'func({args})')
    check_returncode(ret)
    return out


def clamp_max(input, max, inplace=False) -> Tensor:
    call = "diopiClampMax"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(max, Tensor):
        args = args + "input.tensor_handle, max.tensor_handle"
    else:
        call = call + 'Scalar'
        max = byref(Scalar(input.get_dtype(), max))
        args = args + "input.tensor_handle, max"

    func = check_function(call)
    ret = eval(f'func({args})')
    check_returncode(ret)
    return out


def mean(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    assert isinstance(dim, (int, list)) or dim is None,\
        "dim should be int or list or None"

    dim, out = reduce_op_process(input, dim, keepdim, dtype)
    func = check_function("diopiMean")
    dim1 = Sizes(tuple(dim))
    if dtype is None:
        dtype = input.get_dtype()
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, c_int32(dtype.value))
    check_returncode(ret)
    return out


def std(input, unbiased, dim=None, keepdim=False) -> Tensor:
    assert isinstance(dim, (int, list)) or dim is None,\
        "dim should be int or list or None"

    dim, out = reduce_op_process(input, dim, keepdim)
    dim1 = Sizes(tuple(dim))
    func = check_function("diopiStd")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, unbiased)
    check_returncode(ret)
    return out


def min(input, dim=0, keepdim=False) -> Tensor:
    assert isinstance(dim, int), "dim should be int"

    sizeI = list(input.size())
    if keepdim:
        sizeI[dim] = 1
    else:
        del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())
    indices = Tensor(out.size(), Dtype.int64)
    func = check_function("diopiMin")

    ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
               input.tensor_handle, dim)
    check_returncode(ret)
    Res = namedtuple('Res', ['values', 'indices'])
    output = Res(out, indices)
    return output


def convert_reduction(name):
    if name == 'none':
        return 0
    if name == 'mean':
        return 1
    if name == "sum":
        return 2
    return 3


def convert_round_mode(name):
    if name is None:
        return 0
    if name == 'trunc':
        return 1
    if name == "floor":
        return 2
    return 4


def binary_cross_entropy_with_logits(input, target, weight=None,
                                     reduction='mean', pos_weight=None):
    assert input.size() == target.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'
    if pos_weight is not None:
        assert isinstance(pos_weight, Tensor), \
            'pos_weigth must be a Tensor'
        pos_weight = pos_weight.tensor_handle
    else:
        # represent pos_weight = None by pass a nullptr
        pos_weight = c_void_p()

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCEWithLogits")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, pos_weight, c_int64(reduction_mode))
    check_returncode(ret)
    return out


def cross_entropy(input, target, weight=None, ignore_index=- 100,
                  reduction='mean', label_smoothing=0.0):
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = Tensor(target.size(), input.get_dtype())
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCrossEntropyLoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, reduction_mode,
               ignore_index, c_double(label_smoothing))
    check_returncode(ret)
    return out


def mse_loss(input, target, reduction='mean'):
    assert input.shape() == target.shape(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiMSELoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, reduction_mode)
    check_returncode(ret)
    return out


def conv2d(input, weight, bias=None, stride=1,
           padding=0, dilation=1, groups=1) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        bias = bias.tensor_handle
    else:
        bias = c_void_p()

    sizeI = input.size()
    sizeW = list(weight.size())
    assert len(sizeI) == 4 and len(sizeW) == 4,\
        'input and weight must be 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    sizeO.append(sizeW[0])

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    for i in range(-2, 0):
        # equivalent kernel size
        sizeW[i] += (sizeW[i] - 1) * (dilation[i] - 1)
        sizeO.append(int((sizeI[i] - sizeW[i] + 2*padding[i])/stride[i]) + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    dilation = Sizes(tuple(dilation))

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiConvolution2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias, stride, padding, dilation, groups)
    check_returncode(ret)
    return out


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None) -> Tensor:
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    for i in range(-2, 0):
        if ceil_mode:
            sizeO.append(math.ceil((sizeI[i] - kernel_size[i] + 2*padding[i])/stride[i]) + 1)
        else:
            sizeO.append(math.floor((sizeI[i] - kernel_size[i] + 2*padding[i])/stride[i]) + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    out = Tensor(sizeO, input.get_dtype())

    if divisor_override is None:
        divisor_override = c_void_p()
    else:
        divisor_override = c_int64(divisor_override)
        divisor_override = byref(divisor_override)

    func = check_function("diopiAvgPool2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               kernel_size, stride, padding, ceil_mode, count_include_pad,
               divisor_override)
    check_returncode(ret)
    return out


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False) -> Tensor:
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    for i in range(-2, 0):
        tmp_ker_size = kernel_size[i] + (kernel_size[i] - 1) * (dilation[i] - 1)
        tmp_size = (sizeI[i] - tmp_ker_size + 2*padding[i])/stride[i] + 1
        tmp_size = tmp_size if tmp_size > 1 else 1
        if ceil_mode:
            sizeO.append(math.ceil(tmp_size))
        else:
            sizeO.append(math.floor(tmp_size))

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))
    out = Tensor(sizeO, input.get_dtype())

    if not return_indices:
        func = check_function("diopiMaxPool2d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, kernel_size,
                   stride, padding, dilation, ceil_mode)
        check_returncode(ret)
        return out
    else:
        func = check_function("diopiMaxPool2dWithIndices")
        indices = Tensor(sizeO, Dtype.int64)
        ret = func(input.context_handle, out.tensor_handle,
                   indices.tensor_handle, input.tensor_handle,
                   kernel_size, stride, padding, dilation, ceil_mode)
        check_returncode(ret)
        return out, indices


def adaptive_avg_pool2d(input, output_size):
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    for i in range(-2, 0):
        if output_size[i] is None:
            sizeO.append(sizeI[i])
        else:
            sizeO.append(output_size[i])

    out = Tensor(sizeO, input.get_dtype())
    output_size = Sizes((sizeO[-2], sizeO[-1]))

    func = check_function("diopiAdaptiveAvgPool2d")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, output_size)
    check_returncode(ret)
    return out


def adaptive_max_pool2d(input, output_size, return_indices=False):
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    for i in range(-2, 0):
        if output_size[i] is None:
            sizeO.append(sizeI[i])
        else:
            sizeO.append(output_size[i])

    out = Tensor(sizeO, input.get_dtype())
    output_size = Sizes(tuple(output_size))

    if return_indices:
        func = check_function("diopiAdaptiveMaxPool2dWithIndices")
        indices = Tensor(sizeO, Dtype.int64)
        ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
                   input.tensor_handle, output_size)
        check_returncode(ret)
        return out, indices
    else:
        func = check_function("diopiAdaptiveMaxPool2d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, output_size)
    check_returncode(ret)
    return out


def dropout(input, p=0.5, training=True, inplace=False):
    call = "diopiDropout"
    args = 'input.context_handle, '

    if inplace:
        out = input
        call = call + 'Inp'
    else:
        out = raw_like(input)
        args = args + 'out.tensor_handle, '

    args = args + "input.tensor_handle, c_double(p), training"
    func = check_function(call)
    ret = eval(f'func({args})')
    check_returncode(ret)
    return out


def test_dropout(input, p=0.5, training=True, inplace=False):
    input_numpy = input.numpy()
    out = dropout(input, p, training, inplace)
    out_numpy = out.numpy()

    # compute ratio
    real_ratio = np.sum(out_numpy == 0) / out.numel()

    # check data
    remains = out_numpy[out_numpy != 0]
    ref = input_numpy[out_numpy != 0]
    assert np.allclose(remains, ref / (1 - p), 1e-3),\
        "failed to execute dropout"

    return real_ratio


def index_select(input, dim, index) -> Tensor:
    sizeI = list(input.size())
    sizeI[dim] = index.numel()
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiIndexSelect")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, dim, index.tensor_handle)
    check_returncode(ret)
    return out


def select(input, dim, index) -> Tensor:
    sizeI = list(input.size())
    del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiSelect")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim), c_int64(index))
    check_returncode(ret)
    return out


def masked_scatter(input, mask, source) -> Tensor:
    assert mask.get_dtype() == Dtype.bool, \
        "mask must be bool tensor"
    out = raw_like(input)

    func = check_function("diopiMaskedScatter")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               mask.tensor_handle, source.tensor_handle)
    check_returncode(ret)
    return out


def nonzero(input):
    # note: pytorch(1.12) has argument 'as_tuple' to return multiple 1d tensor
    out_tensor_handle = TensorHandle()
    func = check_function("diopiNonzero")
    ret = func(input.context_handle, pointer(out_tensor_handle),
               input.tensor_handle)
    check_returncode(ret)
    out = Tensor.from_handle(out_tensor_handle)
    return out


def linear(input, weight, bias=None) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        bias = bias.tensor_handle
    else:
        bias = c_void_p()

    sizeI = list(input.size())
    sizeW = list(weight.size())
    sizeI[-1] = sizeW[-2] if len(sizeW) == 2 else 1
    out = Tensor(sizeI, input.get_dtype())
    func = check_function("diopiLinear")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias)
    check_returncode(ret)
    return out


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    sizeI = list(input.size())
    sizeW = weight.size()
    sizeI.append(sizeW[-1])
    out = Tensor(sizeI, weight.get_dtype())
    padding_idx = -100 if padding_idx is None else padding_idx

    if max_norm is not None:
        func2 = check_function("diopiEmbeddingRenorm_")
        ret2 = func2(input.context_handle, weight.tensor_handle, input.tensor_handle, c_double(max_norm), c_double(norm_type))
        check_returncode(ret2)

    # note: scale_grad_by_freq and sparse are useless during forward phase
    func = check_function("diopiEmbedding")
    ret = func(input.context_handle, out.tensor_handle, weight.tensor_handle,
               input.tensor_handle, c_int64(padding_idx), scale_grad_by_freq, sparse)
    check_returncode(ret)

    return out


def tril(input, diagonal=0) -> Tensor:
    out = raw_like(input)
    func = check_function("diopiTril")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, diagonal)
    check_returncode(ret)
    return out


def cat(tensors, dim=0) -> Tensor:
    assert isinstance(tensors, (list, tuple)),\
        "tensors must be a list or tuple"
    insNum = len(tensors)
    sum = 0
    c_tensors = []
    for tensor in tensors:
        sizeI = list(tensor.size())
        sum += sizeI[dim]
        c_tensors.append(tensor.tensor_handle)
    c_tensors = (c_void_p * insNum)(*c_tensors)

    sizeI[dim] = sum
    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_function("diopiCat")
    ret = func(tensors[0].context_handle, out.tensor_handle,
               pointer(c_tensors), insNum, dim)
    check_returncode(ret)
    return out


def stack(tensors, dim=0) -> Tensor:
    assert isinstance(tensors, (list, tuple)),\
        "tensors must be a list or tuple"
    insNum = len(tensors)
    outNum = insNum + 1
    sizeI = list(tensors[0].size())
    size_dim = dim
    if dim < 0:
        size_dim = outNum + dim
    sizeI.insert(size_dim, insNum)

    c_tensors = [t.tensor_handle for t in tensors]
    c_tensors = (c_void_p * insNum)(*c_tensors)

    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_function("diopiStack")
    ret = func(tensors[0].context_handle, out.tensor_handle,
               pointer(c_tensors), insNum, dim)
    check_returncode(ret)
    return out


def sort(input, dim=- 1, descending=False, stable=False):
    vals = raw_like(input)
    sizeI = input.size()
    indices = Tensor(sizeI, Dtype.int64)

    stable = c_void_p() if stable is None else pointer(c_bool(stable))

    func = check_function("diopiSort")
    ret = func(input.context_handle, vals.tensor_handle, indices.tensor_handle,
               input.tensor_handle, dim, descending, stable)
    check_returncode(ret)
    return vals, indices


def topk(input, k, dim=-1, largest=True, sorted=True):
    sizeI = list(input.size())
    sizeI[dim] = k
    values = Tensor(sizeI, input.get_dtype())
    indices = Tensor(sizeI, Dtype.int64)

    func = check_function("diopiTopk")
    ret = func(input.context_handle, values.tensor_handle,
               indices.tensor_handle, input.tensor_handle,
               k, dim, largest, sorted)
    check_returncode(ret)
    return values, indices


def transpose(input, dim0, dim1) -> Tensor:
    sizeI = list(input.size())
    sizeI[dim0], sizeI[dim1] = sizeI[dim1], sizeI[dim0]
    strideI = list(input.get_stride())
    strideI[dim0], strideI[dim1] = strideI[dim1], strideI[dim0]
    out = Tensor(sizeI, input.get_dtype(), strideI)

    func = check_function("diopiTranspose")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, dim0, dim1)
    check_returncode(ret)
    return out


def one_hot(input, num_classes=- 1):
    assert num_classes == -1 or num_classes > 0,\
        "num_classes must be -1 or >0"

    sizeI = input.size()
    if num_classes == -1:
        sizeI += (np.max(input.numpy()) + 1, )
        out = Tensor(sizeI, Dtype.int64)
    else:
        sizeI += (num_classes, )
        out = Tensor(sizeI, Dtype.int64)

    func = check_function("diopiOneHot")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, num_classes)
    check_returncode(ret)
    return out


def split(tensor, split_size_or_sections, dim=0):
    assert isinstance(split_size_or_sections, (int, list)),\
        "split_size_or_sections must be int or list"
    sizeI = list(tensor.size())
    sum = sizeI[dim]
    outs = []
    idx = 0
    splitSizes = ()
    is_int = isinstance(split_size_or_sections, int)

    while sum > 0:
        sizeI[dim] = split_size_or_sections if is_int else\
                     split_size_or_sections[idx]
        sizeI[dim] = sizeI[dim] if sum > sizeI[dim] else sum
        idx += 1
        sum -= sizeI[dim]
        splitSizes += (sizeI[dim], )
        out = Tensor(sizeI, tensor.get_dtype())
        outs.append(out)

    c_outs = []
    for i in range(idx):
        c_outs.append(outs[i].tensor_handle)

    c_outs = (c_void_p * idx)(*c_outs)
    splitSizes = Sizes(splitSizes)
    assert sum == 0,\
        "split_size_or_sections should be compatible with tensor shape"
    func = check_function("diopiSplitWithSizes")
    ret = func(tensor.context_handle, pointer(c_outs), c_int64(idx),
               tensor.tensor_handle, splitSizes, c_int64(dim))
    check_returncode(ret)
    return outs


def pow(input, exponent) -> Tensor:
    if not isinstance(input, Tensor):
        assert isinstance(exponent, Tensor),\
            "exponent must be tensor when input is scalar"
        func = check_function("diopiPowScalar")
        # todo: return type = input type or float
        out = raw_like(exponent)
        if isinstance(input, int):
            input = byref(Scalar(Dtype.int64, input))
        else:
            input = byref(Scalar(Dtype.float64, input))
        ret = func(exponent.context_handle, out.tensor_handle, input, exponent.tensor_handle)
    elif not isinstance(exponent, Tensor):
        assert isinstance(input, Tensor),\
            "input must be tensor when exponent is scalar"
        func = check_function("diopiPow")
        out = raw_like(input)
        if isinstance(exponent, int):
            exponent = byref(Scalar(Dtype.int64, exponent))
        else:
            exponent = byref(Scalar(Dtype.float64, exponent))
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, exponent)
    else:
        sizeI = list(input.size())
        sizeE = list(exponent.size())
        sizeO = broadcast_out_size(sizeI, sizeE)
        out = Tensor(sizeO, input.get_dtype())

        func = check_function("diopiPowTensor")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, exponent.tensor_handle)
    check_returncode(ret)
    return out


def where(condition, input, other) -> Tensor:
    # todo: add scalar version for pytorch 1.12
    assert (condition.get_dtype() in (Dtype.bool, Dtype.uint8)),\
        "condition must be a bool tensor"
    sizeX = list(input.size())
    sizeY = list(other.size())
    sizeC = list(condition.size())
    sizeO = broadcast_out_size(sizeX, sizeY)
    sizeO = broadcast_out_size(sizeC, sizeO)
    assert (input.get_dtype() == other.get_dtype()),\
        " input and other shoule be the same type "
    out = Tensor(sizeO, input.get_dtype())

    func = check_function("diopiWhere")
    ret = func(input.context_handle, out.tensor_handle, condition.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_returncode(ret)
    return out


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    assert (isinstance(max_norm, (int, float))),\
        "max_norm must be a int or float"
    assert (isinstance(norm_type, (int, float))),\
        "norm_type must be a int or float"

    if isinstance(parameters, Tensor):
        input = parameters
        parameters = [parameters.tensor_handle]
        parameters = (c_void_p * 1)(*parameters)
        parametersNum = 1
    else:
        input = parameters[0]
        parametersNum = len(parameters)
        parameters = [p.tensor_handle for p in parameters]
        parameters = (c_void_p * parametersNum)(*parameters)

    out = c_double(0.0)

    func = check_function("diopiClipGradNorm")
    func.argtypes = (c_void_p, type(pointer(out)), type(pointer(parameters)), c_int64, c_double, c_double, c_bool)
    ret = func(input.context_handle, pointer(out), pointer(parameters), parametersNum,
               max_norm, norm_type, error_if_nonfinite)
    check_returncode(ret)
    return out.value


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-05) -> Tensor:
    save_mean = mean(input, 1)
    tmp = sqrt(add(std(input, 1), eps))
    tmp_1 = Tensor((1,), input.get_dtype())
    fill(tmp_1, 1)
    save_invstd = div(tmp_1, tmp)

    if weight is None:
        weight = c_void_p()
    else:
        weight = weight.tensor_handle

    if bias is None:
        bias = c_void_p()
    else:
        bias = bias.tensor_handle

    func = check_function("diopiBatchNorm")
    if training:
        assert (running_mean is None and running_var is None),\
            "if trainging, running_mean and running_var are useless"
        running_mean = c_void_p()
        running_var = c_void_p()
    else:
        running_mean = running_mean.tensor_handle
        running_var = running_var.tensor_handle

    out = raw_like(input)
    ret = func(input.context_handle, out.tensor_handle, save_mean.tensor_handle, save_invstd.tensor_handle,
               input.tensor_handle, weight, bias, running_mean, running_var, training,
               c_double(momentum), c_double(eps))
    check_returncode(ret)
    return out


def log_softmax(input, dim, dtype=None):
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    if dtype is None:
        dtype = input.get_dtype()
    out = raw_like(input)

    func = check_function('diopiLogSoftmax')
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim), c_int32(dtype.value))
    check_returncode(ret)
    return out


def hardtanh(input, min_val=- 1.0, max_val=1.0, inplace=False) -> Tensor:
    call = "diopiHardtanh"
    min_val = byref(Scalar(input.get_dtype(), min_val))
    max_val = byref(Scalar(input.get_dtype(), max_val))
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, min_val, max_val)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, min_val, max_val)

    check_returncode(ret)
    return out


def threshold(input, threshold, value, inplace=False) -> Tensor:
    call = "diopiThreshold"
    threshold = byref(Scalar(input.get_dtype(), threshold))
    value = byref(Scalar(input.get_dtype(), value))
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, threshold, value)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, threshold, value)

    check_returncode(ret)
    return out


def gelu(input, approximate='none') -> Tensor:
    assert isinstance(approximate, str),\
        "approximate must be a string."
    out = raw_like(input)
    func = check_function("diopiGelu")

    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, approximate.encode('UTF-8'))

    check_returncode(ret)
    return out


def addcdiv(input, tensor1, tensor2, value=1) -> Tensor:
    size1 = list(tensor1.size())
    size2 = list(tensor2.size())
    sizeI = list(input.size())
    sizeO = broadcast_out_size(size1, size2)
    sizeO = broadcast_out_size(sizeI, sizeO)
    out = Tensor(sizeO, input.get_dtype())
    value = byref(Scalar(input.get_dtype(), value))

    func = check_function("diopiAddcdiv")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               tensor1.tensor_handle, tensor2.tensor_handle, value)
    check_returncode(ret)
    return out


def addmm(input, mat1, mat2, beta=1, alpha=1) -> Tensor:
    size1 = list(mat1.size())
    size2 = mat2.size()
    size1[-1] = size2[-1]
    sizeI = list(input.size())
    sizeO = broadcast_out_size(sizeI, size1)
    out = Tensor(sizeO, input.get_dtype())
    alpha = byref(Scalar(input.get_dtype(), alpha))
    beta = byref(Scalar(input.get_dtype(), beta))

    func = check_function("diopiAddmm")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               mat1.tensor_handle, mat2.tensor_handle, beta, alpha)
    check_returncode(ret)
    return out


def sum(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    assert isinstance(dim, (int, list)) or dim is None,\
        "dim should be int or list"
    func = check_function("diopiSum")
    dim, out = reduce_op_process(input, dim, keepdim, dtype)
    dim1 = Sizes(tuple(dim))
    if dtype is None:
        dtype = input.get_dtype()
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, c_int32(dtype.value))
    check_returncode(ret)
    return out


def max(input, dim, keepdim=False):
    assert isinstance(dim, int), "dim should be int"
    sizeI = list(input.size())
    if keepdim:
        sizeI[dim] = 1
    else:
        del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())
    indices = Tensor(out.size(), Dtype.int64)

    func = check_function("diopiMax")
    ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
               input.tensor_handle, dim)
    check_returncode(ret)
    Res = namedtuple('Res', ['values', 'indices'])
    output = Res(out, indices)
    return output


def any(input, dim, keepdim=False) -> Tensor:
    assert isinstance(dim, int), "dim should be int"
    _, out = reduce_op_process(input, dim, keepdim, dtype=Dtype.bool)
    func = check_function("diopiAny")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim)
    check_returncode(ret)
    return out


def all(input, dim, keepdim=False) -> Tensor:
    assert isinstance(dim, int), "dim should be int"
    _, out = reduce_op_process(input, dim, keepdim, dtype=Dtype.bool)
    func = check_function("diopiAll")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim)
    check_returncode(ret)
    return out


def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = Tensor(target.size(), input.get_dtype())
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCrossNLLLoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, reduction_mode, ignore_index)
    check_returncode(ret)
    return out


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction='none') -> Tensor:
    assert inputs.size() == targets.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if reduction == 'none':
        out = raw_like(inputs)
    else:
        out = Tensor((1,), inputs.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiSigmoidFocalLoss")
    ret = func(inputs.context_handle, out.tensor_handle, inputs.tensor_handle,
               targets.tensor_handle, c_float(alpha), c_float(gamma), reduction_mode)
    check_returncode(ret)
    return out


def nms(boxes, scores, iou_threshold) -> Tensor:
    size_boxes = boxes.size()
    assert len(size_boxes) == 2 and size_boxes[1] == 4,\
        "boxes must be a tensor of shape (N,4)"

    size_scores = scores.size()
    assert len(size_scores) == 1 and size_scores[0] == size_boxes[0],\
        "boxes must be a tensor of shape (N)"

    out_tensor_handle = TensorHandle()
    func = check_function("diopiNms")
    ret = func(boxes.context_handle, pointer(out_tensor_handle), boxes.tensor_handle,
               scores.tensor_handle, c_double(iou_threshold))
    out = Tensor.from_handle(out_tensor_handle)
    check_returncode(ret)
    return out


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False) -> Tensor:
    if isinstance(boxes, Tensor):
        size_boxes = boxes.size()
        assert len(size_boxes) == 2 and size_boxes[1] == 5,\
            "boxes should be a tensor of shape (N,5)"
    elif isinstance(boxes, list):
        size_boxes = boxes[0].size()
        assert len(size_boxes) == 2 and size_boxes[1] == 4,\
            "boxes should be a list of tensor of shape (N,4)"

    sizeI = list(input.size())
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    sizeI[-1] = output_size[-1]
    sizeI[-2] = output_size[-2]

    out = Tensor(sizeI, input.get_dtype())
    func = check_function("diopiRoiAlign")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               boxes.tensor_handle, c_double(spatial_scale), output_size[-2],
               output_size[-1], sampling_ratio, aligned)
    check_returncode(ret)
    return out


def slice_op(input, dim, index) -> Tensor:
    sizeI = list(input.size())
    num = int((index.stop - index.start + index.step - 1)/index.step)
    sizeI[dim] = num
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiSlice")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim, c_int64(index.start), c_int64(index.stop), c_int64(index.step))

    check_returncode(ret)
    return out


def index(input, **kwargs) -> Tensor:
    new_args = []
    hasEllipsis = False
    once = True
    for ele in kwargs.values():
        if ele is None:
            hasEllipsis = True
        else:
            if hasEllipsis and once:
                once = False
                sizeI = input.size()
                sizeE = ele.size()
                length = len(sizeI) - len(sizeE) - len(new_args)
                for i in range(length):
                    tmp = c_void_p()
                    new_args.append(tmp.value)

            new_args.append(ele.tensor_handle)

    nums = len(new_args)
    c_indices = (c_void_p * nums)(*new_args)

    out_tensor_handle = TensorHandle()
    func = check_function("diopiIndex")
    ret = func(input.context_handle, pointer(out_tensor_handle), input.tensor_handle,
               pointer(c_indices), c_int64(nums))
    out = Tensor.from_handle(out_tensor_handle)
    check_returncode(ret)
    return out


def sgd(param, param_grad, buf, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    # buf, param_grad are mutable
    func = check_function("diopiSgd")
    ret = func(param.context_handle, param.tensor_handle, param_grad.tensor_handle, buf.tensor_handle,
               c_double(lr), c_double(momentum), c_double(dampening), c_double(weight_decay), nesterov)
    check_returncode(ret)
    return param, buf


def adaptive_max_pool2d_backward(input, grad_outputs, indices, **kwargs) -> Tensor:
    grad_input = raw_like(input)
    assert len(grad_outputs) == 1,\
        "only input needs do backward"
    func = check_function("diopiAdaptiveMaxPool2dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, indices.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def slice_op_backward(input, grad_outputs, dim, index, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1,\
        "only input needs do backward"
    grad_input = raw_like(input)
    sizeI = input.size()
    sizeI = Sizes(sizeI)

    func = check_function("diopiSliceBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               sizeI, c_int64(dim), c_int64(index.start), c_int64(index.stop), c_int64(index.step))
    check_returncode(ret)
    return {"input": grad_input}


def index_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1,\
        "only input needs do backward"
    grad_input = raw_like(input)
    zeros_like_input = zeros_like(input)
    new_args = []
    hasEllipsis = False
    once = True
    for ele in kwargs.values():
        if ele is None:
            hasEllipsis = True
        else:
            if hasEllipsis and once:
                once = False
                sizeI = input.size()
                sizeE = ele.size()
                length = len(sizeI) - len(sizeE) - len(new_args)
                for i in range(length):
                    tmp = c_void_p()
                    new_args.append(tmp.value)

            new_args.append(ele.tensor_handle)

    nums = len(new_args)
    c_indices = (c_void_p * nums)(*new_args)

    func = check_function("diopiIndexBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, zeros_like_input.tensor_handle,
               pointer(c_indices), c_int64(nums), grad_outputs[0].tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def sigmoid_focal_loss_backward(inputs, grad_outputs, targets, alpha=0.25, gamma=2, reduction='none', **kwargs) -> Tensor:
    assert len(grad_outputs) == 1,\
        "only input needs do backward"
    assert inputs.size() == targets.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    grad_input = raw_like(inputs)
    grad_output = ones_like(inputs)
    if reduction == 'mean':
        fill(grad_output, 1 / inputs.numel())

    func = check_function("diopiSigmoidFocalLossBackward")
    # todo: check why need weight in functions.h
    ret = func(inputs.context_handle, grad_output.tensor_handle, inputs.tensor_handle,
               targets.tensor_handle, c_void_p(), grad_input.tensor_handle, c_float(gamma), c_float(alpha))
    check_returncode(ret)
    return {"inputs": grad_input}


def roi_align_backward(input, grad_outputs, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False) -> Tensor:
    assert len(grad_outputs) == 1,\
        "only accept 1 grad to do backward"
    if isinstance(boxes, Tensor):
        size_boxes = boxes.size()
        assert len(size_boxes) == 2 and size_boxes[1] == 5,\
            "boxes should be a tensor of shape (N,5)"
    elif isinstance(boxes, list):
        size_boxes = boxes[0].size()
        assert len(size_boxes) == 2 and size_boxes[1] == 4,\
            "boxes should be a list of tensor of shape (N,4)"

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    out = raw_like(input)
    sizeI = input.size()
    func = check_function("diopiRoiAlignBackward")
    ret = func(input.context_handle, out.tensor_handle, grad_outputs[0].tensor_handle,
               boxes.tensor_handle, c_double(spatial_scale), c_int64(output_size[-2]),
               c_int64(output_size[-1]), c_int64(sizeI[0]), c_int64(sizeI[1]), c_int64(sizeI[2]),
               c_int64(sizeI[3]), c_int64(sampling_ratio), aligned)
    check_returncode(ret)
    return {"input": out}


def conv2d_backward(input, grad_outputs, weight, bias=None, stride=1,
           padding=0, dilation=1, groups=1, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1,\
        "only accept 1 grad to do backward"
    sizeI = input.size()
    sizeW = weight.size()
    assert len(sizeI) == 4 and len(sizeW) == 4,\
        'input and weight must be 4d tensors'

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    dilation = Sizes(tuple(dilation))

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    out = {"input": grad_input, "weight": grad_weight}

    if bias is None:
        grad_bias = c_void_p()
        sizeBias = c_void_p()
    else:
        gradBias = raw_like(bias)
        grad_bias = gradBias.tensor_handle
        sizeBias = byref(Sizes(bias.size()))
        out.update({"bias": grad_bias})

    # todo: no transposed/output_padding in forward
    transposed = False
    output_padding = Sizes((0, 0))
    func = check_function("diopiConvolution2dBackward")

    ret = func(input.context_handle, grad_input.tensor_handle, grad_weight.tensor_handle, grad_bias,
               grad_outputs[0].tensor_handle, input.tensor_handle, weight.tensor_handle, sizeBias, stride,
               padding, dilation, c_bool(transposed), output_padding, c_int64(groups))
    check_returncode(ret)
    return out


def embedding_backward(input, grad_outputs, weight, padding_idx=None, scale_grad_by_freq=False, sparse=False, **kwargs):
    assert len(grad_outputs) == 1,\
        "only accept 1 grad to do backward"

    grad_weight = raw_like(weight)
    out = {"weight": grad_weight}
    num_weight = weight.size()[0]
    padding_idx = -100 if padding_idx is None else padding_idx
    # note: scale_grad_by_freq and sparse are useless during forward phase
    func = check_function("diopiEmbeddingBackward")
    ret = func(input.context_handle, grad_weight.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, c_int64(num_weight), c_int64(padding_idx), scale_grad_by_freq, sparse)
    check_returncode(ret)
    return out
