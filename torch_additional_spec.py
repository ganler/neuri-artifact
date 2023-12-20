from typing import Callable, List, Optional, Tuple, Union, Type

import z3
import torch

from neuri.abstract.arith import *
from neuri.abstract.op import *
from neuri.abstract.tensor import *
@mark_materialize("torch")
class Det(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_range(2, 3)]
        self.out_ranks = [rank_range(0, 1)]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [AbsTensor(input_shapes[0].shape[:-2], input_shapes[0].dtype)]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        SanityCheck.eq(len(input_shapes), 1)
        return [
            nnsmith_eq(input_shapes[0].shape[-2], input_shapes[0].shape[-1]),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims + 2, out_abs_tensor[0].dtype)]

@mark_materialize("torch")
class InvOp(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_range(2, 2)]
        self.out_ranks = [rank_range(2, 2)]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return [
            nnsmith_eq(input_shapes[0].shape[0], input_shapes[0].shape[1]),
            nnsmith_gt(input_shapes[0].shape[0], 0),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(2, out_abs_tensor[0].dtype)]

@mark_materialize("torch")
class MatrixExp(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_range(2, 2)]
        self.out_ranks = [rank_range(2, 2)]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return [
            nnsmith_eq(input_shapes[0].shape[0], input_shapes[0].shape[1]),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(2, out_abs_tensor[0].dtype)]

@mark_materialize("torch")
class AdaptiveAvgPool1d(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.inp_ranks = [(3,)]
        self.out_ranks = [(3,)]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        inp_shape = input_shapes[0].shape
        SanityCheck.eq(len(inp_shape), 3)
        return [AbsTensor([inp_shape[0], inp_shape[1], self.output_size], input_shapes[0].dtype)]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        inp_shape = input_shapes[0].shape
        return [
            nnsmith_gt(inp_shape[2], 0),
            nnsmith_gt(self.output_size, 0),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(3, out_abs_tensor[0].dtype)]

@mark_materialize("torch")
class AdaptiveMaxPool2d(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self, output_size):
        super().__init__()
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]
        self.output_size = output_size
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        inp_shape = input_shapes[0].shape
        SanityCheck.eq(len(inp_shape), 4)
        return [AbsTensor([inp_shape[0], inp_shape[1], self.output_size, self.output_size], input_shapes[0].dtype)]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        inp_shape = input_shapes[0].shape
        return [
            nnsmith_gt(inp_shape[2], 0),
            nnsmith_gt(inp_shape[3], 0),
            nnsmith_gt(self.output_size, 0),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, out_abs_tensor[0].dtype)]

@mark_materialize("torch")
class Dropout(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self, p):
        super().__init__()
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
        self.p = p
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return [
            nnsmith_ge(self.p, 0),
            nnsmith_le(self.p, 1),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]

@mark_materialize("torch")
class ELU(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [
            nnsmith_gt(self.alpha, 0),
        ]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]

@mark_materialize("torch")
class FeatureAlphaDropout(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self, p):
        super().__init__()
        self.inp_ranks = [rank_from(2)]
        self.out_ranks = [rank_from(2)]
        self.p = p
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return [
            nnsmith_ge(self.p, 0),
            nnsmith_le(self.p, 1),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]

@mark_materialize("torch")
class Hardswish(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return []
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]

@mark_materialize("torch")
class HardTanh(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self, min_val=-1., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [
            nnsmith_le(self.min_val, self.max_val),
        ]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]

@mark_materialize("torch")
class SELU(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return []

@mark_materialize("torch")
class Sigmoid(ElementWiseUnaryOp):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return []

@mark_materialize("torch")
class SiLU(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return []

@mark_materialize("torch")
class Softshrink(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self, lambd):
        super().__init__()
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
        self.lambd = lambd
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [
            nnsmith_ge(self.lambd, 0),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]

@mark_materialize("torch")
class Softsign(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [input_shapes[0]]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]

@mark_materialize("torch")
class Threshold(ElementWiseUnaryOp):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold = threshold
        self.value = value
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [
            nnsmith_ge(self.value, self.threshold),
        ]

@mark_materialize("torch")
class CosineWindow(UnaryOpBase):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]
    def __init__(self, M):
        super().__init__()
        self.inp_ranks = [(0,)]
        self.out_ranks = [(1,)]
        self.M = M
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [AbsTensor(shape=[self.M], dtype=DType.float32)]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return [
            nnsmith_gt(self.M, 0),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(0, DType.float32)]

@mark_materialize("torch")
class Nuttall(UnaryOpBase):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.inp_ranks = [(0,)]
        self.out_ranks = [(1,)]
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 1)
        return [AbsTensor([self.M], DType.float32)]
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return [
            nnsmith_gt(self.M, 0),
        ]
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(0, DType.float32)]

