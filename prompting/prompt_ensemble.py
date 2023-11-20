import yaml
import os

BASIC_INTRO = R'''You are an expert in PyTorch programming. Now we want to extract and symbolize properties of some PyTorch functions.
Overall, the property describes (i) how the data types and shapes of input tensors are transformed (`type_transfer`); and (ii) what constraints
must be satisfied between the input tensors and the operator attributes in order to make the computation valid (`requires`).
The property is specified in a class that inherits from `AbsOpBase` below:

```python
class AbsOpBase(ABC):
    # number of symbolic attributes (e.g., kernel size, stride, etc.) that can impact the constraints
    num_var_param = None
    # All inputs must have the same dimension (e.g., Concat)?
    same_inp_dims = False
    # input dtypes: enumerates all possible input dtype combinations. Size of the list is the number of combinations.
    # Each element is a tuple of allowed input dtypes. NOTE: len(list) can >= the # of inputs, for handling ops with arbitrary arity.
    # For example, [(DType.float32, DType.float32), (DType.float64, DType.float64), (DType.int32, DType.int32)] means that
    # this op can accept one of float32xfloat32, float64xfloat64, and int32xint32 as input dtypes.
    in_dtypes: List[Tuple[DType, ...]]  = None  # Overwrite me!
    out_dtypes: List[Tuple[DType, ...]] = None  # Overwrite me!

    limit_domain = False # Does this operator have limited domain that can easily produce NaN/Inf (e.g., Log)

    def __init__(self):
        """To be overloaded for symbolizable (`Union[z3.IntNumRef, int]`) attributes (if any) which impacts the constraints"""
        self.inp_ranks = []
        self.out_ranks = []
        # The format is like [ Tuple[int, ...], Tuple[int, ...], ... ]
        # where the i-th tuple is the plausible ranks of the i-th input
        # e.g., for avgpool2d, inp_ranks = [(4,)] means that the inputs must be a single 4-dim tensor.
        # Utility functions like `rank_from(start)`, `rank_range(start, end)`, `rank_until(end)`, `rank_all()`
        # can help simplify the specification.
        # e.g., for Add, inp_ranks = [rank_all(), rank_all()] means that the inputs can be any rank.

    @abstractmethod  # Overload me
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        """Computes the output shapes and data types using `input_shapes` and attributes. Exception means rejection."""
        raise NotImplementedError

    # Overload if additional constraints are required
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        """Return constraints (booleans or Z3 predicates) between the input tensors and attributes"""
        return []

    @abstractmethod  # Overload me
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        """Given the output AbsTensor list, deduct the input rank and data type"""
        raise NotImplementedError
```
Meanwhile, this is the additional context for some of the class & method used above:
```python
# Data types
class DType(Enum):
    float16
    float32
    float64
    uint8
    uint16
    uint32
    uint64
    int8
    int16
    int32
    int64
    bool
    complex64
    complex128

# Reusable macros
DTYPE_INCOMMON = [DType.uint16, DType.uint32, DType.uint64]
DTYPE_GEN_ALL = [e for e in DType if e not in DTYPE_INCOMMON]
DTYPE_GEN_NON_BOOL = [dtype for dtype in DTYPE_GEN_ALL if dtype != DType.bool]
DTYPE_GEN_FLOATS = [DType.float16, DType.float32, DType.float64]
DTYPE_GEN_INTS = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]

class AbsTensor:
    def __init__(self, shape: List[Union[int, z3.ExprRef]], dtype: DType):
        self.shape = list(shape)
        self.dtype = DType(dtype)
```
Meanwhile, there are a few preset templates that can be reused by overloading the following classes:
```python
class UnaryOpBase(AbsOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]

    def __init__(self):
        super().__init__()
        self.out_ranks = [rank_all()]

class BinaryOpBase(AbsOpBase):
    in_dtypes = [(i, i) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]

    def __init__(self):
        super().__init__()
        self.out_ranks = [rank_all()]

class TernaryOpBase(AbsOpBase):
    in_dtypes = [(i, i, i) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]

    def __init__(self):
        super().__init__()
        self.out_ranks = [rank_all()]

class ElementWiseUnaryOp(UnaryOpBase):
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

class BcastBinaryOp(BinaryOpBase):
    # by default, output dtype is the same as the first input dtype
    _bcast_out_dtypes = None

    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_all(), rank_all()]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        tgt_shape = broadcast_shapes(*(ish.shape for ish in input_shapes))
        dtype = (
            input_shapes[0].dtype
            if self._bcast_out_dtypes is None
            else self._bcast_out_dtypes[0]
        )
        return [AbsTensor(tgt_shape, dtype)]

    def requires(self, input_shapes):
        return broadcast_cons_binary(*(ish.shape for ish in input_shapes))

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        x, y = bcast_rand_ndims(2, out_abs_tensor[0].ndims)
        return [
            (x, out_abs_tensor[0].dtype),
            (y, out_abs_tensor[0].dtype),
        ]

class BcastBinaryOp1(BcastBinaryOp):  # +-*/ max min
    in_dtypes = [(i, i) for i in DTYPE_GEN_NON_BOOL]
    out_dtypes = [(i,) for i in DTYPE_GEN_NON_BOOL]
    _bcast_out_dtypes = None

class Comparator(BcastBinaryOp):  # > < =
    in_dtypes = [(i, i) for i in DTYPE_GEN_ALL]
    out_dtypes = [(DType.bool,)]
    _bcast_out_dtypes = [DType.bool]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        x, y = bcast_rand_ndims(2, out_abs_tensor[0].ndims)
        in_dtypes = random.choice(self.in_dtypes)
        return [
            (x, in_dtypes[0]),
            (y, in_dtypes[1]),
        ]
```
'''

EXAMPLE_LINEAR = R'''# Example
```python
"""
class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)[source]¶
Applies a linear transformation to the incoming data: y=xAT+by = xA^T + by=xAT+b
This module supports TensorFloat32.
On certain ROCm devices, when using float16 inputs this module will use different precision for backward.

Parameters

in_features (int) – size of each input sample
out_features (int) – size of each output sample
bias (bool) – If set to False, the layer will not learn an additive bias.
Default: True


Shape:
Input: (∗,Hin)(*, H_{in})(∗,Hin​) where ∗*∗ means any number of
dimensions including none and Hin=in_featuresH_{in} = \text{in\_features}Hin​=in_features.
Output: (∗,Hout)(*, H_{out})(∗,Hout​) where all but the last dimension
are the same shape as the input and Hout=out_featuresH_{out} = \text{out\_features}Hout​=out_features.


Variables

weight (torch.Tensor) – the learnable weights of the module of shape
(out_features,in_features)(\text{out\_features}, \text{in\_features})(out_features,in_features). The values are
initialized from U(−k,k)\mathcal{U}(-\sqrt{k}, \sqrt{k})U(−k​,k​), where
k=1in_featuresk = \frac{1}{\text{in\_features}}k=in_features1​
bias – the learnable bias of the module of shape (out_features)(\text{out\_features})(out_features).
If bias is True, the values are initialized from
U(−k,k)\mathcal{U}(-\sqrt{k}, \sqrt{k})U(−k​,k​) where
k=1in_featuresk = \frac{1}{\text{in\_features}}k=in_features1​



Examples:
>>> m = nn.Linear(20, 30)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 30])
"""
@mark_materialize("torch")
class Linear(UnaryOpBase):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self, ifeat: Union[int, z3.ExprRef], ofeat: Union[int, z3.ExprRef]):
        super().__init__()
        self.ifeat = ifeat
        self.ofeat = ofeat
        self.inp_ranks = [rank_from(1)]
        self.out_ranks = [rank_from(1)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        return [
            AbsTensor(
                shape=[*input_shapes[0].shape[:-1], self.ofeat], dtype=DType.float32
            )
        ]

    def requires(self, input_shapes: List[AbsTensor]) -> List[z3.ExprRef]:
        ConstraintCheck.true(input_shapes[0].ndims >= 1)
        return [
            nnsmith_ge(self.ifeat, 1),
            nnsmith_ge(self.ofeat, 1),
            nnsmith_eq(input_shapes[0].shape[-1], self.ifeat),
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, DType.float32)]

# !Translate the spec marked by `mark_materialize` into a callable in PyTorch
def forward_fn(op: Linear) -> Callable:
    return torch.nn.Linear(in_features=op.ifeat, out_features=op.ofeat)
```
'''

EXAMPLE_FLATTEN = R'''# Example
```python
"""
torch.flatten(input, start_dim=0, end_dim=-1) → Tensor¶
Flattens input by reshaping it into a one-dimensional tensor. If start_dim or end_dim
are passed, only dimensions starting with start_dim and ending with end_dim are flattened.
The order of elements in input is unchanged.
Unlike NumPy’s flatten, which always copies input’s data, this function may return the original object, a view,
or copy. If no dimensions are flattened, then the original object input is returned. Otherwise, if input can
be viewed as the flattened shape, then that view is returned. Finally, only if the input cannot be viewed as the
flattened shape is input’s data copied. See torch.Tensor.view() for details on when a view will be returned.

Note
Flattening a zero-dimensional tensor will return a one-dimensional view.


Parameters

input (Tensor) – the input tensor.
start_dim (int) – the first dim to flatten
end_dim (int) – the last dim to flatten

Example:
>>> t = torch.tensor([[[1, 2],
...                    [3, 4]],
...                   [[5, 6],
...                    [7, 8]]])
>>> torch.flatten(t)
tensor([1, 2, 3, 4, 5, 6, 7, 8])
>>> torch.flatten(t, start_dim=1)
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]])
"""

@mark_materialize("torch")
class Flatten(UnaryOpBase):
    # It is compatible to all data types
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_from(0)] # Input tensor can be any non-scalar tensor
        self.out_ranks = [(1,)] # Flattened tensor has only one dimension

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        inp = input_shapes[0]
        return [
            AbsTensor(
                shape=[prod(inp.shape)],
                dtype=inp.dtype,
            )
        ]

    # No need to implement/overload the "requires" function since there is no constraint between the input and the attributes

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        # The rank of output tensor is 1 anyways such that the input ranks can be any of the input ranks.
        return [(random.randint(self.inp_ranks[0], self.inp_ranks[-1]), out_abs_tensor[0].dtype)]

# !Translate the spec marked by `mark_materialize` into a callable in PyTorch
def forward_fn(op: Flatten) -> Callable:
    return torch.Tensor.flatten
```
'''


EXAMPLE_MATMUL = R'''# Example
```python
"""
torch.matmul(input, other, *, out=None) → Tensor¶
Matrix product of two tensors.
The behavior depends on the dimensionality of the tensors as follows:

If both tensors are 1-dimensional, the dot product (scalar) is returned.
If both arguments are 2-dimensional, the matrix-matrix product is returned.
If the first argument is 1-dimensional and the second argument is 2-dimensional,
a 1 is prepended to its dimension for the purpose of the matrix multiply.
After the matrix multiply, the prepended dimension is removed.
If the first argument is 2-dimensional and the second argument is 1-dimensional,
the matrix-vector product is returned.
If both arguments are at least 1-dimensional and at least one argument is
N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
The non-matrix (i.e. batch) dimensions are broadcasted (and thus
must be broadcastable).  For example, if input is a
(j×1×n×n)(j \times 1 \times n \times n)(j×1×n×n) tensor and other is a (k×n×n)(k \times n \times n)(k×n×n)
tensor, out will be a (j×k×n×n)(j \times k \times n \times n)(j×k×n×n) tensor.
Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
are broadcastable, and not the matrix dimensions. For example, if input is a
(j×1×n×m)(j \times 1 \times n \times m)(j×1×n×m) tensor and other is a (k×m×p)(k \times m \times p)(k×m×p)
tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
matrix dimensions) are different. out will be a (j×k×n×p)(j \times k \times n \times p)(j×k×n×p) tensor.


This operation has support for arguments with sparse layouts. In particular the
matrix-matrix (both arguments 2-dimensional) supports sparse arguments with the same restrictions
as torch.mm()

Note
The 1-dimensional dot product version of this function does not support an out parameter.


Parameters

input (Tensor) – the first tensor to be multiplied
other (Tensor) – the second tensor to be multiplied


Keyword Arguments
out (Tensor, optional) – the output tensor.


Example:
>>> # vector x vector
>>> tensor1 = torch.randn(3)
>>> tensor2 = torch.randn(3)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([])
>>> # matrix x vector
>>> tensor1 = torch.randn(3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([3])
>>> # batched matrix x broadcasted vector
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3])
>>> # batched matrix x batched matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(10, 4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
>>> # batched matrix x broadcasted matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
"""

@mark_materialize("torch")
class MatMul(BinaryOpBase):
    in_dtypes = [
        (i, i)
        for i in DTYPE_GEN_NON_BOOL
        if i not in [DType.complex64, DType.complex128]
    ]
    out_dtypes = [
        (i,) for i in DTYPE_GEN_NON_BOOL if i not in [DType.complex64, DType.complex128]
    ]

    def __init__(self):
        super().__init__()
        # Consider at most 3D tensors (batched mm)
        self.inp_ranks = [rank_range(1, 3), rank_range(1, 3)]
        self.out_ranks = [rank_until(3)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        # shape: [*batches(?), *rc (row and col)]
        lhs = input_shapes[0].shape
        rhs = input_shapes[1].shape

        lrc = lhs[-2:]
        rrc = rhs[-2:]
        orc = [*lrc[:-1], *rrc[1:]]

        lbatch = lhs[: -len(lrc)]
        rbatch = rhs[: -len(rrc)]
        batches = []
        if len(lbatch) > len(rbatch):
            batches = lbatch[: len(lbatch) - len(rbatch)]
            for x, y in zip(lbatch[len(batches) :], rbatch):
                batches.append(nnsmith_max(x, y))
        else:
            batches = rbatch[: len(rbatch) - len(lbatch)]
            for x, y in zip(lbatch, rbatch[len(batches) :]):
                batches.append(nnsmith_max(x, y))

        return [AbsTensor([*batches, *orc], input_shapes[0].dtype)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        cons = []

        lhs = input_shapes[0].shape
        rhs = input_shapes[1].shape

        lrc = lhs[-2:]
        rrc = rhs[-2:]
        cons.append(lrc[-1] == rrc[0])

        # CHECK: batch dim broadcastable
        lbatch = lhs[: -len(lrc)]
        rbatch = rhs[: -len(rrc)]
        common_tail = min(len(lbatch), len(rbatch))
        for x, y in zip(lbatch[-common_tail:], rbatch[-common_tail:]):
            cons.append(nnsmith_or(x == y, nnsmith_or(x == 1, y == 1)))
        return cons

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        # rank(a) = batch_rank(a) + rc_rank(a)
        # rank(b) = batch_rank(b) + rc_rank(b)
        # out_rank = max(br_a, br_b) + (rcr_a + rcr_b) - 2
        # 1 <= rcr_a or rcr_b <= min(2, out_rank + 2)
        # br_a = ranks[0], rcr_a = ranks[1]
        # br_b = ranks[2], rcr_b = ranks[3]
        ranks = [0, 1, 0, 1]

        def check_sat():
            return (
                out_abs_tensor[0].ndims
                == max(ranks[0], ranks[2]) + (ranks[1] + ranks[3]) - 2
            )

        while not check_sat():
            inc_candidates = []
            inc_candidates.append(1 if ranks[1] < 2 else 0)
            inc_candidates.append(3 if ranks[3] < 2 else 2)
            choice = random.choice(inc_candidates)
            ranks[choice] += 1

        return [
            (ranks[0] + ranks[1], out_abs_tensor[0].dtype),
            (ranks[2] + ranks[3], out_abs_tensor[0].dtype),
        ]

# !Translate the spec marked by `mark_materialize` into a callable in PyTorch
def forward_fn(op: MatMul) -> Callable:
    return torch.matmul
```
'''

EXAMPLE_PAD = R'''# Example
```python
"""
torch.nn.functional.pad(input, pad, mode='constant', value=None) → Tensor¶
Pads tensor.

Padding size:The padding size by which to pad some dimensions of input
are described starting from the last dimension and moving forward.
⌊len(pad)2⌋\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor⌊2len(pad)​⌋ dimensions
of input will be padded.
For example, to pad only the last dimension of the input tensor, then
pad has the form
(padding_left,padding_right)(\text{padding\_left}, \text{padding\_right})(padding_left,padding_right);
to pad the last 2 dimensions of the input tensor, then use
(padding_left,padding_right,(\text{padding\_left}, \text{padding\_right},(padding_left,padding_right,
padding_top,padding_bottom)\text{padding\_top}, \text{padding\_bottom})padding_top,padding_bottom);
to pad the last 3 dimensions, use
(padding_left,padding_right,(\text{padding\_left}, \text{padding\_right},(padding_left,padding_right,
padding_top,padding_bottom\text{padding\_top}, \text{padding\_bottom}padding_top,padding_bottom
padding_front,padding_back)\text{padding\_front}, \text{padding\_back})padding_front,padding_back).

Padding mode:See torch.nn.CircularPad2d, torch.nn.ConstantPad2d,
torch.nn.ReflectionPad2d, and torch.nn.ReplicationPad2d
for concrete examples on how each of the padding modes works. Constant
padding is implemented for arbitrary dimensions. Circular, replicate and
reflection padding are implemented for padding the last 3 dimensions of a
4D or 5D input tensor, the last 2 dimensions of a 3D or 4D input tensor,
or the last dimension of a 2D or 3D input tensor.

Parameters

input (Tensor) – N-dimensional tensor
pad (tuple) – m-elements tuple, where
m2≤\frac{m}{2} \leq2m​≤ input dimensions and mmm is even.
mode – 'constant', 'reflect', 'replicate' or 'circular'.
Default: 'constant'
value – fill value for 'constant' padding. Default: 0



Examples:
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p1d = (1, 1) # pad last dim by 1 on each side
>>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
>>> print(out.size())
torch.Size([3, 3, 4, 4])
>>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
>>> out = F.pad(t4d, p2d, "constant", 0)
>>> print(out.size())
torch.Size([3, 3, 8, 4])
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
>>> out = F.pad(t4d, p3d, "constant", 0)
>>> print(out.size())
torch.Size([3, 9, 7, 3])
"""
class Pad(UnaryOpBase):
    num_var_param = _pad_num_var_param()
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]

    def __init__(self, padding_list, pad_t):
        super().__init__()
        self.padding_list = padding_list
        self.inp_ranks = [rank_from(len(padding_list) // 2)]
        self.out_ranks = [rank_from(len(padding_list) // 2)]
        assert (
            len(self.padding_list) % 2 == 0
        ), f"padding_list must be even, got {self.padding_list}"

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        pad = self.padding_list
        isv = input_shapes[0].shape
        cons = []
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            # When using negative padding, neither side should erase more than the original size
            cons.append(nnsmith_gt(nnsmith_add(pad[i * 2], isv[j]), 0))
            cons.append(nnsmith_gt(nnsmith_add(pad[i * 2 + 1], isv[j]), 0))
            cons.append(
                nnsmith_gt(
                    nnsmith_add(pad[i * 2 + 1], nnsmith_add(pad[i * 2], isv[j])), 0
                )
            )
        for s in input_shapes[0].shape[1:]:
            cons.append(nnsmith_gt(s, 0))
        return cons

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        isv = input_shapes[0].shape
        pad = self.padding_list
        s = list(isv)
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            s[j] = nnsmith_add(nnsmith_add(s[j], pad[i * 2]), pad[i * 2 + 1])
        return [AbsTensor(s, input_shapes[0].dtype)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]

@mark_materialize("torch")
class ConstPad(Pad):
    def __init__(self, *padding_list):
        super().__init__(padding_list, "constant")

@mark_materialize("torch")
class ReplicatePad(Pad):
    num_var_param = _pad_num_var_param(2, max=6)

    def __init__(self, *padding_list):
        super().__init__(padding_list, "replicate")
        self.inp_ranks = [rank_range(len(padding_list) // 2 + 1, 4)]
        self.out_ranks = [rank_range(len(padding_list) // 2 + 1, 4)]

@mark_materialize("torch")
class ReflectPad(Pad):
    num_var_param = _pad_num_var_param(2, max=6)

    def __init__(self, *padding_list):
        super().__init__(padding_list, "reflect")
        self.inp_ranks = [rank_range(len(padding_list) // 2 + 1, 4)]
        self.out_ranks = [rank_range(len(padding_list) // 2 + 1, 4)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        cons = super().requires(input_shapes)
        pad = self.padding_list
        isv = input_shapes[0].shape
        for i in range(len(pad) // 2):
            j = len(isv) - 1 - i
            # per torch's complaint: Padding size should be less than the corresponding input dimension
            cons.append(nnsmith_lt(pad[i * 2], isv[j]))
            cons.append(nnsmith_lt(pad[i * 2 + 1], isv[j]))
            # same sign to avoid ORT bugs
            cons.append(nnsmith_ge(pad[i * 2] * pad[i * 2 + 1], 0))
        return cons

# !Translate the spec marked by `mark_materialize` into a callable in PyTorch
def forward_fn(op: Pad):
    if op.extra_attrs["type"] == "constant":
        # 0 easily cause division by zero...
        # 1 easily cause false positives (sqrt(1) = 0.99999... != 1 in ORT, so floor(sqrt(1))=0)
        return lambda x: torch.nn.functional.pad(
            x, op.padding_list, "constant", value=0.5
        )
    elif op.extra_attrs["type"] == "replicate" or op.extra_attrs["type"] == "reflect":
        return lambda x: torch.nn.functional.pad(
            x, op.padding_list, op.extra_attrs["type"]
        )
```
'''

EXAMPLE_BATCHNORM = R'''# Example
```python
"""
class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)[source]¶
Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension).

Because the Batch Normalization is done over the C dimension, computing statistics
on (N, H, W) slices, it’s common terminology to call this Spatial Batch Normalization.

Parameters

num_features (int) – CCC from an expected input of size
(N,C,H,W)(N, C, H, W)(N,C,H,W)
...

Shape:
Input: (N,C,H,W)(N, C, H, W)(N,C,H,W)
Output: (N,C,H,W)(N, C, H, W)(N,C,H,W) (same shape as input)

Examples:
>>> # With Learnable Parameters
>>> m = nn.BatchNorm2d(100)
>>> # Without Learnable Parameters
>>> m = nn.BatchNorm2d(100, affine=False)
>>> input = torch.randn(20, 100, 35, 45)
>>> output = m(input)
"""

@mark_materialize("torch")
class BatchNorm2d(ElementWiseUnaryOp):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self, nfeat):
        super().__init__()
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]
        self.nfeat = nfeat

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, DType.float32)]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [
            nnsmith_eq(self.nfeat, input_shapes[0].shape[1]),
            nnsmith_ge(input_shapes[0].shape[0], 2),
        ]

# !Translate the spec marked by `mark_materialize` into a callable in PyTorch
def forward_fn(op: BatchNorm2d) -> Callable:
    return torch.nn.BatchNorm2d(num_features=op.nfeat)
```
'''


def make_prompt(doc):
    doc = doc.strip()
    return (
        BASIC_INTRO
        + EXAMPLE_FLATTEN
        + EXAMPLE_BATCHNORM
        + f'''
Now, given the documentation:

```
"""
{doc}
"""
```

Follow the format of examples (in "# Example") shown above to implement a specification class as well as a "forward_fn" function that transforms it into a real PyTorch callable:
'''
    )


def make_prompt_v2(doc):
    doc = doc.strip()
    return (
        BASIC_INTRO
        + EXAMPLE_BATCHNORM
        + EXAMPLE_PAD
        + f'''
Now, your task is to follow the format of examples (in "# Example") shown above:
(1). Implement a specification class by overloading `AbsOpBase` or one of its subclasses (e.g., `UnaryOpBase`).
(2). Implement a `forward_fn` function that transforms the specification class into a real PyTorch callable.

NOTES:
1. **DO NOT** use on any PyTorch APIs in `type_transfer` and `requires`. Instead, reason about the constraints and output shapes symbolically using the `nnsmith_*` APIs
2.1 The `__init__` function for spec marked by `mark_materialize` can only take arguments whose type is `Union[z3.IntNumRef, int]` which are the integer attributes that can impact  `type_transfer` or `requires`
2.2 If an attribute is not a symbolizable integer (boolean is not integer) or it is not used by `type_transfer` or `requires`, don't model it in `__init__`
3. For non-symbolizable attributes that are required in `forward_fn` for constructing the PyTorch callable, one can use some default values or literals to hardcode it
4. `forward_fn` should take an instance of the specification class as input and return a callable that takes the same number of arguments (>= 1) as the number of inputs of the operator (in `type_transfer`)
5. You may assume the importing statements to be as follows, but don't repeat/include such statements in your response code:

```python
import random
from typing import List, Tuple, Union, Callable

import z3

from nnsmith.abstract.op import *
from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import *
from nnsmith.abstract.tensor import AbsTensor
```

Your turn! Given the documentation:

"""
{doc}
"""
'''
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-33b-instruct", trust_remote_code=True
    )

    prefix = os.path.join(os.path.dirname(__file__), "prompts")
    for yaml_file in sorted(os.listdir(prefix)):
        full_path = os.path.join(prefix, yaml_file)
        with open(full_path, "r") as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)
        prompt = make_prompt_v2(doc["doc"])
        doc["prompt"] = prompt
        with open(full_path, "w") as f:
            yaml.dump(doc, f)
