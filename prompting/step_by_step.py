"""
## Step-by-step Prompting for Long Specifications Generation
0. Let output spec code S = "class " 
1. Decompose the specification generation task into multiple steps.
2. In each step, use a step-specific prompt comprised of {few-shot examples, constraints, and instructions}.
3. Parse the prompt+output to S, continue to a new step of 2.
"""

from typing import List, Callable

from prompting.request import Client


class Step:
    def __init__(self, prompter, composer, eos=None, validator=None):
        self.prompter: Callable[[str], str] = prompter
        self.eos: List[str] = eos or []
        # (prompt, completion, old code) =composer=> new code
        self.composer: Callable[[str, str], str] = composer
        self.validator: Callable[[str], None] = validator  # validate the code


def step_by_step_gen(client: Client, steps: List[Step]):
    """For each step, we have: (i) step-wise prompt; and (ii) parser the output to code."""
    code = ""
    # TODO(@ganler): consider tree of steps
    for step in steps:
        prompt = step.prompter(code)
        completion = client.textgen(prompt=prompt, stop_sequences=step.eos)
        # print(prompt)
        # print("---------")
        # print(completion)
        new_code = step.composer(prompt, completion, code)
        if step.validator:  # TODO(@ganler): consider retry mechanism
            step.validator(code)
        code = new_code
    # print("\n\n---------\n")
    return code


if __name__ == "__main__":
    import yaml
    import os
    import argparse

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from prompting.request import TGIClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", help="Path to validate", required=True)
    args = parser.parse_args()

    # setups
    os.makedirs(args.output_dir, exist_ok=True)
    client = TGIClient(model="http://127.0.0.1:8080", max_new_tokens=2048)
    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    def generate_spec(api, doc, debug=False) -> str:
        steps = []

        # step 1: determine class names and input/output dtypes
        # - few-shot prompting
        steps.append(
            Step(
                prompter=lambda _: f"""\
# Reusable macros
DTYPE_GEN_ALL = ...
DTYPE_GEN_NON_BOOL = [dtype for dtype in DTYPE_GEN_ALL if dtype != DType.bool]
DTYPE_GEN_FLOATS = [DType.float16, DType.float32, DType.float64]
DTYPE_GEN_INTS = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]

# torch.flatten
class Flatten(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

# torch.matmul
class MatMul(BinaryOpBase):
    in_dtypes = [
        (i, i)
        for i in DTYPE_GEN_NON_BOOL
        if i not in [DType.complex64, DType.complex128]
    ]
    out_dtypes = [
        (i,) for i in DTYPE_GEN_NON_BOOL if i not in [DType.complex64, DType.complex128]
    ]

# torch.fft.fftfreq
class Pad(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]

# torch.nn.BatchNorm2d
class BatchNorm2d(ElementWiseUnaryOp):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

# {api}
class""",
                composer=lambda prmpt, cmpln, old: "class" + cmpln,
                eos=["\n\n", "\n#", "\ndef", "\nclass"],
                validator=None,
            )
        )

        # step 2: figure out the attributes and ranks
        # - few-shot prompting
        # - chain of thoughts
        steps.append(
            Step(
                prompter=lambda code: f"""\
# torch.matmul
class MatMul(BinaryOpBase):
    '''
    Signature: torch.matmul(input, other, *, out=None)
    Example:
    ```
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([3])
    ```
    Argument analysis:
    - `input`:
        - is a tensor: YES
        So not in __init__
    - `other`:
        - is a tensor: YES
        So not in __init__
    - other arguments are not quite interesting
    Input/Output Rank analysis (scalar is 0-rank):
    - has 2 inputs:
        - rank of #1: 1-3
        - rank of #2: 1-3
    - has 1 output:
        - rank of #1: min-3
    '''
    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks
        self.inp_ranks = [rank_range(1, 3), rank_range(1, 3)]
        self.out_ranks = [rank_until(3)]
# END

# nn.BatchNorm2d
class BatchNorm2d(ElementWiseUnaryOp):
    '''
    Signature: torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
    ```
    >>> m = nn.BatchNorm2d(100)
    >>> input = torch.randn(20, 100, 35, 45)
    >>> m(input).size()
    torch.Size([20, 100, 35, 45])
    ```
    Argument analysis:
    - `num_features`:
        - is a tensor: NO
        - is a symbolizable integer: YES
        - impacts output shapes or input constraints: YES
        So in __init__
    - `eps`:
        - is a tensor: NO
        - is a symbolizable integer: NO
        So not in __init__
    - `momentum` is similar to `eps`
    - other arguments are not quite interesting
    Input/Output Rank analysis (scalar is 0-rank):
    - has 1 input:
        - rank of #1: 4
    - has 1 output:
        - rank of #1: 4
    '''
    def __init__(self, nfeat):
        super().__init__()
        # attributes
        self.nfeat = nfeat
        # input/output ranks
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]
# END

# torch.nn.functional.relu
class ReLU(UnaryOpBase):
    '''
    Signature: torch.nn.functional.relu(input, inplace=False)
    ```
    >>> input = torch.randn(2)
    >>> torch.nn.functional.relu(input).size()
    torch.Size([2])
    ```
    Argument analysis:
    - `input`:
        - is a tensor: YES
        So not in __init__
    - `inplace`:
        - is a tensor: NO
        - is a symbolizable integer: NO
        So not in __init__
    Input/Output Rank analysis (scalar is 0-rank):
    - has 1 input:
        - rank of #1: all
    - has 1 output:
        - rank of #1: all
    '''
    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]

# END

# {api}
{code.splitlines()[0]}
    '''
    Signature: {api}(""",
                composer=lambda prmpt, cmpln, old: "\n".join(
                    l
                    for l in (old + cmpln.split("'''\n", 1)[-1]).splitlines()
                    if l.strip() and not l.strip().startswith("#")
                ),
                eos=["\n# END"],
                validator=None,
            )
        )

        # step 3: input constraints
        # - few-shot prompting
        steps.append(
            Step(
                prompter=lambda code: f"""\
# torch.flatten
class Flatten(UnaryOpBase):
    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks (scalar is 0-rank)
        self.inp_ranks = [rank_from(0)]
        self.out_ranks = [(1,)]

    def requires(self, itensors: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [] # No symbolic constraints, any inputs are fine
# END

# nn.BatchNorm2d
class BatchNorm2d(ElementWiseUnaryOp):
    def __init__(self, nfeat):
        super().__init__()
        # attributes
        self.nfeat = nfeat
        # input/output ranks
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

    def requires(self, itensors: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [
            nnsmith_eq(self.nfeat, itensors[0].shape[1]),
            nnsmith_ge(itensors[0].shape[0], 2),
        ]
# END

# torch.matmul
class MatMul(BinaryOpBase):
    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks
        self.inp_ranks = [rank_range(1, 3), rank_range(1, 3)]
        self.out_ranks = [rank_until(3)]

    def requires(self, itensors: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
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
# END

# {api}
{code}

    def requires(self, itensors: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:""",
                composer=lambda prmpt, cmpln, old: "\n".join(
                    l
                    for l in (
                        old
                        + "\n    def requires(self, itensors: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:"
                        + cmpln
                    ).splitlines()
                    if l.strip() and not l.strip().startswith("#")
                ),
                eos=["\n# END"],
                validator=None,
            )
        )

        # step 3: shape propagation
        # - few-shot prompting
        steps.append(
            Step(
                prompter=lambda code: f"""\
# torch.flatten
class Flatten(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks (scalar is 0-rank)
        self.inp_ranks = [rank_from(0)]
        self.out_ranks = [(1,)]

    def type_transfer(self, itensors: List[AbsTensor]) -> List[AbsTensor]:
        inp = itensors[0]
        return [
            AbsTensor(
                shape=[prod(inp.shape)],
                dtype=inp.dtype,
            )
        ]
    
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        # The rank of output tensor is 1 anyways such that the input ranks can be any of the input ranks.
        return [(random.randint(self.inp_ranks[0], self.inp_ranks[-1]), out_abs_tensor[0].dtype)]
# END

# torch.nn.BatchNorm2d
class BatchNorm2d(ElementWiseUnaryOp):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self, nfeat):
        super().__init__()
        # attributes
        self.nfeat = nfeat
        # input/output ranks
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

    def requires(self, itensors: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        return [
            nnsmith_eq(self.nfeat, itensors[0].shape[1]),
            nnsmith_ge(itensors[0].shape[0], 2),
        ]
    
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(4, DType.float32)]
# END

# torch.nn.functional.relu
class ReLU(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]

    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]

    def type_transfer(self, itensors: List[AbsTensor]) -> List[AbsTensor]:
        return [AbsTensor(itensors[0].shape, itensors[0].dtype)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]
# END

# {api}
{code}

    def type_transfer(self, itensors: List[AbsTensor]) -> List[AbsTensor]:""",
                composer=lambda prmpt, cmpln, old: "\n".join(
                    l
                    for l in (
                        old
                        + "\n    def type_transfer(self, itensors: List[AbsTensor]) -> List[AbsTensor]:"
                        + cmpln
                    ).splitlines()
                    if l.strip() and not l.strip().startswith("#")
                ),
                eos=["\n# END"],
                validator=None,
            )
        )

        # step 4: implementation
        # - few-shot prompting
        steps.append(
            Step(
                prompter=lambda code: f"""\
# torch.flatten
class Flatten(UnaryOpBase):
    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks (scalar is 0-rank)
        self.inp_ranks = [rank_from(0)]
        self.out_ranks = [(1,)]

def forward_fn(op: Flatten) -> Callable:
    return torch.Tensor.flatten
# END

# torch.nn.BatchNorm2d
class BatchNorm2d(ElementWiseUnaryOp):
    def __init__(self, nfeat):
        super().__init__()
        # attributes
        self.nfeat = nfeat
        # input/output ranks
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

def forward_fn(op: BatchNorm2d) -> Callable:
    return torch.nn.BatchNorm2d(num_features=op.nfeat)
# END

# torch.nn.functional.relu
class ReLU(UnaryOpBase):
    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks
        self.inp_ranks = [rank_all()]
        self.out_ranks = [rank_all()]

def forward_fn(op: ReLU) -> Callable:
    return torch.nn.functional.relu
# END

# torch.matmul
class MatMul(BinaryOpBase):
    def __init__(self):
        super().__init__()
        # attributes
        # None
        # input/output ranks
        self.inp_ranks = [rank_range(1, 3), rank_range(1, 3)]
        self.out_ranks = [rank_until(3)]

def forward_fn(op: MatMul) -> Callable:
    return torch.matmul
# END

# torch.nn.MaxPool2d
class MaxPool2d(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]
    out_dtypes = [(i,) for i in DTYPE_GEN_FLOATS]

    def __init__(
        self,
        kernel_h_size,
        kernel_w_size,
        stride,
        padding,
    ):
        super().__init__()
        # attributes
        self.kernel_h_size = kernel_h_size
        self.kernel_w_size = kernel_w_size
        self.stride = stride
        self.padding = padding
        # input/output ranks
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

def forward_fn(op: MaxPool2d) -> Callable:
    return torch.nn.MaxPool2d(
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
    )
# END

# torch.Tensor.tril
class Tril(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self, diagonal):
        super().__init__()
        # attributes
        self.diagonal = diagonal
        # input/output ranks
        self.inp_ranks = [(2,)]
        self.out_ranks = [(2,)]

def forward_fn(op: Tril) -> Callable:
    return lambda x: torch.tril(x, diagonal=op.diagonal)

# {api}
{code}

def forward_fn(op:""",
                composer=lambda prmpt, cmpln, old: "\n".join(
                    l
                    for l in (old + "\ndef forward_fn(op:" + cmpln).splitlines()
                    if l.strip() and not l.strip().startswith("#")
                ),
                eos=["\n# END"],
                validator=None,
            )
        )

        return step_by_step_gen(client, steps)

    prefix = os.path.join(os.path.dirname(__file__), "prompts")
    with progress_bar as p:
        for yaml_file in p.track(sorted(os.listdir(prefix))):
            full_path = os.path.join(prefix, yaml_file)
            with open(full_path, "r") as f:
                doc = yaml.load(f, Loader=yaml.FullLoader)

            code = generate_spec(doc["api"], doc["doc"])
            target_path = os.path.join(args.output_dir, f"{doc['api']}.txt")

            with open(target_path, "w") as f:
                f.write(code)
