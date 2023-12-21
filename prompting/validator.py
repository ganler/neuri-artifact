import os
from dataclasses import dataclass
import ast
from typing import Callable, List, Optional, Tuple, Union, Type
import traceback

from termcolor import colored

from neuri.abstract.arith import *
from neuri.abstract.op import *
from neuri.materialize.torch.symbolnet import random_tensor
import z3

import torch


@dataclass
class SpecCode:
    cls_name: str
    spec: str
    fwd_code: str
    dbg_info: Optional[str] = None


def compilable(code) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def test_spec_consistency(spec_code: SpecCode):
    exec(spec_code.spec)
    node_t: Type[AbsOpBase] = eval(spec_code.cls_name)
    exec(spec_code.fwd_code)
    fwd = eval("forward_fn")

    # operator creation
    op_param_n = node_t.get_num_var_param()
    op_params = [z3.Int(f"opsym_{k}") for k in range(op_param_n)]
    op: AbsOpBase = node_t(*op_params)

    # input variables
    for _ in range(5):
        ranks = [random.choice(choices) for choices in op.inp_ranks]
        solver = z3.Solver()
        inputs: List[AbsTensor] = []
        for i, (rank, idtypes) in enumerate(zip(ranks, op.in_dtypes[-1])):
            itensor = AbsTensor(
                shape=[z3.Int(f"i{i}_{j}") for j in range(rank)], dtype=idtypes
            )
            inputs.append(itensor)
            for s in itensor.shape:
                solver.add(s >= 1)

        # check input constraints
        solver.add(*op.checked_requires(inputs))
        otensors = op.checked_type_transfer(inputs)
        for ot in otensors:
            solver.add(*ot.gt_zero())

        # TODO: check harder by using more models
        assert solver.check() == z3.sat, f"Cannot solve for {node_t}; input: {inputs}"
        m = solver.model()

        concrete_op = concretize_op(op, m)
        aitensors: List[AbsTensor] = []
        for inp in inputs:
            shape = []
            for s in inp.shape:
                shape.append(m.eval(s).as_long())
            aitensors.append(AbsTensor(shape=shape, dtype=inp.dtype))
        fn = fwd(concrete_op)
        cotensors = fn(
            *[
                random_tensor(shape=cit.shape, dtype=cit.dtype.torch())
                for cit in aitensors
            ]
        )

        if isinstance(cotensors, torch.Tensor):
            cotensors = [cotensors]

        assert isinstance(cotensors, (list, tuple)), f"{type(cotensors)}"

        for cotensor, aotensor in zip(
            cotensors, concrete_op.checked_type_transfer(aitensors)
        ):
            assert (
                cotensor.dtype == aotensor.dtype.torch()
            ), f"{cotensor.dtype} vs {aotensor.dtype.torch()}"
            assert (
                list(cotensor.shape) == aotensor.shape
            ), f"{list(cotensor.shape)} vs {aotensor.shape}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to validate")
    args = parser.parse_args()

    specs = dict()

    paths = sorted(os.listdir(args.path))
    for sample_path in paths:
        abs_sample_path = os.path.join(args.path, sample_path)
        with open(abs_sample_path, "r") as f:
            content = f.read()
        samples = content.split("@@@@@@@@@@@@@@@@")
        samples = [sample for sample in samples if sample.strip()]
        samples = [sample.split("```python")[-1].split("```")[0] for sample in samples]
        for sid, s in enumerate(samples):
            lines = [l for l in s.split("\n") if l]
            if lines[0].startswith("    "):
                for i in range(len(lines)):
                    if lines[i].startswith("    "):
                        lines[i] = lines[i][4:]
            s = "\n".join([l for l in lines if l and not l.startswith("#")])
            dbg_info = f"{abs_sample_path} :: {sid}"
            if "def forward_fn" not in s:
                print(f"--- No `forward_fn` defined: {dbg_info}")
                continue

            spec = s.split("def forward_fn")[0]
            if sum([1 for l in spec.split("\n") if l.startswith("class")]) > 1:
                print(f"--- More than 1 class defined: {dbg_info}")
            fwd_code = "def forward_fn" + s.split("def forward_fn")[1]

            # check compilable
            if not compilable(spec):
                print(f"--- Spec not compilable: {dbg_info}")
                continue

            # check compilable
            if not compilable(fwd_code):
                print(f"--- Fwd code not compilable: {dbg_info}")
                continue

            cls_name = spec.split("class")[1].split("(")[0].strip()
            specs.setdefault(cls_name, []).append(
                SpecCode(cls_name, spec, fwd_code, dbg_info)
            )

    print(colored(f"Compilability: {len(specs)} / {len(paths)} spec types", "yellow"))
    print(
        f"Found {len(specs)} spec classes and {sum(len(l) for l in specs.values())} total samples"
    )

    print("--- dynamically check consistency ...")

    tested_specs = {}
    for cls_name in specs:
        for s in specs[cls_name]:
            try:
                test_spec_consistency(s)
                tested_specs.setdefault(cls_name, []).append(s)
                print(colored(f"--- Validated: {cls_name}", "green"))
            except Exception as e:
                print(colored(f"--- Failed to check consistency: {cls_name}", "red"))
                print(f"--- Sample path: {s.dbg_info}")
                traceback.print_exc()

    print(
        f"Tested {len(tested_specs)} spec classes ~ {sum(len(l) for l in tested_specs.values())} samples"
    )

    with open("torch_additional_spec.py", "w") as f:
        f.write(
            r"""from typing import Callable, List, Optional, Tuple, Union, Type

import z3
import torch

from neuri.abstract.arith import *
from neuri.abstract.op import *
from neuri.abstract.tensor import *
"""
        )
        for cls_name in tested_specs:
            f.write(
                '@mark_materialize("torch")\n' + tested_specs[cls_name][0].spec + "\n"
            )

    with open("torch_additional_fwd.py", "w") as f:
        f.write(
            r"""from neuri.materialize.torch.forward import operator_impl

import torch
"""
        )
        for cls_name in tested_specs:
            f.write(
                f"@operator_impl({cls_name})\n"
                + tested_specs[cls_name][0].fwd_code
                + "\n\n"
            )
