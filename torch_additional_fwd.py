from neuri.materialize.torch.forward import operator_impl

import torch
@operator_impl(Det)
def forward_fn(op: Det) -> Callable:
    return lambda x: torch.linalg.det(x)

@operator_impl(InvOp)
def forward_fn(op: InvOp) -> Callable:
    return torch.linalg.inv

@operator_impl(MatrixExp)
def forward_fn(op: MatrixExp) -> Callable:
    return lambda x: torch.linalg.matrix_exp(x)

@operator_impl(AdaptiveAvgPool1d)
def forward_fn(op: AdaptiveAvgPool1d) -> Callable:
    return lambda x: torch.nn.functional.adaptive_avg_pool1d(x, op.output_size)

@operator_impl(AdaptiveMaxPool2d)
def forward_fn(op: AdaptiveMaxPool2d) -> Callable:
    return lambda x: torch.nn.functional.adaptive_max_pool2d(x, op.output_size)

@operator_impl(Dropout)
def forward_fn(op: Dropout) -> Callable:
    return lambda x: torch.nn.functional.dropout(x, p=op.p, training=True, inplace=False)

@operator_impl(ELU)
def forward_fn(op: ELU) -> Callable:
    return lambda x: torch.nn.functional.elu(x, alpha=op.alpha)

@operator_impl(FeatureAlphaDropout)
def forward_fn(op: FeatureAlphaDropout) -> Callable:
    return lambda x: torch.nn.functional.feature_alpha_dropout(
        x, p=op.p, training=True, inplace=False
    )

@operator_impl(Hardswish)
def forward_fn(op: Hardswish) -> Callable:
    return torch.nn.functional.hardswish

@operator_impl(HardTanh)
def forward_fn(op: HardTanh) -> Callable:
    return lambda x: torch.nn.functional.hardtanh(x, op.min_val, op.max_val)

@operator_impl(SELU)
def forward_fn(op: SELU) -> Callable:
    return torch.nn.functional.selu

@operator_impl(Sigmoid)
def forward_fn(op: Sigmoid) -> Callable:
    return torch.nn.functional.sigmoid

@operator_impl(SiLU)
def forward_fn(op: SiLU) -> Callable:
    return torch.nn.functional.silu

@operator_impl(Softshrink)
def forward_fn(op: Softshrink) -> Callable:
    return lambda x: torch.nn.functional.softshrink(x, op.lambd)

@operator_impl(Softsign)
def forward_fn(op: Softsign) -> Callable:
    return torch.nn.functional.softsign

@operator_impl(Threshold)
def forward_fn(op: Threshold) -> Callable:
    return lambda x: torch.nn.functional.threshold(x, op.threshold, op.value)

@operator_impl(CosineWindow)
def forward_fn(op: CosineWindow) -> Callable:
    return lambda x: torch.signal.windows.cosine(op.M, sym=True, dtype=torch.float32)

@operator_impl(Nuttall)
def forward_fn(op: Nuttall) -> Callable:
    return lambda x: torch.signal.windows.nuttall(op.M, sym=True, dtype=torch.float32)

