PROMPT = R"""You are now given a docstring of a PyTorch API and please summarize its shaping properties within a markdown code block:

## Original Docstring

'''
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
'''


```

## API signature and argument/return types

## Shape constraints to each input tensor

## Deduction rules of the output shapes from the input shapes

## 1~3 examples of input-output shapes
```

Please ignore information that is irrelevant to shaping properties. Here is the docstring:
"""
