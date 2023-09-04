#!/bin/bash

set -e
set -x

cd "$(dirname "$0")" || exit 1

git clone https://github.com/pytorch/pytorch.git pytorch-cov
cd pytorch-cov
git checkout f7520cb51e7208fd7c4c0d57d786c7b0207718bc
git submodule sync
git submodule update --init --recursive
git apply ../instrument-torch.patch
# PyTorch dependencies
conda install -y astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
conda install mkl mkl-include
pip install -r requirements.txt
USE_CPP_CODE_COVERAGE=1 \
USE_KINETO=0 BUILD_CAFFE2=0 USE_DISTRIBUTED=0 USE_NCCL=0 BUILD_TEST=0 USE_XNNPACK=0 \
USE_QNNPACK=0 USE_MIOPEN=0 BUILD_CAFFE2_OPS=0 USE_TENSORPIPE=0 \
USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0 CC=clang-14 CXX=clang++-14 \
python setup.py develop
