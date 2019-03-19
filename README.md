## Install

System Requirements:
- Install [git-lfs](https://git-lfs.github.com/)
- CUDA 10 in the system

Steps:
```
git clone --recursive git@github.com:simon-mo/fijit-sys.git
cd fijit-sys
mkdir build
cd build
cmake ..
make -j8 fijit-sys

# testing
./fijit-sys --model ../data/sanity_check.onnx --input ../data/resnet50_input.onnx --max-block 20
```

Usage:
```
./fijit-sys -h
FIJIT Inference Engine
Usage:
  fijit-sys [OPTION...]

  -m, --model arg       Path to the model ONNX file
  -i, --input arg       Path to the input ONNX file
      --max-block arg   Max block for TVM ops
      --input-name arg  Override input tensor name
  -h, --help            Print help

```
