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
./fijit-sys ../data/resnet50.onnx ../data/resnet50_input.onnx 0
```
