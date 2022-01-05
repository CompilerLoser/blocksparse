# Blocksparse

## Blocksparse installation in ubuntu18.04 and CUDA10
The tensorflow and blocksparse installed through pip are always not compatible with each other, we need to build these two libraries by source code compilation. You need a fixed gcc compiler to compile them, and ensure that it is at least version 5 or higher, otherwise there will be some symbols in the compiled library that cannot be matched.

### Compile tensorflow
* Create a new env and make sure the python version is 3.6 to adapt to the tensorflow.
* Install bazel
* `git clone https://github.com/tensorflow/tensorflow.git` and checkout to r1.13
* `./configure` 
* `bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`
* `bazel-bin/tensorflow/tools/pip_package/build_pip_package path/to/install/tf/whl` and install the whl to your env.

### Compile blocksparse
Before compile blocksparse, you need to:
* `export LD_LIBRARY_PATH=path/to/nccl/lib:path/to/mpi/lib:$LD_LIBRARY_PATH`
* `git clone https://github.com/openai/blocksparse.git` and 
