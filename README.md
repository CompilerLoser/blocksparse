# Blocksparse

## Blocksparse installation in ubuntu18.04 and CUDA10
gcc > 5

### Compile tensorflow
* Create a new env python version == 3.6.
* Install bazel
* `git clone https://github.com/tensorflow/tensorflow.git` and checkout to r1.13
* `./configure` 
* `bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`   
* `bazel-bin/tensorflow/tools/pip_package/build_pip_package path/to/install/tf/whl` 
* Install the whl to your env.

### Compile blocksparse
* `export LD_LIBRARY_PATH=path/to/nccl/lib:path/to/mpi/lib:$LD_LIBRARY_PATH`
* `git clone https://github.com/openai/blocksparse.git`   

* change the Makefile as follows:
    * set `-D_CLIBCXX_USE_CXX11_ABI=1` in `CCFLAGS` and `NVCCFLAG`
    * set `CUDA_HOME NCCL_HOME MPI_HOME`
    * add `-gencode=arch=compute_75,code=compute_75` to `NVCCFLAG` (specific to RTX 2080)

you may also need to remove some code in `matmul_op_gpu.cu`.
* comment out the code with `mma_sync` primtives in about line 262
* change `matmul_op_gpu.cu:318` to `if (false)`.

In the new env with tensorflow installed 
* `make compile` and `pip install dist/*.whl`

checkout https://github.com/openai/blocksparse#readme when you after this.
