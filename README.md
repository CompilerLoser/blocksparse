# Blocksparse

## Blocksparse installation in ubuntu18.04 and CUDA10
The tensorflow and blocksparse installed through pip are always not compatible with each other, we need to build these two libraries by source. We need a fixed gcc compiler to compile them, and ensure that it is at least version 5 or higher, otherwise there may be some unmatched symbols in the compiled library.

### Compile tensorflow
* Create a new env and make sure the python version is 3.6.
* Install bazel
* `git clone https://github.com/tensorflow/tensorflow.git` and checkout to r1.13
* `./configure` 
* `bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`   
just google if there is something wrong.
* `bazel-bin/tensorflow/tools/pip_package/build_pip_package path/to/install/tf/whl` 
* Install the whl to your env.

### Compile blocksparse
Before compile blocksparse, you need to:
* `export LD_LIBRARY_PATH=path/to/nccl/lib:path/to/mpi/lib:$LD_LIBRARY_PATH`
* `git clone https://github.com/openai/blocksparse.git`   

* and change the Makefile as follows:
    * set `-D_CLIBCXX_USE_CXX11_ABI=1` in `CCFLAGS` and `NVCCFLAG`
    * set `CUDA_HOME NCCL_HOME MPI_HOME`
    * add `-gencode=arch=compute_75,code=compute_75` to `NVCCFLAG` (specific to RTX 2080)
    * make sure use the same compiler with tensorflow. 

Due to some CUDA and driver version issues(I guess?) you may also need to remove some code in `matmul_op_gpu.cu`.
* comment out the code with `mma_sync` primtives in about line 262
* and use low version `hmma_gemm_64x64x32_TN_vec8` function to replace it: change `matmul_op_gpu.cu:318` to `if (false)`.
* for me who want to use sparse attention, this change has no effect, but if you need its matmul gpu kernel, you can use the API defined in `/usr/local/cuda/include/mma.h` to rewrite it :)


In the new env with tensorflow installed, run:  
* `make compile` and `pip install dist/*.whl`

Go to https://github.com/openai/blocksparse#readme when you after this.
