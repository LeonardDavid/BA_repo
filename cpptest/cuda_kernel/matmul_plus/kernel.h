#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

#include <pybind11/pybind.h>

#define SIZEX 3
#define SIZEY 3
#define SIZEMUL SIZEX*SIZEY

#define BLKXSIZE 32
#define BLKYSIZE 32
// max number of threads/block: 32x32 = 1024

#define GRIDXSIZE SIZEMUL/SIZEX
#define GRIDYSIZE SIZEMUL/SIZEY

torch::Tensor matrix_mult_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor z);

PYBIND11_MODULE(matmul, m){
    m.def("matmul", &matrix_mult_cuda);
}

inline int index(int i, int j){
    return i*SIZEX + j;
}