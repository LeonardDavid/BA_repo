#include <stdio.h>
#include "kernel.h"

torch::Tensor matrix_mult(torch::Tensor x, torch::Tensor y, torch::Tensor z){
    return matrix_mult_cuda(x,y,z);
}