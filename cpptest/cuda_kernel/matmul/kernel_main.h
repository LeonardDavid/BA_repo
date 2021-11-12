#include <stdio.h>
#include "kernel.h"

int matrix_mult(int *x, int *y, int *z){
    return matrix_mult_cuda(x,y,z);
}