#include <stdio.h>
#include "cuda_kernel.h"

float layer1_conv(unsigned char * const x, float * layer){
  return layer1_conv_cuda(x, layer);
}