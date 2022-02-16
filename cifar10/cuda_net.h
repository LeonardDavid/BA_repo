#include <stdio.h>
#include "cuda_kernel.h"

float layer1_conv(unsigned char x[][32][32][3], float * layer){ // unsigned char * const x / unsigned char x[][32][32][3]
  return layer1_conv_cuda(x, layer);
}