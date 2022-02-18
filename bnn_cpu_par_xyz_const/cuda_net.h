#include <stdio.h>
#include "cuda_kernel.h"

float layer1_conv(unsigned char * const x, float * layer, size_t nr_layer){
  return layer1_conv_cuda(x, layer, nr_layer);
}

float layer2_maxpool(float * layer1, float * layer2, size_t nr_layer){
  return layer2_maxpool_cuda(layer1, layer2, nr_layer);
}

// float layer3_step(float * layer1, unsigned long long * layer2, size_t nr_layer){
//   return layer3_step_cuda(layer1, layer2, nr_layer);
// }

float layer4_conv(unsigned long long * layer1, signed short * layer2, size_t nr_layer){
  return layer4_conv_cuda(layer1, layer2, nr_layer);
}

float layer5_maxpool(signed short * layer1, signed short * layer2, size_t nr_layer){
  return layer5_maxpool_cuda(layer1, layer2, nr_layer);
}

float layer8_gemm(unsigned long long * layer1, signed short * layer2, size_t nr_layer){
  return layer8_gemm_cuda(layer1, layer2, nr_layer);
}

float layer10_gemm(unsigned long long * layer1, signed short * layer2, size_t nr_layer){
  return layer10_gemm_cuda(layer1, layer2, nr_layer);
}