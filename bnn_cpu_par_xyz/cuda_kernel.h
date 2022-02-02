#include "utils.h"

float layer1_conv_cuda(unsigned char * const x, float * layer, size_t nr_layer);

float layer2_maxpool_cuda(float * layer1, float * layer2, size_t nr_layer);

float layer3_step_cuda(float * layer1, unsigned long long * layer2, size_t nr_layer);

float layer4_conv_cuda(unsigned long long * layer1, signed short * layer2, size_t nr_layer);

float layer5_maxpool_cuda(signed short * layer1, signed short * layer2, size_t nr_layer);

float layer8_gemm_cuda(unsigned long long * layer1, signed short * layer2, size_t nr_layer);

float layer10_gemm_cuda(unsigned long long * layer1, signed short * layer2, size_t nr_layer);
