#include <stdio.h>
#include "cuda_kernel.h"

int predict_NeuralNet(unsigned char * const x, float * pred){
  return predict_NeuralNet_Cuda(x, pred);
}