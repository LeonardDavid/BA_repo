#include <stdio.h>
#include "cuda_kernel.h"

int cuda_k(int c){
  matadd(c);
  return 0;
}