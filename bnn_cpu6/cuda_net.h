#include <stdio.h>
#include "cuda_kernel.h"

int cuda_k(signed short *a){
  return matadd(a);
}