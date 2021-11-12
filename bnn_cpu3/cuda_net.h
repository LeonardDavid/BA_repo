#include <stdio.h>
#include "cuda_kernel.h"

int cuda_k(signed short (*a)[DATAYSIZE][DATAXSIZE]){
  return matadd(a);
}