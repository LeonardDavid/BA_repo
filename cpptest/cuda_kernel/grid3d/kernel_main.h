#include <stdio.h>
#include "kernel.h"

int cuda_k(short *x){
    return matadd(x);
}