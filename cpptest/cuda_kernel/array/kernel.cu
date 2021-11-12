#include <iostream>
#include "kernel.h"

__global__ void  mykernel(short *d_x, short *d_r, int n){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<n){
        d_r[i] = 10*d_x[i];
        // printf("%d - %d\n",d_x[i],d_r[i]);
    }
}

int matadd(short *x){

    short *d_x, *d_r;
    cudaMalloc((void **) &d_x, 5*sizeof(int));
    cudaMalloc((void **) &d_r, 5*sizeof(int));

    cudaMemcpy(d_x, x, (5*sizeof(int)), cudaMemcpyHostToDevice);

    mykernel<<<1,5>>>(d_x, d_r, 5);

    cudaMemcpy(x, d_r, (5*sizeof(int)), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_r);

    std::cout<<"cuda done!"<<std::endl;

    return 0;
}