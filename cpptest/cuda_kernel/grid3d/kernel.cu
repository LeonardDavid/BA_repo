#include <iostream>
#include "kernel.h"
#include "utils.cuh"

__global__ void  mykernel(short *d_x, short *d_r, short *d_m){
    
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    int m = blockDim.z * blockIdx.z + threadIdx.z;

    if(h<5 && w<5){
        if(m<4){
            d_m[index3D_cuda(h,w,m,5,4)] = m;
            printf("d[%d]=%d\n ",index3D_cuda(h,w,m,5,4),d_m[index3D_cuda(h,w,m,5,4)]);
        }
        for(int i=0; i<3; i++){
            for(int j=0;j<3;j++){
                if(m<4){
                    d_r[index3D_cuda(h,w,m,5,4)] += d_m[index3D_cuda(h,w,m,5,4)]*d_x[index3D_cuda(h,w,m,5,4)]*i*j;
                }
            }
        }
        // d_r[h*5+w] = 10*d_x[h*5+w];
        // printf("%d - %d\n",d_x[i],d_r[i]);
    }
}

int matadd(short *x){

    short *d_x, *d_r, *d_m;
    cudaMalloc((void **) &d_x, 25*sizeof(int));
    cudaMalloc((void **) &d_r, 25*sizeof(int));
    cudaMalloc((void **) &d_m, 100*sizeof(int));

    cudaMemcpy(d_x, x, (25*sizeof(int)), cudaMemcpyHostToDevice);

    const int BLKXSIZE = 5;
    const int BLKYSIZE = 5;
    const int GRIDXSIZE = 1;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 4;
    
    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); 
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    mykernel<<<numBlocks,threadsPerBlock>>>(d_x, d_r, d_m);

    cudaMemcpy(x, d_r, (25*sizeof(int)), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_r);

    std::cout<<"cuda done!"<<std::endl;

    return 0;
}