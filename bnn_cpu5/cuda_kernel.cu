#include <stdio.h>
#include <iostream>
#include "cuda_kernel.h"

// // set a 3D volume
// // To compile it with nvcc execute: nvcc -O2 -o cuda_matadd.out cuda_matadd.cu
// // Can be found in cuda_kernel.h
//  // define the data set size (cubic volume)
//  #define DATAXSIZE 14
//  #define DATAYSIZE 64
//  #define DATAZSIZE 64
//  // define the chunk sizes that each threadblock will work on
//  // change them and eventually check performance by timing different sizes
//  #define BLKXSIZE 32
//  #define BLKYSIZE 4
//  #define BLKZSIZE 4

 // for cuda error checking
#define cudaCheckErrors(msg) \
do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        return 1; \
    } \
} while (0)

__global__ void set(signed short *d_a, signed short *d_r){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<DATAMUL){
        d_r[i] = 10*d_a[i];
        // printf("%d - %d\n",d_x[i],d_r[i]);
    }
}

int matadd(signed short *a){

    const dim3 threadsPerBlock(BLKXSIZE,BLKYSIZE);
    const dim3 numBlocks(DATAMUL / threadsPerBlock.x, DATAMUL / threadsPerBlock.y);

    // overall data set sizes
 
    signed short *d_a;
    signed short *d_r;  // storage for result computed on device
 
    // allocate GPU device buffers
    cudaMalloc((void **) &d_a, DATAMUL*sizeof(signed short));
    cudaMalloc((void **) &d_r, DATAMUL*sizeof(signed short));
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data to device
    cudaMemcpy(d_a, a, (DATAMUL*sizeof(signed short)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");
    
    // compute result
    set<<<numBlocks,threadsPerBlock>>>(d_a, d_r);
    cudaCheckErrors("Kernel launch failure");

    // copy output data back to host
    cudaMemcpy(a, d_r, (DATAMUL*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // std::cout<<"check a(kernel):"<<std::endl;
    //   for(int i=0; i<DATAXSIZE; i++){
    //     for(int j=0; j<DATAYSIZE; j++){
    //       for(int k=0; k<DATAZSIZE; k++){
    //         std::cout<<a[i][j][k]<<" ";
    //       }
    //       std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    //   }
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");

    printf("Results check!\n");

    cudaFree(d_a);
    cudaFree(d_r);
    cudaCheckErrors("cudaFree fail");

    return 0;
}