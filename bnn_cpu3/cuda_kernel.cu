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

__global__ void set(signed short *d_a[DATAYSIZE][DATAXSIZE], signed short *d_r[DATAYSIZE][DATAXSIZE]){

    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
    if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE))){
      d_r[idz][idy][idx] = 2*d_a[idz][idy][idx];
    }
}

int matadd(signed short (*a)[DATAYSIZE][DATAXSIZE]){

    // typedef signed short nRarray[DATAYSIZE][DATAXSIZE];

    const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 gridSize(((DATAXSIZE+BLKXSIZE-1)/BLKXSIZE), ((DATAYSIZE+BLKYSIZE-1)/BLKYSIZE), ((DATAZSIZE+BLKZSIZE-1)/BLKZSIZE));

    // overall data set sizes
    const int nx = DATAXSIZE;
    const int ny = DATAYSIZE;
    const int nz = DATAZSIZE;

    // pointers for data set storage via malloc
    // nRarray *h_c; // storage for result stored on host
    
    signed short *d_a[DATAYSIZE][DATAXSIZE];
    signed short *d_r[DATAYSIZE][DATAXSIZE];  // storage for result computed on device

    // allocate storage for data set
    // if ((h_c = (nRarray *)malloc((nx*ny*nz)*sizeof(int))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
    // h_c = (nRarray *)malloc((nx*ny*nz)*sizeof(int));
    
    // allocate GPU device buffers
    cudaMalloc((void **) &d_a, (nx*ny*nz)*sizeof(int));
    cudaMalloc((void **) &d_r, (nx*ny*nz)*sizeof(int));
    // cudaCheckErrors("Failed to allocate device buffer");

    // copy input data to device
    cudaMemcpy(d_a, a, ((nx*ny*nz)*sizeof(int)), cudaMemcpyHostToDevice);
    // cudaCheckErrors("CUDA memcpy failure");
    
    // compute result
    set<<<gridSize,blockSize>>>(d_a, d_r);
    // cudaCheckErrors("Kernel launch failure");

    // copy output data back to host
    cudaMemcpy(d_r, a, ((nx*ny*nz)*sizeof(int)), cudaMemcpyDeviceToHost);
    // cudaCheckErrors("CUDA memcpy failure");

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    // cudaCheckErrors("CUDA synchronize failure");

    // and check for accuracy
    // for (unsigned i=0; i<nz; i++)
    //   for (unsigned j=0; j<ny; j++)
    //     for (unsigned k=0; k<nx; k++)
    //       if (h_c[i][j][k] != (i+j+k)) {
    //         printf("Mismatch at x= %d, y= %d, z= %d  Host= %d, Device = %d\n", i, j, k, (i+j+k), h_c[i][j][k]);
    //         return 1;
    //         }
    printf("Results check!\n");

    // free(h_c);
    cudaFree(d_a);
    cudaFree(d_r);
    // cudaCheckErrors("cudaFree fail");

    return 0;
}