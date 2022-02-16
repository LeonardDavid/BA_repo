#include <stdio.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <chrono>

#include "cuda_kernel.h"
#include "netW.hpp"
#include "utils.cuh"

using namespace std;

__global__ void layer1_conv_kernel(unsigned char *d_cuda_layer_0_output, float *d_layer_1_bias, signed char *d_cuda_layer_1_weight, float *d_cuda_layer_1_output){

    // https://github.com/ULHPC/tutorials/blob/devel/cuda/exercises/convolution/LoG_gpu_solution.cu

    int N = 32, kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    // batches in x-dir
    int b = blockIdx.x;
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(m < 128) {
                d_cuda_layer_1_output[index4D_cuda(b,h,w,m,32,32,128)] = d_layer_1_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 32) {
                for (int kW = 0; kW < kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 32) {
                        if(b < BATCH_SIZE){
                            for (int c = 0; c < 3; c++) {
                                if(m < 128) {
                                    d_cuda_layer_1_output[index4D_cuda(b,h,w,m,32,32,128)] += d_cuda_layer_1_weight[index4D_cuda(kH,kW,c,m,3,3,128)] * d_cuda_layer_0_output[index4D_cuda(b,iH,iW,c,32,32,3)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer1_conv_cuda(unsigned char x[][32][32][3], float * cuda_layer_1_output){ // unsigned char * const x / unsigned char x[][32][32][3]
    
    setUniGPU();// use the second GPU on Uni-server because the first is used most of the time

    // initialize layer_0_output where x is the input image
    unsigned char (*layer_0_output)[BATCH_SIZE][32][32][3] = (unsigned char (*)[BATCH_SIZE][32][32][3]) x;

    // flatten 3D -> 1D arrays
    // flatten layer_1_weight
    signed char *cuda_layer_1_weight = (signed char *) layer_1_weight;

    // flatten layer_0_output
    unsigned char *cuda_layer_0_output = (unsigned char *) layer_0_output;
// printf("0\n");
    // prepare for kernel call
    // declare storage on device
    unsigned char *d_cuda_layer_0_output; // storage on device for cuda_layer_0_output
    float *d_layer_1_bias; // storage on device for layer_1_bias
    signed char *d_cuda_layer_1_weight; // storage on device for cuda_layer_1_weight
    float *d_cuda_layer_1_output; // RESULT storage on device for cuda_layer_1_output
// printf("1\n");
    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_0_output, BATCH_SIZE*3072*sizeof(unsigned char)); // 3072 = 32x32x3 dim of cuda_layer_0_output
    // printf("1.1: %lu\n", BATCH_SIZE*3072*sizeof(unsigned char));
    cudaMalloc((void **) &d_layer_1_bias, 128*sizeof(float)); // 128 = dim of layer_1_bias
    // printf("1.2: %lu\n", 128*sizeof(float));
    cudaMalloc((void **) &d_cuda_layer_1_weight, 3456*sizeof(signed char)); // 3456 = 3x3x3x128 dim of layer_1_weight
    // printf("1.3: %lu\n", 3456*sizeof(signed char));
    cudaMalloc((void **) &d_cuda_layer_1_output, BATCH_SIZE*131072*sizeof(float)); // 131072 = 32x32x128 dim of layer_1_output
    // printf("1.4 :%lu\n", BATCH_SIZE*131072*sizeof(float));
    cudaCheckErrors("Failed to allocate device buffer");
// printf("2\n");
    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_0_output, cuda_layer_0_output, (BATCH_SIZE*3072*sizeof(unsigned char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_1_bias, layer_1_bias, (128*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_1_weight, cuda_layer_1_weight, (3456*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");
// printf("3\n");
    // define thread and block sizes
    const int BLKXSIZE = 32;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 32;
    const int GRIDZSIZE = 128;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);
    
    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
// printf("4\n");
    // compute result - kernel call
    cudaEventRecord(start);
    layer1_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_0_output, d_layer_1_bias, d_cuda_layer_1_weight, d_cuda_layer_1_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);
// printf("5\n");
    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);
// printf("6\n");
    // copy result from device to host
    cudaMemcpy(cuda_layer_1_output, d_cuda_layer_1_output, (BATCH_SIZE*131072*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");
// printf("7\n");
    // free the memory
    cudaFree(d_cuda_layer_0_output);
    cudaFree(d_layer_1_bias);
    cudaFree(d_cuda_layer_1_weight);
    cudaFree(d_cuda_layer_1_output);
    cudaCheckErrors("cudaFree fail");
// printf("8\n");
    // float sum = 0;
    // ofstream g("layer1/par.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum=0;
    //     for(int i=b*131072;i<(b+1)*131072;i++){
    //         sum += cuda_layer_1_output[i];
    //         g<<cuda_layer_1_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum<<endl;
    // }
    
    return milliseconds;
}