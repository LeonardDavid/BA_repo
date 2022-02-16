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

    // prepare for kernel call
    // declare storage on device
    unsigned char *d_cuda_layer_0_output; // storage on device for cuda_layer_0_output
    float *d_layer_1_bias; // storage on device for layer_1_bias
    signed char *d_cuda_layer_1_weight; // storage on device for cuda_layer_1_weight
    float *d_cuda_layer_1_output; // RESULT storage on device for cuda_layer_1_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_0_output, BATCH_SIZE*32*32*3*sizeof(unsigned char)); // 3072 = 32x32x3 dim of cuda_layer_0_output
    cudaMalloc((void **) &d_layer_1_bias, 128*sizeof(float)); // 128 = dim of layer_1_bias
    cudaMalloc((void **) &d_cuda_layer_1_weight, 3*3*3*128*sizeof(signed char)); // 3456 = 3x3x3x128 dim of layer_1_weight
    cudaMalloc((void **) &d_cuda_layer_1_output, BATCH_SIZE*32*32*128*sizeof(float)); // 131072 = 32x32x128 dim of layer_1_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_0_output, cuda_layer_0_output, (BATCH_SIZE*32*32*3*sizeof(unsigned char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_1_bias, layer_1_bias, (128*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_1_weight, cuda_layer_1_weight, (3*3*3*128*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

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

    // compute result - kernel call
    cudaEventRecord(start);
    layer1_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_0_output, d_layer_1_bias, d_cuda_layer_1_weight, d_cuda_layer_1_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_1_output, d_cuda_layer_1_output, (BATCH_SIZE*32*32*128*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_0_output);
    cudaFree(d_layer_1_bias);
    cudaFree(d_cuda_layer_1_weight);
    cudaFree(d_cuda_layer_1_output);
    cudaCheckErrors("cudaFree fail");

    // // checksum L1 = 5720315.5
    // float sum_gpu = 0;
    // ofstream gg1("layer1/par.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum_gpu = 0;
    //     for(int i=b*32*32*128;i<(b+1)*32*32*128;i++){
    //         sum_gpu += cuda_layer_1_output[i];
    //         gg1<<cuda_layer_1_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"layer 1(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    
    return milliseconds;
}

__global__ void layer3_conv_kernel(unsigned long long *d_cuda_layer_2_output, float *d_layer_3_bias, unsigned long long *d_cuda_layer_3_weight, float *d_cuda_layer_3_output){

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
                d_cuda_layer_3_output[index4D_cuda(b,h,w,m,32,32,128)] = d_layer_3_bias[m];
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
                            for (int c = 0; c < 2; c++) {
                                if(m < 128) {
                                    d_cuda_layer_3_output[index4D_cuda(b,h,w,m,32,32,128)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_3_weight[index4D_cuda(kH,kW,m,c,3,128,2)] ^ d_cuda_layer_2_output[index4D_cuda(b,iH,iW,c,32,32,128)])) - 64; // ,128?
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer3_conv_cuda(unsigned long long * cuda_layer_2_output, float * cuda_layer_3_output){
    
    setUniGPU();// use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // flatten layer_3_weight
    signed char *cuda_layer_3_weight = (signed char *) layer_3_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_2_output; // storage on device for cuda_layer_2_output
    float *d_layer_3_bias; // storage on device for layer_3_bias
    unsigned long long *d_cuda_layer_3_weight; // storage on device for cuda_layer_3_weight
    float *d_cuda_layer_3_output; // RESULT storage on device for cuda_layer_3_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_2_output, BATCH_SIZE*32*32*2*64*sizeof(unsigned long long)); // 131072 = 32x32x2x64 dim of cuda_layer_2_output
    cudaMalloc((void **) &d_layer_3_bias, 128*sizeof(float)); // 128 = dim of layer_3_bias
    cudaMalloc((void **) &d_cuda_layer_3_weight, 3*3*128*2*64*sizeof(unsigned long long)); // 147456 = 3x3x128x2x64 dim of layer_3_weight
    cudaMalloc((void **) &d_cuda_layer_3_output, BATCH_SIZE*32*32*128*sizeof(float)); // 131072 = 32x32x128 dim of layer_3_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_2_output, cuda_layer_2_output, (BATCH_SIZE*32*32*2*64*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_3_bias, layer_3_bias, (128*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_3_weight, cuda_layer_3_weight, (3*3*128*2*64*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

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

    // compute result - kernel call
    cudaEventRecord(start);
    layer3_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_2_output, d_layer_3_bias, d_cuda_layer_3_weight, d_cuda_layer_3_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_3_output, d_cuda_layer_3_output, (BATCH_SIZE*32*32*128*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_2_output);
    cudaFree(d_layer_3_bias);
    cudaFree(d_cuda_layer_3_weight);
    cudaFree(d_cuda_layer_3_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer4_maxpool_kernel(float *d_cuda_layer_3_output, float *d_cuda_layer_4_output, float lowest){

    int N = 16, kernel_size = 2;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int c = blockIdx.z; // neurons in z-dir

    int b = blockIdx.x; // Batches index in grid x dir
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(c < 128) {
                d_cuda_layer_4_output[index4D_cuda(b,h,w,c,16,16,128)] = lowest;
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            for (int kW = 0; kW < kernel_size; kW++){
                if(b < BATCH_SIZE){
                    if(c < 128) {
                        d_cuda_layer_4_output[index4D_cuda(b,h,w,c,16,16,128)] = fmax(d_cuda_layer_3_output[index4D_cuda(b,(h * 2 + kH),(w * 2 + kW),c,32,32,128)], d_cuda_layer_4_output[index4D_cuda(b,h,w,c,16,16,128)]);
                    }
                }
            }
        }
    }
}

float layer4_maxpool_cuda(float * cuda_layer_3_output, float * cuda_layer_4_output){

    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // no 3D arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    float *d_cuda_layer_3_output; // storage on device for cuda_layer_3_output
    float *d_cuda_layer_4_output; // RESULT storage on device for cuda_layer_4_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_3_output, BATCH_SIZE*32*32*128*sizeof(float)); // 131072 = 32x32x128 dim of layer_3_output
    cudaMalloc((void **) &d_cuda_layer_4_output, BATCH_SIZE*16*16*128*sizeof(float)); // 32768 = 16x16x128 dim of layer_4_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_3_output, cuda_layer_3_output, (BATCH_SIZE*32*32*128*sizeof(float)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 16;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 16;
    const int GRIDZSIZE = 128;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // std library not allowed on device
    const float LOWEST = std::numeric_limits<float>::lowest();

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer4_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_3_output, d_cuda_layer_4_output, LOWEST);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_4_output, d_cuda_layer_4_output, (BATCH_SIZE*16*16*128*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory
    cudaFree(d_cuda_layer_3_output);
    cudaFree(d_cuda_layer_4_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

float layer6_conv_cuda(unsigned long long * cuda_layer_5_output, float * cuda_layer_6_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    return 0;
}

float layer8_conv_cuda(unsigned long long * cuda_layer_7_output, float * cuda_layer_8_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    return 0;
}

float layer9_maxpool_cuda(float * cuda_layer_8_output, float * cuda_layer_9_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    return 0;
}

float layer11_conv_cuda(unsigned long long * cuda_layer_10_output, float * cuda_layer_11_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    return 0;
}

float layer13_conv_cuda(unsigned long long * cuda_layer_12_output, float * cuda_layer_13_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    return 0;
}

float layer14_maxpool_cuda(float * cuda_layer_13_output, float * cuda_layer_14_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    return 0;
}

float layer17_gemm_cuda(unsigned long long * cuda_layer_16_output, float * cuda_layer_17_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    return 0;
}

float layer19_gemm_cuda(unsigned long long * cuda_layer_18_output, float * cuda_layer_19_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    return 0;
}
