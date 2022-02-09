#include <stdio.h>
#include <iostream>
#include <fstream>

#include "cuda_kernel.h"
#include "netW.hpp"
#include "utils.cuh"

using namespace std;

// Layer 4 - Convolution (xyz)

__global__ void layer4_conv_kernel(unsigned long long *d_cuda_layer_3_output, float *d_layer_4_bias, unsigned long long *d_cuda_layer_4_weight, signed short *d_cuda_layer_4_output){
    
    int N = 14, kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    int b = blockIdx.x; //batches in x-dir
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b<BATCH_SIZE){
            if(m<NR_NEURONS) {
                d_cuda_layer_4_output[index4D_cuda(b,h,w,m,14,14,64)] = d_layer_4_bias[m]; // = 0;
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH<kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 14) {
                for (int kW = 0; kW<kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 14) {
                        if(b<BATCH_SIZE){
                            if(m<NR_NEURONS) {
                                for (int c = 0; c < 1; c++) {
                                    // atomicAddShort(&d_cuda_layer_4_output[index3D_cuda(h,w,m,14,64)], 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_4_weight[index4D_cuda(kH,kW,m,c,3,64,1)] ^ d_cuda_layer_3_output[index3D_cuda(iH,iW,c,14,64)])) - 64);
                                    d_cuda_layer_4_output[index4D_cuda(b,h,w,m,14,14,64)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_4_weight[index4D_cuda(kH,kW,m,c,3,64,1)] ^ d_cuda_layer_3_output[index4D_cuda(b,iH,iW,c,14,14,64)])) - 64;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer4_conv_cuda(unsigned long long * cuda_layer_3_output, signed short * cuda_layer_4_output){

    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // flatten layer_4_weight
    unsigned long long *cuda_layer_4_weight = (unsigned long long *) layer_4_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_3_output; // storage on device for cuda_layer_3_output
    float *d_layer_4_bias; // storage on device for layer_4_bias
    unsigned long long *d_cuda_layer_4_weight; // storage on device for cuda_layer_4_weight
    signed short *d_cuda_layer_4_output; // RESULT storage on device for cuda_layer_4_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_3_output, BATCH_SIZE*12544*sizeof(unsigned long long)); // 196=14x14 dim of cuda_layer_4_output
    cudaMalloc((void **) &d_layer_4_bias, 64*sizeof(float)); // 64 = dim of layer_4_bias
    cudaMalloc((void **) &d_cuda_layer_4_weight, 36864*sizeof(unsigned long long)); // 576 = 3x3x64x[1x64] dim of layer_4_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_4_output, BATCH_SIZE*12544*sizeof(signed short)); // 12544 = 14x14x64 dim of layer_4_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_3_output, cuda_layer_3_output, (BATCH_SIZE*12544*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_4_bias, layer_4_bias, (64*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_4_weight, cuda_layer_4_weight, (36864*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 14;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 14;
    const int GRIDZSIZE = NR_NEURONS;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 for loops 14 iterations each
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer4_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_3_output, d_layer_4_bias, d_cuda_layer_4_weight, d_cuda_layer_4_output);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // copy result from device to host
    cudaMemcpy(cuda_layer_4_output, d_cuda_layer_4_output, (BATCH_SIZE*12544*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");    
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory
    cudaFree(d_cuda_layer_3_output);
    cudaFree(d_layer_4_bias);
    cudaFree(d_cuda_layer_4_weight);
    cudaFree(d_cuda_layer_4_output);
    cudaCheckErrors("cudaFree fail");

    // checksum L4 = 6334.000000
    // float sum = 0;
    // ofstream g("layer_4_par.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum=0;
    //     for(int i=b*12544;i<(b+1)*12544;i++){
    //         sum += cuda_layer_4_output[i];
    //         g<<cuda_layer_4_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum<<endl;
    // }

    return milliseconds;
}

// Layer 8 - Gemm (xyz)
__global__ void layer8_gemm_kernel(unsigned long long *d_cuda_layer_7_output, float *d_layer_8_bias, unsigned long long *d_cuda_layer_8_weight, signed short *d_cuda_layer_8_output){

    int z = blockDim.x * blockIdx.z + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int d = z*blockDim.x+y;

    int b = blockIdx.x;

    if(d < 2048){
        if(b < BATCH_SIZE){
            d_cuda_layer_8_output[b*2048 + d] = d_layer_8_bias[d];
            for (int i = 0; i < 49; i++) {
                d_cuda_layer_8_output[b*2048 + d] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_8_weight[d*49+i] ^ d_cuda_layer_7_output[i])) - 64;
            }
        }
    }
}

float layer8_gemm_cuda(unsigned long long * cuda_layer_7_output, signed short * cuda_layer_8_output){

    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // flatten layer_8_weight
    unsigned long long *cuda_layer_8_weight = (unsigned long long *) layer_8_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_7_output; // storage on device for cuda_layer_7_output
    float *d_layer_8_bias;  // storage on device for layer_8_bias
    unsigned long long *d_cuda_layer_8_weight; // storage on device for cuda_layer_8_weight
    signed short *d_cuda_layer_8_output; // RESULT storage on device for cuda_layer_8_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_7_output, BATCH_SIZE*49*sizeof(unsigned long long)); // 49=7x7 dim of cuda_layer_7_output
    cudaMalloc((void **) &d_layer_8_bias, 2048*sizeof(float)); // 2048 = dim of layer_8_bias
    cudaMalloc((void **) &d_cuda_layer_8_weight, 100352*sizeof(unsigned long long)); // 100352 = 2048x49 dim of layer_8_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_8_output, BATCH_SIZE*2048*sizeof(signed short)); // 2048 = dim of layer_8_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_7_output, cuda_layer_7_output, (BATCH_SIZE*49*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_8_bias, layer_8_bias, (2048*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_8_weight, cuda_layer_8_weight, (100352*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 32;
    const int BLKYSIZE = 32;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 2;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // 1 for loop 2048 iterations
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer8_gemm_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_7_output, d_layer_8_bias, d_cuda_layer_8_weight, d_cuda_layer_8_output);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // copy result from device to host
    cudaMemcpy(cuda_layer_8_output, d_cuda_layer_8_output, (BATCH_SIZE*2048*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory
    cudaFree(d_cuda_layer_7_output);
    cudaFree(d_layer_8_bias);
    cudaFree(d_cuda_layer_8_weight);
    cudaFree(d_cuda_layer_8_output);
    cudaCheckErrors("cudaFree fail");

    // checksum L8 = 8936.000000
    // float sum = 0;
    // ofstream g("layer_8_par1.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum=0;
    //     for(int i=b*2048;i<(b+1)*2048;i++){
    //         sum += cuda_layer_8_output[i];
    //         g<<cuda_layer_8_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum<<endl;
    // }
    return milliseconds;
}
