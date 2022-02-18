#include <stdio.h>
#include <iostream>
#include <fstream>

#include "cuda_kernel.h"
#include "netW.hpp"
#include "utils.cuh"

using namespace std;

/* need to flatten at runtime:

    -layer_1_weight[3][3][1][64]-
    -layer_4_weight[3][3][64][1]-
    -layer_8_weight[2048][49]-
    -layer_10_weight[10][32]-

    TODO:
    now: flatten arrays like layer7: unsigned long long *layer_7_output = (unsigned long long *) layer_6_output;
        maybe flatten in cpp file instead of cuda file
    later: to increase performance, have them flat in the file from the beggining

    cuda steps:
    // flatten 3D -> 1D arrays

    // prepare for kernel call
    // declare storage on device

    // allocate GPU device buffers

    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device

    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes


    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start);
    // compute result - kernel call

    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host

    cudaCheckErrors("CUDA memcpy failure");

    // free the memory

    cudaCheckErrors("cudaFree fail");

    // checksum

    return milliseconds;
*/

// Layer 1 - Convolution

__global__ void layer1_conv_kernel(unsigned char *d_cuda_layer_0_output, float *d_layer_1_bias, signed char *d_cuda_layer_1_weight, float *d_cuda_layer_1_output){

    // https://github.com/ULHPC/tutorials/blob/devel/cuda/exercises/convolution/LoG_gpu_solution.cu

    int N = 28, kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        for(int b=0;b<BATCH_SIZE;b++){
            if(m<NR_NEURONS) {
                d_cuda_layer_1_output[index4D_cuda(b,h,w,m,28,28,64)] = d_layer_1_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH<kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 28) {
                for (int kW = 0; kW<kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 28) {
                    // ii = kj + ix - center;
                    // jj = ki + iy - center;
                        for(int b=0;b<BATCH_SIZE;b++){
                            for (int c = 0; c < 1; c++) {
                                if(m<NR_NEURONS) {
                                    // atomicAdd(&d_cuda_layer_1_output[index4D_cuda(b,bid,tid,m,28,28,64)], d_cuda_layer_1_weight[index4D_cuda(kH,kW,c,m,3,1,64)] * d_cuda_layer_0_output[index4D_cuda(b,iH,iW,c,28,28,1)]);
                                    d_cuda_layer_1_output[index4D_cuda(b,h,w,m,28,28,64)] += d_cuda_layer_1_weight[index4D_cuda(kH,kW,c,m,3,1,64)] * d_cuda_layer_0_output[index4D_cuda(b,iH,iW,c,28,28,1)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
}

float layer1_conv_cuda(unsigned char * const x, float * cuda_layer_1_output){

    // size_t free, total;
    // printf("\n");
    // cudaMemGetInfo(&free,&total);   
    // printf("before: %d KB free of total %d KB\n",free/1024,total/1024);
    
    // initialize layer_0_output where x is the input image
    unsigned char (*layer_0_output)[BATCH_SIZE][28][1] = (unsigned char (*)[BATCH_SIZE][28][1]) x;

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
    cudaMalloc((void **) &d_cuda_layer_0_output, BATCH_SIZE*784*sizeof(unsigned char)); // 784 = 28x28 dim of cuda_layer_0_output
    cudaMalloc((void **) &d_layer_1_bias, 64*sizeof(float)); // 64 = dim of layer_1_bias
    cudaMalloc((void **) &d_cuda_layer_1_weight, 576*sizeof(signed char)); // 576 = 3x3x1x64 dim of layer_1_weight
    cudaMalloc((void **) &d_cuda_layer_1_output, BATCH_SIZE*50176*sizeof(float)); // 50176 = 28x28x64 dim of layer_1_output
    cudaCheckErrors("Failed to allocate device buffer");

    // cudaMemGetInfo(&free,&total);   
    // printf("after: %d KB free of total %d KB\n",free/1024,total/1024);

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_0_output, cuda_layer_0_output, (BATCH_SIZE*784*sizeof(unsigned char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_1_bias, layer_1_bias, (64*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_1_weight, cuda_layer_1_weight, (576*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    // const int dataxsize = 28;
    // const int dataysize = 28;
    // const int datamul = dataxsize*dataysize;
    const int BLKXSIZE = 28;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 28;
    const int GRIDZSIZE = NR_NEURONS;
    
    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 for loops 28 iterations each
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
    cudaMemcpy(cuda_layer_1_output, d_cuda_layer_1_output, (BATCH_SIZE*50176*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_0_output);
    cudaFree(d_layer_1_bias);
    cudaFree(d_cuda_layer_1_weight);
    cudaFree(d_cuda_layer_1_output);
    cudaCheckErrors("cudaFree fail");
    
    // checksum
    /*
        Note: observed small discrepency when calculating the sum in separate program:
        for 1 batch of 50176 elements:
            (-sum original: -605468.8125)
            - sum here:     -605468.8125
            - sum separate: -605476.240866
        this difference is constant, could be caused by rounding errors
        the outputs appear to be the same as the original implementation (including the sum)
        -> not important for now, but good to know in case something does not add up later
    */
    // float sum = 0;
    // ofstream g("layer_1_par.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum=0;
    //     for(int i=b*50176;i<(b+1)*50176;i++){
    //         sum += cuda_layer_1_output[i];
    //         g<<cuda_layer_1_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum<<endl;
    // }
    return milliseconds;
}

// Layer 2 - Maxpool

__global__ void layer2_maxpool_kernel(float *d_cuda_layer_1_output, float *d_cuda_layer_2_output, float lowest){

    int N = 14, kernel_size = 2;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int c = blockIdx.z; // neurons in z-dir

    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(c<NR_NEURONS) {
            d_cuda_layer_2_output[index3D_cuda(h,w,c,14,64)] = lowest;
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH<kernel_size; kH++){
            for (int kW = 0; kW<kernel_size; kW++){
                for(int b = 0; b < BATCH_SIZE; b++){
                    if(c<NR_NEURONS) {
                        d_cuda_layer_2_output[index3D_cuda(h,w,c,14,64)] = fmax(d_cuda_layer_1_output[index4D_cuda(b,(h * 2 + kH),(w * 2 + kW),c,28,28,64)], d_cuda_layer_2_output[index3D_cuda(h,w,c,14,64)]);
                    }
                }
            }
        }
    }
}

float layer2_maxpool_cuda(float * cuda_layer_1_output, float * cuda_layer_2_output){
    // flatten 3D -> 1D arrays
    // no 3D arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    float *d_cuda_layer_1_output; // storage on device for cuda_layer_1_output
    float *d_cuda_layer_2_output; // RESULT storage on device for cuda_layer_2_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_1_output, 50176*sizeof(float)); // 50176 = 28x28x64 dim of layer_1_output
    cudaMalloc((void **) &d_cuda_layer_2_output, 12544*sizeof(float)); // 12544 = 14x14x64 dim of layer_2_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_1_output, cuda_layer_1_output, (50176*sizeof(float)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 14;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = 1;
    const int GRIDYSIZE = 14;
    const int GRIDZSIZE = NR_NEURONS;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 foor loops 14 iterations each
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
    layer2_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_1_output, d_cuda_layer_2_output, LOWEST);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_2_output, d_cuda_layer_2_output, (12544*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_1_output);
    cudaFree(d_cuda_layer_2_output);
    cudaCheckErrors("cudaFree fail");

    // checksum
    // float sum = 0;
    // ofstream g("layer_2_par.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum=0;
    //     for(int i=b*12544;i<(b+1)*12544;i++){
    //         sum += cuda_layer_2_output[i];
    //         g<<cuda_layer_2_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum<<endl;
    // }
    return milliseconds;
}

// Layer 4 - Convolution

__global__ void layer4_conv_kernel(unsigned long long *d_cuda_layer_3_output, float *d_layer_4_bias, unsigned long long *d_cuda_layer_4_weight, signed short *d_cuda_layer_4_output){
    
    int N = 28, kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(m<NR_NEURONS) {
            d_cuda_layer_4_output[index3D_cuda(h,w,m,14,64)] = d_layer_4_bias[m]; // = 0;
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
                        if(m<NR_NEURONS) {
                            for (int c = 0; c < 1; c++) {
                                // atomicAddShort(&d_cuda_layer_4_output[index3D_cuda(h,w,m,14,64)], 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_4_weight[index4D_cuda(kH,kW,m,c,3,64,1)] ^ d_cuda_layer_3_output[index3D_cuda(iH,iW,c,14,64)])) - 64);
                                d_cuda_layer_4_output[index3D_cuda(h,w,m,14,64)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_4_weight[index4D_cuda(kH,kW,m,c,3,64,1)] ^ d_cuda_layer_3_output[index3D_cuda(iH,iW,c,14,64)])) - 64;
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer4_conv_cuda(unsigned long long * cuda_layer_3_output, signed short * cuda_layer_4_output){
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
    cudaMalloc((void **) &d_cuda_layer_3_output, 12544*sizeof(unsigned long long)); // 196=14x14 dim of cuda_layer_4_output
    cudaMalloc((void **) &d_layer_4_bias, 64*sizeof(float)); // 64 = dim of layer_4_bias
    cudaMalloc((void **) &d_cuda_layer_4_weight, 36864*sizeof(unsigned long long)); // 576 = 3x3x64x[1x64] dim of layer_4_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_4_output, 12544*sizeof(signed short)); // 12544 = 14x14x64 dim of layer_4_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_3_output, cuda_layer_3_output, (12544*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_4_bias, layer_4_bias, (64*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_4_weight, cuda_layer_4_weight, (36864*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 14;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = 1;
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

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");    
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_4_output, d_cuda_layer_4_output, (12544*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_3_output);
    cudaFree(d_layer_4_bias);
    cudaFree(d_cuda_layer_4_weight);
    cudaFree(d_cuda_layer_4_output);
    cudaCheckErrors("cudaFree fail");

    // checksum
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

// Layer 5 - Maxpool
__global__ void layer5_maxpool_kernel(signed short * d_cuda_layer_4_output, signed short * d_cuda_layer_5_output, signed short lowest){

    int N = 14, kernel_size = 2;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int c = blockIdx.z; // neurons in z-dir

    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(c<NR_NEURONS) {
            d_cuda_layer_5_output[index3D_cuda(h,w,c,7,64)] = lowest;
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH<kernel_size; kH++){
            for (int kW = 0; kW<kernel_size; kW++){
                if(c<NR_NEURONS) {
                    d_cuda_layer_5_output[index3D_cuda(h,w,c,7,64)] = 
                    (d_cuda_layer_4_output[index3D_cuda((h * 2 + kH),(w * 2 + kW),c,14,64)] >= d_cuda_layer_5_output[index3D_cuda(h,w,c,7,64)]) ? 
                    d_cuda_layer_4_output[index3D_cuda((h * 2 + kH),(w * 2 + kW),c,14,64)] : d_cuda_layer_5_output[index3D_cuda(h,w,c,7,64)];
                }
            }
        }
    }
}

float layer5_maxpool_cuda(signed short * cuda_layer_4_output, signed short * cuda_layer_5_output){
    // flatten 3D -> 1D arrays
    // no arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    signed short *d_cuda_layer_4_output; // storage on device for cuda_layer_4_output
    signed short *d_cuda_layer_5_output; // RESULT storage on device for cuda_layer_5_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_4_output, 12544*sizeof(signed short)); // 12544 = 14x14xx64 dim of layer_4_output
    cudaMalloc((void **) &d_cuda_layer_5_output, 3136*sizeof(signed short)); // 3136 = 7x7x64 dim of layer_5_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_4_output, cuda_layer_4_output, (12544*sizeof(signed short)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 7;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = 1;
    const int GRIDYSIZE = 7;
    const int GRIDZSIZE = NR_NEURONS;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 foor loops 7 iterations each
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // std library not allowed on device
    const signed short LOWEST = std::numeric_limits<signed short>::lowest();

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer5_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_4_output, d_cuda_layer_5_output, LOWEST);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_5_output, d_cuda_layer_5_output, (3136*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_4_output);
    cudaFree(d_cuda_layer_5_output);
    cudaCheckErrors("cudaFree fail");

    // checksum
    // float sum = 0;
    // ofstream g("layer_5_par.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum=0;
    //     for(int i=b*3136;i<(b+1)*3136;i++){
    //         sum += cuda_layer_5_output[i];
    //         g<<cuda_layer_5_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum<<endl;
    // }
    return milliseconds;
}

// Layer 6 - Step
// skipped for now

// Layer 8 - Gemm
__global__ void layer8_gemm_kernel(unsigned long long *d_cuda_layer_7_output, float *d_layer_8_bias, unsigned long long *d_cuda_layer_8_weight, signed short *d_cuda_layer_8_output){

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int d = x*32+y;

    if(d < 2048){
        d_cuda_layer_8_output[d] = d_layer_8_bias[d];
        for (int i = 0; i < 49; i++) {
            d_cuda_layer_8_output[d] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_8_weight[d*49+i] ^ d_cuda_layer_7_output[i])) - 64;
        }
    }
}

float layer8_gemm_cuda(unsigned long long * cuda_layer_7_output, signed short * cuda_layer_8_output){
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
    cudaMalloc((void **) &d_cuda_layer_7_output, 49*sizeof(unsigned long long)); // 49=7x7 dim of cuda_layer_7_output
    cudaMalloc((void **) &d_layer_8_bias, 2048*sizeof(float)); // 2048 = dim of layer_8_bias
    cudaMalloc((void **) &d_cuda_layer_8_weight, 100352*sizeof(unsigned long long)); // 100352 = 2048x49 dim of layer_8_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_8_output, 2048*sizeof(signed short)); // 2048 = dim of layer_8_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_7_output, cuda_layer_7_output, (49*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_8_bias, layer_8_bias, (2048*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_8_weight, cuda_layer_8_weight, (100352*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 32;
    const int BLKYSIZE = 32;
    const int GRIDXSIZE = 2;
    const int GRIDYSIZE = 1;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // 1 for loop 2048 iterations
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE);

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

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_8_output, d_cuda_layer_8_output, (2048*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_7_output);
    cudaFree(d_layer_8_bias);
    cudaFree(d_cuda_layer_8_weight);
    cudaFree(d_cuda_layer_8_output);
    cudaCheckErrors("cudaFree fail");

    // checksum
    // int sum = 0;
    // for (int i = 0; i < 2048; i++) {
    //     sum += cuda_layer_8_output[i];
    //     // cout<<cuda_layer_4_output[i]<<" ";   
    // }
    return milliseconds;
}

// Layer 10 - Gemm
__global__ void layer10_gemm_kernel(unsigned long long *d_cuda_layer_9_output, float *d_layer_10_bias, unsigned long long *d_cuda_layer_10_weight, signed short *d_cuda_layer_10_output){

    int d = blockDim.x * blockIdx.x + threadIdx.x;

    if(d < 10){
        d_cuda_layer_10_output[d] = d_layer_10_bias[d];
        for (int i = 0; i < 32; i++) {
            d_cuda_layer_10_output[d] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_10_weight[d*32+i] ^ d_cuda_layer_9_output[i])) - 64;
        }
    }
}

float layer10_gemm_cuda(unsigned long long * cuda_layer_9_output, signed short * cuda_layer_10_output){
    // flatten 3D -> 1D arrays
    // flatten layer_10_weight
    unsigned long long *cuda_layer_10_weight = (unsigned long long *) layer_10_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_9_output; // storage on device for cuda_layer_9_output
    float *d_layer_10_bias;  // storage on device for layer_10_bias
    unsigned long long *d_cuda_layer_10_weight; // storage on device for cuda_layer_10_weight
    signed short *d_cuda_layer_10_output; // RESULT storage on device for cuda_layer_10_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_9_output, 32*sizeof(unsigned long long)); // 32 = dim of cuda_layer_9_output
    cudaMalloc((void **) &d_layer_10_bias, 10*sizeof(float)); // 10 = dim of layer_10_bias
    cudaMalloc((void **) &d_cuda_layer_10_weight, 320*sizeof(unsigned long long)); // 320 = 32x10 dim of layer_10_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_10_output, 10*sizeof(signed short)); // 10 = dim of layer_10_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_9_output, cuda_layer_9_output, (32*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_10_bias, layer_10_bias, (10*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_10_weight, cuda_layer_10_weight, (320*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 10;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = 1;
    const int GRIDYSIZE = 1;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // 1 for loop 10 iterations
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer10_gemm_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_9_output, d_layer_10_bias, d_cuda_layer_10_weight, d_cuda_layer_10_output);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // copy result from device to host
    cudaMemcpy(cuda_layer_10_output, d_cuda_layer_10_output, (10*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_9_output);
    cudaFree(d_layer_10_bias);
    cudaFree(d_cuda_layer_10_weight);
    cudaFree(d_cuda_layer_10_output);
    cudaCheckErrors("cudaFree fail");

    // checksum
    // int sum = 0;
    // for (int i = 0; i < 10; i++) {
    //     sum += cuda_layer_10_output[i];
    //     // cout<<cuda_layer_4_output[i]<<" ";   
    // }
    return milliseconds;
}

