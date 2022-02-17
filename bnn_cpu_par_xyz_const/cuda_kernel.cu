#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

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

    // copy result from device to host

    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory

    cudaCheckErrors("cudaFree fail");

    // checksum

    return milliseconds;
*/

// Layer 1 - Convolution

__global__ void layer1_conv_kernel(unsigned char *d_cuda_layer_0_output, float *d_layer_1_bias, signed char *d_cuda_layer_1_weight, float *d_cuda_layer_1_output, 
    const size_t ISh, const size_t ISw, const size_t OSh, const size_t OSw, const size_t OSm, const size_t KSh, const size_t KSw, const size_t KSc, const size_t s0, const size_t s1, const size_t p0, const size_t p1){

    int b = blockIdx.x; // batches in x-dir
    int m = blockIdx.z; // neurons in z-dir

    // https://github.com/ULHPC/tutorials/blob/devel/cuda/exercises/convolution/LoG_gpu_solution.cu

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (KSh - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (KSw - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*OSh +ix;

    // bias is applied to every pixel
    if(tid < OSh){
        if(b < BATCH_SIZE){
            if(m < OSm) {
                d_cuda_layer_1_output[index4D_cuda(b,h,w,m,OSh,OSw,OSm)] = d_layer_1_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < OSh*OSw){
        for (int kH = 0; kH < KSh; kH++){
            int iH = h * s0 + kH - p0;
            if (iH >= 0 && iH < ISh) {
                for (int kW = 0; kW < KSw; kW++){
                    int iW = w * s1 + kW - p1;
                    if (iW >= 0 && iW < ISw) {
                        if(b < BATCH_SIZE){
                            for (int c = 0; c < KSc; c++) {
                                if(m < OSm) {
                                    // atomicAdd(&d_cuda_layer_1_output[index4D_cuda(b,bid,tid,m,28,28,64)], d_cuda_layer_1_weight[index4D_cuda(kH,kW,c,m,3,1,64)] * d_cuda_layer_0_output[index4D_cuda(b,iH,iW,c,28,28,1)]);
                                    d_cuda_layer_1_output[index4D_cuda(b,h,w,m,OSh,OSw,OSm)] += d_cuda_layer_1_weight[index4D_cuda(kH,kW,c,m,KSw,KSc,OSm)] * d_cuda_layer_0_output[index4D_cuda(b,iH,iW,c,ISh,ISw,KSc)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
}

float layer1_conv_cuda(unsigned char * const x, float * cuda_layer_1_output, size_t nr_layer){
    
    // printf("layer: %zu\n",nr_layer);
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // initialize layer_0_output where x is the input image
    unsigned char (*layer_0_output)[BATCH_SIZE][28][1] = (unsigned char (*)[BATCH_SIZE][28][1]) x;

    // declare constants for current layer
    const size_t ISh = input_shape[nr_layer][1];       // 28
    const size_t ISw = input_shape[nr_layer][2];       // 28
    // const size_t ISm = input_shape[nr_layer][0];       // 64
    const size_t OSh = output_shape[nr_layer][1];      // 28
    const size_t OSw = output_shape[nr_layer][2];      // 28
    const size_t OSm = output_shape[nr_layer][0];      // 64
    const size_t KSh = kernel_shape[nr_layer][1];      // 3
    const size_t KSw = kernel_shape[nr_layer][2];      // 3
    const size_t KSc = kernel_shape[nr_layer][0];      // 1
    const size_t s0 = strides[nr_layer][0];            // 1
    const size_t s1 = strides[nr_layer][1];            // 1
    const size_t p0 = pads[nr_layer][0];               // 1
    const size_t p1 = pads[nr_layer][1];               // 1

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
    const size_t size0 = BATCH_SIZE*ISh*ISw; // BATCH_SIZE*28*28 = input shape of layer1 = cuda_layer_0_output
    const size_t size1 = BATCH_SIZE*OSh*OSw*OSm; // BATCH_SIZE*28*28*64 = output shape of layer1 = cuda_layer_1_output

    cudaMalloc((void **) &d_cuda_layer_0_output, size0*sizeof(unsigned char)); // 784 = 28x28 dim of cuda_layer_0_output
    cudaMalloc((void **) &d_layer_1_bias, bias_size[nr_layer]*sizeof(float)); // 64 = dim of layer_1_bias
    cudaMalloc((void **) &d_cuda_layer_1_weight, weight_size[nr_layer]*sizeof(signed char)); // 576 = 3x3x1x64 dim of layer_1_weight
    cudaMalloc((void **) &d_cuda_layer_1_output, size1*sizeof(float)); // 50176 = 28x28x64 dim of layer_1_output
    cudaCheckErrors("Failed to allocate device buffer");

    // cudaMemGetInfo(&free,&total);   
    // printf("after: %d KB free of total %d KB\n",free/1024,total/1024);

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_0_output, cuda_layer_0_output, (size0*sizeof(unsigned char)), cudaMemcpyHostToDevice); // BATCH_SIZE*28*28
    cudaMemcpy(d_layer_1_bias, layer_1_bias, (bias_size[nr_layer]*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_1_weight, cuda_layer_1_weight, (weight_size[nr_layer]*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = OSw; // 28
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = OSh; // 28
    const int GRIDZSIZE = OSm; // 64
    
    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 for loops 28 iterations each
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer1_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_0_output, d_layer_1_bias, d_cuda_layer_1_weight, d_cuda_layer_1_output,
        ISh, ISw, OSh, OSw, OSm, KSh, KSw, KSc, s0, s1, p0, p1);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_1_output, d_cuda_layer_1_output, (size1*sizeof(float)), cudaMemcpyDeviceToHost); // BATCH_SIZE*28*28*64
    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory
    cudaFree(d_cuda_layer_0_output);
    cudaFree(d_layer_1_bias);
    cudaFree(d_cuda_layer_1_weight);
    cudaFree(d_cuda_layer_1_output);
    cudaCheckErrors("cudaFree fail");
    
    // checksum L1 = -605468.812500
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

__global__ void layer2_maxpool_kernel(float *d_cuda_layer_1_output, float *d_cuda_layer_2_output, float lowest,
    const size_t ISh, const size_t ISw, const size_t ISm, const size_t OSh, const size_t OSw, const size_t OSm, const size_t KSh, const size_t KSw, const size_t s0, const size_t s1){

    int b = blockIdx.x; // batches in x-dir
    int c = blockIdx.z; // neurons in z-dir

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (KSh - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (KSw - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*OSh +ix;

    // bias is applied to every pixel
    if(tid < OSh){
        if(b < BATCH_SIZE){
            if(c < OSm) {
                d_cuda_layer_2_output[index4D_cuda(b,h,w,c,OSh,OSw,OSm)] = lowest;
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < OSh*OSw){
        for (int kH = 0; kH < KSh; kH++){
            for (int kW = 0; kW < KSw; kW++){
                if(b < BATCH_SIZE){
                    if(c < OSm) {
                        d_cuda_layer_2_output[index4D_cuda(b,h,w,c,OSh,OSw,OSm)] = fmax(d_cuda_layer_1_output[index4D_cuda(b,(h * s0 + kH),(w * s1 + kW),c,ISh,ISw,OSm)], d_cuda_layer_2_output[index4D_cuda(b,h,w,c,OSh,OSw,OSm)]);
                    }
                }
            }
        }
    }
}

float layer2_maxpool_cuda(float * cuda_layer_1_output, float * cuda_layer_2_output, size_t nr_layer){
    // printf("layer: %zu\n",nr_layer);
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // declare constants for current layer
    const size_t ISh = input_shape[nr_layer][1];       // 28
    const size_t ISw = input_shape[nr_layer][2];       // 28
    const size_t ISm = input_shape[nr_layer][0];       // 64
    const size_t OSh = output_shape[nr_layer][1];      // 14
    const size_t OSw = output_shape[nr_layer][2];      // 14
    const size_t OSm = output_shape[nr_layer][0];      // 64
    const size_t KSh = kernel_shape[nr_layer][1];      // 2
    const size_t KSw = kernel_shape[nr_layer][2];      // 2
    // const size_t KSc = kernel_shape[nr_layer][0];      // 0
    const size_t s0 = strides[nr_layer][0];            // 2
    const size_t s1 = strides[nr_layer][1];            // 2
    // const size_t p0 = pads[nr_layer][0];               // 0
    // const size_t p1 = pads[nr_layer][1];               // 0

    // flatten 3D -> 1D arrays
    // no 3D arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    float *d_cuda_layer_1_output; // storage on device for cuda_layer_1_output
    float *d_cuda_layer_2_output; // RESULT storage on device for cuda_layer_2_output

    // allocate GPU device buffers
    const size_t size1 = BATCH_SIZE*ISh*ISw*ISm; // BATCH_SIZE*28*28*64 = input shape of layer2 = cuda_layer_1_output
    const size_t size2 = BATCH_SIZE*OSh*OSw*OSm; // BATCH_SIZE*14*14*64 = output shape of layer2 = cuda_layer_2_output

    cudaMalloc((void **) &d_cuda_layer_1_output, size1*sizeof(float)); // 50176 = 28x28x64 dim of layer_1_output
    cudaMalloc((void **) &d_cuda_layer_2_output, size2*sizeof(float)); // 12544 = 14x14x64 dim of layer_2_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_1_output, cuda_layer_1_output, (size1*sizeof(float)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = OSw;   // 14
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = OSh;  // 14
    const int GRIDZSIZE = OSm;  // 64

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
    layer2_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_1_output, d_cuda_layer_2_output, LOWEST,
        ISh, ISw, ISm, OSh, OSw, OSm, KSh, KSw, s0, s1);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_2_output, d_cuda_layer_2_output, (size2*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory
    cudaFree(d_cuda_layer_1_output);
    cudaFree(d_cuda_layer_2_output);
    cudaCheckErrors("cudaFree fail");

    // checksum L2 = 455610.125000
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

// Layer 3 - Step
// TODO WORK IN PROGRESS

__global__ void layer3_step_kernel(float *d_cuda_layer_2_output, signed short *d_layer_3_threshold, unsigned long long *d_cuda_layer_3_output, unsigned long long *d_res_cuda_layer_3_output){


    // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    // int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;

    /* 
        - why only 4096 iterations?? using printf("1 "); and counting the 1s
            (also true for every other kernel, sometimes the number ranges between 3000-9000)
        - thread divergence? -> shouldn't lead to different results, just a performance impact
            -> underlying problem is still the fact that the kernel is not executing all 12544 threads as it should
    */

    // printf("1 ");
    if(h<14 && w<14 && c<64){ // && c<64
        // printf("1 ");
        // for(int c=0;c<64;c++){
            // printf("1 ");
            if (d_cuda_layer_2_output[index3D_cuda(h,w,c,14,64)] > d_layer_3_threshold[c]) {
                // printf("%d ",d_layer_3_threshold[c]);
                // if(h==1)
                    // printf("%llu ",d_cuda_layer_3_output[index3D_cuda(h,w,c,14,64)]);
                // printf("(%llu - ",d_res_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)]);
                d_res_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)] = d_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)] | (1ULL << (63 - c % 64));
                // d_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)] |= (1ULL << (63 - c % 64));
                // printf("%llu) ",d_res_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)]);
            } else {
                // printf("%d ",d_layer_3_threshold[c]);
                // if(h==1)
                    // printf("~%llu ",d_cuda_layer_3_output[index3D_cuda(h,w,c,14,64)]);
                // printf("(%llu - ",d_res_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)]);
                d_res_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)] = d_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)] & ~(1ULL << (63 - c % 64));
                // d_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)] &= ~(1ULL << (63 - c % 64));
                // printf("%llu) ",d_res_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)]);
            }
        // }
    }
}

float layer3_step_cuda(float * cuda_layer_2_output, unsigned long long * cuda_layer_3_output, size_t nr_layer){
    // flatten 3D -> 1D arrays
    // layer_3_output filled with 64 ULL values at each 14x14 points

    // prepare for kernel call
    // declare storage on device
    float *d_cuda_layer_2_output;   // storage on device for cuda_layer_2_output
    signed short *d_layer_3_threshold; // storage on device for layer_3_threshold
    unsigned long long *d_cuda_layer_3_output; // storage on device for cuda_layer_3_output
    unsigned long long *d_res_cuda_layer_3_output; // RESULT storage on device for cuda_layer_3_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_2_output, 12544*sizeof(float)); // 12544 = 14x14x64 dim of cuda_layer_2_output
    cudaMalloc((void **) &d_layer_3_threshold, 64*sizeof(signed short)); // 64 = dim of layer_3_threshold
    cudaMalloc((void **) &d_cuda_layer_3_output, 12544*sizeof(unsigned long long)); // 196 = 14x14x[1x64] dim of cuda_layer_3_output [ULL]
    cudaMalloc((void **) &d_res_cuda_layer_3_output, 12544*sizeof(unsigned long long)); 
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_2_output, cuda_layer_2_output, (12544*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_3_threshold, layer_3_threshold, (64*sizeof(signed short)), cudaMemcpyHostToDevice);

    cudaMemcpy(d_cuda_layer_3_output, cuda_layer_3_output, (12544*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int DATAXSIZE = 14;
    const int DATAYSIZE = 14;
    const int DATAZSIZE = 64;

    const int BLKXSIZE = 14;
    const int BLKYSIZE = 14;
    const int BLKZSIZE = 4;

    const int GRIDXSIZE = (DATAXSIZE+BLKXSIZE-1)/BLKXSIZE; // 4
    const int GRIDYSIZE = (DATAYSIZE+BLKYSIZE-1)/BLKYSIZE; // 4
    const int GRIDZSIZE = (DATAZSIZE+BLKZSIZE-1)/BLKZSIZE; // 1

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE, BLKZSIZE); // the 3 for loops 14x14x64 iterations
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer3_step_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_2_output, d_layer_3_threshold, d_cuda_layer_3_output, d_res_cuda_layer_3_output);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // copy result from device to host
    cudaMemcpy(cuda_layer_3_output, d_res_cuda_layer_3_output, (12544*sizeof(unsigned long long)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory
    cudaFree(d_cuda_layer_2_output);
    cudaFree(d_layer_3_threshold);
    cudaFree(d_cuda_layer_3_output);
    cudaFree(d_res_cuda_layer_3_output);
    cudaCheckErrors("cudaFree fail");

    // checksum
    // int sum = 0;
    // // summation of ULL values leads to overflow -> sum up only the last digit
    // for (int i = 0; i < 12544; i++) {
    //     // if(i%196==0)
    //     //     cout<<endl;
    //     // cout<<cuda_layer_3_output[i]<<" ";
    //     sum += cuda_layer_3_output[i]%10; 
    // }

    return milliseconds;
}

// END WORK IN PROGRESS

// Layer 4 - Convolution

__global__ void layer4_conv_kernel(unsigned long long *d_cuda_layer_3_output, float *d_layer_4_bias, unsigned long long *d_cuda_layer_4_weight, signed short *d_cuda_layer_4_output,
    const size_t ISh, const size_t ISw, const size_t ISm, const size_t OSh, const size_t OSw, const size_t OSm, const size_t KSh, const size_t KSw, const size_t KSc, const size_t s0, const size_t s1, const size_t p0, const size_t p1){

    int b = blockIdx.x; // batches in x-dir
    int m = blockIdx.z; // neurons in z-dir

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (KSh - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (KSw - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*OSh +ix;

    // bias is applied to every pixel
    if(tid < OSh){
        if(b < BATCH_SIZE){
            if(m < OSm) {
                d_cuda_layer_4_output[index4D_cuda(b,h,w,m,OSh,OSw,OSm)] = d_layer_4_bias[m]; // = 0;
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < OSh*OSw){
        for (int kH = 0; kH < KSh; kH++){
            int iH = h * s0 + kH - p0;
            if (iH >= 0 && iH < ISh) {
                for (int kW = 0; kW < KSw; kW++){
                    int iW = w * s1 + kW - p1;
                    if (iW >= 0 && iW < ISw) {
                        if(b < BATCH_SIZE){
                            if(m < OSm) {
                                for (int c = 0; c < ISm/BINARY_WORD_SIZE; c++) {
                                    // atomicAddShort(&d_cuda_layer_4_output[index3D_cuda(h,w,m,14,64)], 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_4_weight[index4D_cuda(kH,kW,m,c,3,64,1)] ^ d_cuda_layer_3_output[index3D_cuda(iH,iW,c,14,64)])) - 64);
                                    d_cuda_layer_4_output[index4D_cuda(b,h,w,m,OSh,OSw,OSm)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_4_weight[index4D_cuda(kH,kW,m,c,KSw,OSm,ISm/BINARY_WORD_SIZE)] ^ d_cuda_layer_3_output[index4D_cuda(b,iH,iW,c,ISh,ISw,ISm)])) - BINARY_WORD_SIZE;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer4_conv_cuda(unsigned long long * cuda_layer_3_output, signed short * cuda_layer_4_output, size_t nr_layer){
    // printf("layer: %zu\n",nr_layer);
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // declare constants for current layer
    const size_t ISh = input_shape[nr_layer][1];       // 14
    const size_t ISw = input_shape[nr_layer][2];       // 14
    const size_t ISm = input_shape[nr_layer][0];       // 64
    const size_t OSh = output_shape[nr_layer][1];      // 14
    const size_t OSw = output_shape[nr_layer][2];      // 14
    const size_t OSm = output_shape[nr_layer][0];      // 64
    const size_t KSh = kernel_shape[nr_layer][1];      // 3
    const size_t KSw = kernel_shape[nr_layer][2];      // 3
    const size_t KSc = kernel_shape[nr_layer][0];      // 1
    const size_t s0 = strides[nr_layer][0];            // 1
    const size_t s1 = strides[nr_layer][1];            // 1
    const size_t p0 = pads[nr_layer][0];               // 1
    const size_t p1 = pads[nr_layer][1];               // 1

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
    const size_t size3 = BATCH_SIZE*ISh*ISw*ISm; // BATCH_SIZE*14*14*64 = input shape of layer4 = cuda_layer_3_output
    const size_t size4 = BATCH_SIZE*OSh*OSw*OSm; // BATCH_SIZE*14*14*64 = output shape of layer4 = cuda_layer_4_output
    const size_t weight4 = 64*weight_size[nr_layer]; // 36864=64x576=3x3x64x[1x64] dim of layer_4_weight [ULL]

    cudaMalloc((void **) &d_cuda_layer_3_output, size3*sizeof(unsigned long long)); // 12544=64x196=14x14 dim of cuda_layer_4_output
    cudaMalloc((void **) &d_layer_4_bias, bias_size[nr_layer]*sizeof(float)); // 64 = dim of layer_4_bias
    cudaMalloc((void **) &d_cuda_layer_4_weight, weight4*sizeof(unsigned long long)); 
    cudaMalloc((void **) &d_cuda_layer_4_output, size4*sizeof(signed short)); // 12544 = 14x14x64 dim of layer_4_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_3_output, cuda_layer_3_output, (size3*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_4_bias, layer_4_bias, (bias_size[nr_layer]*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_4_weight, cuda_layer_4_weight, (weight4*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = OSw;   // 14
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = OSh;  // 14
    const int GRIDZSIZE = OSm;  // 64

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 for loops 14 iterations each
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer4_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_3_output, d_layer_4_bias, d_cuda_layer_4_weight, d_cuda_layer_4_output,
        ISh, ISw, ISm, OSh, OSw, OSm, KSh, KSw, KSc, s0, s1, p0, p1);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // copy result from device to host
    cudaMemcpy(cuda_layer_4_output, d_cuda_layer_4_output, (size4*sizeof(signed short)), cudaMemcpyDeviceToHost);
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

// Layer 5 - Maxpool
__global__ void layer5_maxpool_kernel(signed short * d_cuda_layer_4_output, signed short * d_cuda_layer_5_output, signed short lowest,
    const size_t ISh, const size_t ISw, const size_t ISm, const size_t OSh, const size_t OSw, const size_t OSm, const size_t KSh, const size_t KSw, const size_t s0, const size_t s1){

    int b = blockIdx.x; // batches in x-dir
    int c = blockIdx.z; // neurons in z-dir

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x + (KSh - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (KSw - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*OSh +ix;

    // bias is applied to every pixel
    if(tid < OSh){
        if(b < BATCH_SIZE){
            if(c < OSm) {
                d_cuda_layer_5_output[index4D_cuda(b,h,w,c,OSh,OSw,OSm)] = lowest;
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < OSh*OSw){
        for (int kH = 0; kH < KSh; kH++){
            for (int kW = 0; kW < KSw; kW++){
                if(b < BATCH_SIZE){
                    if(c < OSm) {
                        d_cuda_layer_5_output[index4D_cuda(b,h,w,c,OSh,OSw,OSm)] = 
                        (d_cuda_layer_4_output[index4D_cuda(b,(h * s0 + kH),(w * s1 + kW),c,ISh,ISw,OSm)] >= d_cuda_layer_5_output[index4D_cuda(b,h,w,c,OSh,OSw,OSm)]) ? 
                        d_cuda_layer_4_output[index4D_cuda(b,(h * s0 + kH),(w * s1 + kW),c,ISh,ISw,OSm)] : d_cuda_layer_5_output[index4D_cuda(b,h,w,c,OSh,OSw,OSm)];
                    }
                }
            }
        }
    }
}

float layer5_maxpool_cuda(signed short * cuda_layer_4_output, signed short * cuda_layer_5_output, size_t nr_layer){
    // printf("layer: %zu\n",nr_layer);
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // declare constants for current layer
    const size_t ISh = input_shape[nr_layer][1];       // 14
    const size_t ISw = input_shape[nr_layer][2];       // 14
    const size_t ISm = input_shape[nr_layer][0];       // 64
    const size_t OSh = output_shape[nr_layer][1];      // 7
    const size_t OSw = output_shape[nr_layer][2];      // 7
    const size_t OSm = output_shape[nr_layer][0];      // 64
    const size_t KSh = kernel_shape[nr_layer][1];      // 2
    const size_t KSw = kernel_shape[nr_layer][2];      // 2
    // const size_t KSc = kernel_shape[nr_layer][0];      // 0
    const size_t s0 = strides[nr_layer][0];            // 2
    const size_t s1 = strides[nr_layer][1];            // 2
    // const size_t p0 = pads[nr_layer][0];               // 0
    // const size_t p1 = pads[nr_layer][1];               // 0


    // flatten 3D -> 1D arrays
    // no arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    signed short *d_cuda_layer_4_output; // storage on device for cuda_layer_4_output
    signed short *d_cuda_layer_5_output; // RESULT storage on device for cuda_layer_5_output

    // allocate GPU device buffers
    const size_t size4 = BATCH_SIZE*ISh*ISw*ISm; // BATCH_SIZE*14*14*64 = input shape of layer5 = cuda_layer_4_output
    const size_t size5 = BATCH_SIZE*OSh*OSw*OSm; // BATCH_SIZE*7*7*64 = output shape of layer5 = cuda_layer_5_output

    cudaMalloc((void **) &d_cuda_layer_4_output, size4*sizeof(signed short)); // 12544 = 14x14x64 dim of layer_4_output
    cudaMalloc((void **) &d_cuda_layer_5_output, size5*sizeof(signed short)); // 3136 = 7x7x64 dim of layer_5_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_4_output, cuda_layer_4_output, (size4*sizeof(signed short)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = OSw;   // 7
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = OSh;  // 7
    const int GRIDZSIZE = OSm;  // 64

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
    layer5_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_4_output, d_cuda_layer_5_output, LOWEST,
        ISh, ISw, ISm, OSh, OSw, OSm, KSh, KSw, s0, s1);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // copy result from device to host
    cudaMemcpy(cuda_layer_5_output, d_cuda_layer_5_output, (size5*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory
    cudaFree(d_cuda_layer_4_output);
    cudaFree(d_cuda_layer_5_output);
    cudaCheckErrors("cudaFree fail");

    // checksum = 81406.0000
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
__global__ void layer8_gemm_kernel(unsigned long long *d_cuda_layer_7_output, float *d_layer_8_bias, unsigned long long *d_cuda_layer_8_weight, signed short *d_cuda_layer_8_output,
    const size_t ISm, const size_t OSm){

    int z = blockDim.x * blockIdx.z + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int d = z*blockDim.x+y;

    int b = blockIdx.x;

    if(d < OSm){
        if(b < BATCH_SIZE){
            d_cuda_layer_8_output[b*OSm + d] = d_layer_8_bias[d];
            for (int i = 0; i < ISm/BINARY_WORD_SIZE; i++) {
                d_cuda_layer_8_output[b*OSm + d] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_8_weight[d*(ISm/BINARY_WORD_SIZE)+i] ^ d_cuda_layer_7_output[b*ISm/BINARY_WORD_SIZE+i])) - BINARY_WORD_SIZE;
            }
        }
    }
}

float layer8_gemm_cuda(unsigned long long * cuda_layer_7_output, signed short * cuda_layer_8_output, size_t nr_layer){
    // printf("layer: %zu\n",nr_layer);
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // const size_t ISh = input_shape[nr_layer][1];       // 1
    // const size_t ISw = input_shape[nr_layer][2];       // 1
    const size_t ISm = input_shape[nr_layer][0];       // 3136
    const size_t OSh = output_shape[nr_layer][1];      // 1
    const size_t OSw = output_shape[nr_layer][2];      // 1
    const size_t OSm = output_shape[nr_layer][0];      // 2048
    // const size_t KSh = kernel_shape[nr_layer][1];      // 0
    // const size_t KSw = kernel_shape[nr_layer][2];      // 0
    // const size_t KSc = kernel_shape[nr_layer][0];      // 0
    // const size_t s0 = strides[nr_layer][0];            // 0
    // const size_t s1 = strides[nr_layer][1];            // 0
    // const size_t p0 = pads[nr_layer][0];               // 0
    // const size_t p1 = pads[nr_layer][1];               // 0

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
    const size_t size7 = BATCH_SIZE*(ISm/BINARY_WORD_SIZE); // BATCH_SIZE*(3136/64) = BATCH_SIZE*49 = input shape (offset by BINARY_WORD_SIZE) of layer8 = cuda_layer_7_output
    const size_t size8 = BATCH_SIZE*OSh*OSw*OSm; // BATCH_SIZE*1*1*2048 = output shape of layer8 = cuda_layer_8_output

    cudaMalloc((void **) &d_cuda_layer_7_output, size7*sizeof(unsigned long long)); // 49=7x7 dim of cuda_layer_7_output
    cudaMalloc((void **) &d_layer_8_bias, bias_size[nr_layer]*sizeof(float)); // 2048 = dim of layer_8_bias
    cudaMalloc((void **) &d_cuda_layer_8_weight, weight_size[nr_layer]*sizeof(unsigned long long)); // 100352 = 2048x49 dim of layer_8_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_8_output, size8*sizeof(signed short)); // 2048 = dim of layer_8_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_7_output, cuda_layer_7_output, (size7*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_8_bias, layer_8_bias, (bias_size[nr_layer]*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_8_weight, cuda_layer_8_weight, (weight_size[nr_layer]*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    /*
        Maximum threads in a block: 1024 => Maximum block size 32x32
        OSm can get bigger than 1024, if that is the case, then set block size to maximum (32x32) and put multiple blocks in z-dir
        else if OSm is smaller than 1024, then only create 1 (square) block in z-dir, of size ceil(sqrt(OSm))
    */
    size_t bxy = std::min(32.0, std::ceil(sqrt(OSm)));
    size_t gz = std::max(1.0, std::ceil(OSm/1024));
    const int BLKXSIZE = bxy;   // 32
    const int BLKYSIZE = bxy;   // 32
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = gz;   // 2

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // 1 for loop 2048 iterations
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer8_gemm_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_7_output, d_layer_8_bias, d_cuda_layer_8_weight, d_cuda_layer_8_output,
        ISm, OSm);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // copy result from device to host
    cudaMemcpy(cuda_layer_8_output, d_cuda_layer_8_output, (size8*sizeof(signed short)), cudaMemcpyDeviceToHost);
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
    // ofstream g("layer_8_par.out");
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

// Layer 10 - Gemm
__global__ void layer10_gemm_kernel(unsigned long long *d_cuda_layer_9_output, float *d_layer_10_bias, unsigned long long *d_cuda_layer_10_weight, signed short *d_cuda_layer_10_output,
    const size_t ISm, const size_t OSm){

    int z = blockDim.x * blockIdx.z + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int d = z*blockDim.x+y;

    int b = blockIdx.x;

    if(d < OSm){
        if(b<BATCH_SIZE){
            d_cuda_layer_10_output[b*OSm + d] = d_layer_10_bias[d];
            for (int i = 0; i < ISm/BINARY_WORD_SIZE; i++) {
                d_cuda_layer_10_output[b*OSm + d] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_10_weight[d*(ISm/BINARY_WORD_SIZE)+i] ^ d_cuda_layer_9_output[b*ISm/BINARY_WORD_SIZE+i])) - BINARY_WORD_SIZE;
            }
        }
    }
}

float layer10_gemm_cuda(unsigned long long * cuda_layer_9_output, signed short * cuda_layer_10_output, size_t nr_layer){
    // printf("layer: %zu\n",nr_layer);
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // const size_t ISh = input_shape[nr_layer][1];       // 1
    // const size_t ISw = input_shape[nr_layer][2];       // 1
    const size_t ISm = input_shape[nr_layer][0];       // 2048
    const size_t OSh = output_shape[nr_layer][1];      // 1
    const size_t OSw = output_shape[nr_layer][2];      // 1
    const size_t OSm = output_shape[nr_layer][0];      // 10
    // const size_t KSh = kernel_shape[nr_layer][1];      // 0
    // const size_t KSw = kernel_shape[nr_layer][2];      // 0
    // const size_t KSc = kernel_shape[nr_layer][0];      // 0
    // const size_t s0 = strides[nr_layer][0];            // 0
    // const size_t s1 = strides[nr_layer][1];            // 0
    // const size_t p0 = pads[nr_layer][0];               // 0
    // const size_t p1 = pads[nr_layer][1];               // 0

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
    const size_t size9 = BATCH_SIZE*(ISm/BINARY_WORD_SIZE); // BATCH_SIZE*(2048/64) = BATCH_SIZE*32 = input shape (offset by BINARY_WORD_SIZE) of layer10 = cuda_layer_9_output
    const size_t size10 = BATCH_SIZE*OSh*OSw*OSm; // BATCH_SIZE*1*1*10 = output shape of layer10 = cuda_layer_10_output

    cudaMalloc((void **) &d_cuda_layer_9_output, size9*sizeof(unsigned long long)); // 32 = dim of cuda_layer_9_output
    cudaMalloc((void **) &d_layer_10_bias, bias_size[nr_layer]*sizeof(float)); // 10 = dim of layer_10_bias
    cudaMalloc((void **) &d_cuda_layer_10_weight, weight_size[nr_layer]*sizeof(unsigned long long)); // 320 = 32x10 dim of layer_10_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_10_output, size10*sizeof(signed short)); // 10 = dim of layer_10_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_9_output, cuda_layer_9_output, (size9*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_10_bias, layer_10_bias, (bias_size[nr_layer]*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_10_weight, cuda_layer_10_weight, (weight_size[nr_layer]*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    /*
        Maximum threads in a block: 1024 => Maximum block size 32x32
        OSm can get bigger than 1024, if that is the case, then set block size to maximum (32x32) and put multiple blocks in z-dir
        else if OSm is smaller than 1024, then only create 1 (square) block in z-dir, of size ceil(sqrt(OSm))
    */
    size_t bxy = std::min(32.0, std::ceil(sqrt(OSm)));
    size_t gz = std::max(1.0, std::ceil(OSm/1024));
    const int BLKXSIZE = bxy;   // 4
    const int BLKYSIZE = bxy;   // 4
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = gz;   // 1

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // 1 for loop 10 iterations
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer10_gemm_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_9_output, d_layer_10_bias, d_cuda_layer_10_weight, d_cuda_layer_10_output,
        ISm, OSm);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // copy result from device to host
    cudaMemcpy(cuda_layer_10_output, d_cuda_layer_10_output, (size10*sizeof(signed short)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // free the memory
    cudaFree(d_cuda_layer_9_output);
    cudaFree(d_layer_10_bias);
    cudaFree(d_cuda_layer_10_weight);
    cudaFree(d_cuda_layer_10_output);
    cudaCheckErrors("cudaFree fail");

    // checksum L10 = -666.000000
    // float sum = 0;
    // ofstream g("layer_10_par1.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum=0;
    //     for(int i=b*10;i<(b+1)*10;i++){
    //         sum += cuda_layer_10_output[i];
    //         g<<cuda_layer_10_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum<<endl;
    // }
    return milliseconds;
}

