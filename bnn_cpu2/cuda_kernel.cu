#include <stdio.h>
#include <iostream>
#include "cuda_kernel.h"

__global__ void mykernel(int c, float* d_A, float* d_C, int N){

    // printf("Hello from mykernel @ %d\n", c);

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        d_C[i] = 2 * d_A[i];
}

int matadd(int c){

    int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    // float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i=0;i<N;i++){
        h_A[i] = c;
        // h_B[i] = 2.0f;
    }

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    // float* d_B;
    // cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    mykernel<<<blocksPerGrid, threadsPerBlock>>>(c, d_A, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // for(int i=0;i<N;i++){
    //     std::cout<<"h_A["<<i<<"]"<<": "<<h_A[i]<<std::endl;
    // }

    // for(int i=0;i<N;i++){
    //     std::cout<<"h_B["<<i<<"]"<<": "<<h_B[i]<<std::endl;
    // }

    // for(int i=0;i<N;i++){
    //     std::cout<<"h_C["<<i<<"]"<<": "<<h_C[i]<<std::endl;
    // }

    // std::cout<<"h_C[0] @ "<<c<<": "<<h_C[0]<<std::endl;

    // Free device memory
    cudaFree(d_A);
    // cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    free(h_A);
    // free(h_B);

    return h_C[0];
}