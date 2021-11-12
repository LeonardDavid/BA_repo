#include "netW.hpp"
#include "cuda_kernel.h"
#include <iostream>

using namespace std;

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

__device__ int index3D_cuda(const int x, const int y, const int z, const int sizey, const int sizez) {
  return x*sizey*sizez + y*sizez + z;
}

__device__ int index4D_cuda(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
  return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}

// Layer 1 - Convolution

__global__ void layer1_conv_kernel(unsigned char *d_cuda_layer_0_output, float *d_layer_1_bias, signed char *d_cuda_layer_1_weight, float *d_cuda_layer_1_output){

  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (h<28 && w<28){
      for (int m = 0; m < 64; m++) {
          d_cuda_layer_1_output[index3D_cuda(h,w,m,28,64)] = d_layer_1_bias[m];
      }
      for (int kH = 0; kH < 3; kH++) {
      int iH = h * 1 + kH - 1;
      if (iH >= 0 && iH < 28) {
          for (int kW = 0; kW < 3; kW++) {
          int iW = w * 1 + kW - 1;
          if (iW >= 0 && iW < 28) {
              for (int c = 0; c < 1; c++) {
                  for (int m = 0; m < 64; m++) {
                      d_cuda_layer_1_output[index3D_cuda(h,w,m,28,64)] += d_cuda_layer_1_weight[index4D_cuda(kH,kW,c,m,3,1,64)] * d_cuda_layer_0_output[index3D_cuda(iH,iW,c,28,1)];
                  }
              }
          }
          }
      }
      }
  }
}

float layer1_conv_cuda(unsigned char * const x, float * cuda_layer_1_output){

  // initialize layer_0_output where x is the input image
  unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

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
  cudaMalloc((void **) &d_cuda_layer_0_output, 784*sizeof(unsigned char)); // 784 = 28x28 dim of cuda_layer_0_output
  cudaMalloc((void **) &d_layer_1_bias, 64*sizeof(float)); // 64 = dim of layer_1_bias
  cudaMalloc((void **) &d_cuda_layer_1_weight, 576*sizeof(signed char)); // 576 = 3x3x1x64 dim of layer_1_weight
  cudaMalloc((void **) &d_cuda_layer_1_output, 50176*sizeof(float)); // 50176 = 28x28x64 dim of layer_1_output
  cudaCheckErrors("Failed to allocate device buffer");

  // copy input data from host on device
  cudaMemcpy(d_cuda_layer_0_output, cuda_layer_0_output, (784*sizeof(unsigned char)), cudaMemcpyHostToDevice);
  cudaMemcpy(d_layer_1_bias, layer_1_bias, (64*sizeof(float)), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cuda_layer_1_weight, cuda_layer_1_weight, (576*sizeof(signed char)), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA memcpy failure");

  // define thread and block sizes
  const int BLKXSIZE = 28;
  const int BLKYSIZE = 28;
  const int GRIDXSIZE = 1;
  const int GRIDYSIZE = 1;
  
  const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 for loops 28 iterations each
  const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE);

  // compute result - kernel call
  layer1_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_0_output, d_layer_1_bias, d_cuda_layer_1_weight, d_cuda_layer_1_output);
  cudaCheckErrors("Kernel launch failure");

  // copy result from device to host
  cudaMemcpy(cuda_layer_1_output, d_cuda_layer_1_output, (50176*sizeof(float)), cudaMemcpyDeviceToHost);
  cudaCheckErrors("CUDA memcpy failure");

  // synchronize threads
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA synchronize failure");

  // free the memory
  cudaFree(d_cuda_layer_0_output);
  cudaFree(d_layer_1_bias);
  cudaFree(d_cuda_layer_1_weight);
  cudaFree(d_cuda_layer_1_output);
  cudaCheckErrors("cudaFree fail");
  
  // checksum
  float sum = 0;
  // for (int i = 0; i < 50176; i++) {
  //     sum += cuda_layer_1_output[i];
  //     // cout<<cuda_layer_1_output[i]<<" ";   
  // }
  return sum;
}

// Layer 2 - Maxpool

__global__ void layer2_maxpool_kernel(float *d_cuda_layer_1_output, float *d_cuda_layer_2_output, float lowest){

  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;

  if(h<14 && w<14){
      for (int c = 0; c < 64; c++) {
          d_cuda_layer_2_output[index3D_cuda(h,w,c,14,64)] = lowest;
      }
      for (int kH = 0; kH < 2; kH++) {
          for (int kW = 0; kW < 2; kW++) {
              for (int c = 0; c < 64; c++) {
                  d_cuda_layer_2_output[index3D_cuda(h,w,c,14,64)] = fmax(d_cuda_layer_1_output[index3D_cuda((h * 2 + kH),(w * 2 + kW),c,28,64)], d_cuda_layer_2_output[index3D_cuda(h,w,c,14,64)]);
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
  const int BLKYSIZE = 14;
  const int GRIDXSIZE = 1;
  const int GRIDYSIZE = 1;

  const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 foor loops 14 iterations each
  const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE);

  // std library not allowed on device
  const float LOWEST = std::numeric_limits<float>::lowest();
      
  // compute result - kernel call
  layer2_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_1_output, d_cuda_layer_2_output, LOWEST);
  cudaCheckErrors("Kernel launch failure");

  // copy result from device to host
  cudaMemcpy(cuda_layer_2_output, d_cuda_layer_2_output, (12544*sizeof(float)), cudaMemcpyDeviceToHost);
  cudaCheckErrors("CUDA memcpy failure");

  // synchronize threads
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA synchronize failure");

  // free the memory
  cudaFree(d_cuda_layer_1_output);
  cudaFree(d_cuda_layer_2_output);
  cudaCheckErrors("cudaFree fail");

  // checksum
  float sum = 0;
  // for (int i = 0; i < 12544; i++) {
  //     sum += cuda_layer_2_output[i];
  //     // cout<<cuda_layer_2_output[i]<<" ";  
  // }

  // all elements (and sum) = 0 (answer is correct at the end) why?
  return sum;
}

// Layer 3 - Step
// TODO WORK IN PROGRESS

__global__ void layer3_step_kernel(float *d_cuda_layer_2_output, signed short *d_layer_3_threshold, unsigned long long *d_cuda_layer_3_output){

  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;

  if(h<14 && w<14 && c<64){
      if (d_cuda_layer_2_output[index3D_cuda(h,w,c,14,64)] > d_layer_3_threshold[c]) {
          // if(h==1)
          //     printf("%llu ",d_cuda_layer_3_output[index3D_cuda(h,w,c,14,64)]);
          d_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)] |= (1ULL << (63 - c % 64));
        } else {
          // if(h==1)
          //     printf("~%llu ",d_cuda_layer_3_output[index3D_cuda(h,w,c,14,64)]);
          d_cuda_layer_3_output[index3D_cuda(h,w,(c/64),14,1)] &= ~(1ULL << (63 - c % 64));
        }
  }
}

int layer3_step_cuda(float * cuda_layer_2_output, unsigned long long * cuda_layer_3_output){
  // flatten 3D -> 1D arrays
  // layer_3_output filled with 64 ULL values at each 14x14 points
  // for(int i=0;i<14;i++){
  //     for(int j=0;j<14;j++){
  //         for(int k=0;k<64;k++){
  //             cuda_layer_3_output[index3D(i,j,k,14,64)] = layer_3_output[i][j][k];
  //         }
  //     }
  // }

  // for (int h = 0; h < 14; h++) {
  //     for (int w = 0; w < 14; w++) {
  //         for (int c = 0; c < 64; c++) {
  //         cout<<cuda_layer_3_output[index3D(h,w,c,14,64)]<<" ";
  //         // if(h==0 && w==0)
  //         //   cout<<"before: "<<layer_3_output[h][w][c];

  //         // cuda_layer_3_output[index3D(h,w,c,14,1)] = layer_3_output[h][w][c];
         
  //         // if(h==0 && w==0)
  //         //   cout<<" after: "<<cuda_layer_3_output[index3D(h,w,c,14,64)]<<endl;
  //         // cout<<"("<<c<<" - "<<c/64<<")";
  //         }
  //         // cout<<endl;
  //     }
  // // cout<<endl;
  // }
  // cout<<endl;

  // cout<<endl<<"-------------------------"<<endl;
  // cout<<"layer_3_output[i]: ";
  // cout<<layer_3_output[0][0][0]<<" ";
  // cout<<layer_3_output[0][1][0]<<" ";
  // cout<<layer_3_output[0][2][0]<<" ";
  // cout<<layer_3_output[0][3][0]<<" ";
  // cout<<layer_3_output[0][4][0]<<" ";
  // cout<<endl;

  // cout<<"cuda_layer_3_output[i]: ";
  // for(int i=0;i<5;i++){
  // cout<<cuda_layer_3_output[i]<<" ";
  // }
  // cout<<endl;

  // cout<<"cuda_layer_3_output[index3D]: ";
  // cout<<cuda_layer_3_output[index3D(0,0,0,14,1)]<<" ";
  // cout<<cuda_layer_3_output[index3D(0,1,0,14,1)]<<" ";
  // cout<<cuda_layer_3_output[index3D(0,2,0,14,1)]<<" ";
  // cout<<cuda_layer_3_output[index3D(0,3,0,14,1)]<<" ";
  // cout<<cuda_layer_3_output[index3D(0,4,0,14,1)]<<" ";
  // cout<<endl;

  // prepare for kernel call
  // declare storage on device
  float *d_cuda_layer_2_output;   // storage on device for cuda_layer_2_output
  signed short *d_layer_3_threshold; // storage on device for layer_3_threshold
  unsigned long long *d_cuda_layer_3_output; // RESULT storage on device for cuda_layer_3_output
  
  // allocate GPU device buffers
  cudaMalloc((void **) &d_cuda_layer_2_output, 12544*sizeof(float)); // 12544 = 14x14x64 dim of cuda_layer_2_output
  cudaMalloc((void **) &d_layer_3_threshold, 64*sizeof(signed short)); // 64 = dim of layer_3_threshold
  cudaMalloc((void **) &d_cuda_layer_3_output, 12544*sizeof(unsigned long long)); // 196 = 14x14x[1x64] dim of cuda_layer_3_output [ULL]
  cudaCheckErrors("Failed to allocate device buffer");

  // copy input data from host on device
  cudaMemcpy(d_cuda_layer_2_output, cuda_layer_2_output, (12544*sizeof(float)), cudaMemcpyHostToDevice);
  cudaMemcpy(d_layer_3_threshold, layer_3_threshold, (64*sizeof(signed short)), cudaMemcpyHostToDevice);

  cudaMemcpy(d_cuda_layer_3_output, cuda_layer_3_output, (12544*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA memcpy failure");

  // define thread and block sizes
  const int BLKXSIZE = 14;
  const int BLKYSIZE = 14;
  const int BLKZSIZE = 4;
  const int GRIDXSIZE = 4;
  const int GRIDYSIZE = 4;
  const int GRIDZSIZE = 1;

  const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE, BLKZSIZE); // the 3 for loops 14x14x64 iterations
  const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

  // compute result - kernel call
  layer3_step_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_2_output, d_layer_3_threshold, d_cuda_layer_3_output);
  cudaCheckErrors("Kernel launch failure");

  // copy result from device to host
  cudaMemcpy(cuda_layer_3_output, d_cuda_layer_3_output, (12544*sizeof(unsigned long long)), cudaMemcpyDeviceToHost);
  cudaCheckErrors("CUDA memcpy failure");

  // synchronize threads
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA synchronize failure");

  // free the memory
  cudaFree(d_cuda_layer_2_output);
  cudaFree(d_layer_3_threshold);
  cudaFree(d_cuda_layer_3_output);
  cudaCheckErrors("cudaFree fail");

  // checksum
  int sum = 0;
  for (int i = 0; i < 12544; i++) {
      sum += cuda_layer_3_output[i]%10;
      // cout<<cuda_layer_3_output[i]<<" ";  
  }

  return sum;
}

// END WORK IN PROGRESS

// Layer 4 - Convolution

__global__ void layer4_conv_kernel(unsigned long long *d_cuda_layer_3_output, float *d_layer_4_bias, unsigned long long *d_cuda_layer_4_weight, signed short *d_cuda_layer_4_output){

  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;

  if(h<14 && w<14){
      for (int m = 0; m < 64; m++) {
          // original code always equals to 0, because the biases are float and output is signed short (?)
          d_cuda_layer_4_output[index3D_cuda(h,w,m,14,64)] = d_layer_4_bias[m]; // = 0;
      }
      for (int kH = 0; kH < 3; kH++) {
      int iH = h * 1 + kH - 1;
      if (iH >= 0 && iH < 14) {
          for (int kW = 0; kW < 3; kW++) {
          int iW = w * 1 + kW - 1;
          if (iW >= 0 && iW < 14) {
              for (int m = 0; m < 64; m++) {
                  for (int c = 0; c < 1; c++) {
                      d_cuda_layer_4_output[index3D_cuda(h,w,m,14,64)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_4_weight[index4D_cuda(kH,kW,m,c,3,64,1)] ^ d_cuda_layer_3_output[index3D_cuda(iH,iW,c,14,64)])) - 64;
                     }
              }
          }
          }
      }
      }
  }
}

int layer4_conv_cuda(unsigned long long * cuda_layer_3_output, signed short * cuda_layer_4_output){
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
  const int BLKYSIZE = 14;
  const int GRIDXSIZE = 1;
  const int GRIDYSIZE = 1;

  const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 for loops 14 iterations each
  const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE);

  // compute result - kernel call
  layer4_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_3_output, d_layer_4_bias, d_cuda_layer_4_weight, d_cuda_layer_4_output);
  cudaCheckErrors("Kernel launch failure");

  // copy result from device to host
  cudaMemcpy(cuda_layer_4_output, d_cuda_layer_4_output, (12544*sizeof(signed short)), cudaMemcpyDeviceToHost);
  cudaCheckErrors("CUDA memcpy failure");

  // synchronize threads
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA synchronize failure");

  // free the memory
  cudaFree(d_cuda_layer_3_output);
  cudaFree(d_layer_4_bias);
  cudaFree(d_cuda_layer_4_weight);
  cudaFree(d_cuda_layer_4_output);
  cudaCheckErrors("cudaFree fail");

  // checksum
  int sum = 0;
  // for (int i = 0; i < 12544; i++) {
  //     sum += cuda_layer_4_output[i];
  //     // cout<<cuda_layer_4_output[i]<<" ";   
  // }
  return sum;
}

// Layer 5 - Maxpool
__global__ void layer5_maxpool_kernel(signed short * d_cuda_layer_4_output, signed short * d_cuda_layer_5_output, signed short lowest){

  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;

  if(h<7 && w<7){
      for (int c = 0; c < 64; c++) {
          d_cuda_layer_5_output[index3D_cuda(h,w,c,7,64)] = lowest;
      }
      for (int kH = 0; kH < 2; kH++) {
          for (int kW = 0; kW < 2; kW++) {
              for (int c = 0; c < 64; c++) {
                  d_cuda_layer_5_output[index3D_cuda(h,w,c,7,64)] = 
                  (d_cuda_layer_4_output[index3D_cuda((h * 2 + kH),(w * 2 + kW),c,14,64)] >= d_cuda_layer_5_output[index3D_cuda(h,w,c,7,64)]) ? 
                  d_cuda_layer_4_output[index3D_cuda((h * 2 + kH),(w * 2 + kW),c,14,64)] : d_cuda_layer_5_output[index3D_cuda(h,w,c,7,64)];
              }
          }
      }
  }
}

int layer5_maxpool_cuda(signed short * cuda_layer_4_output, signed short * cuda_layer_5_output){
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
  const int BLKYSIZE = 7;
  const int GRIDXSIZE = 1;
  const int GRIDYSIZE = 1;

  const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 foor loops 7 iterations each
  const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE);

  // std library not allowed on device
  const signed short LOWEST = std::numeric_limits<signed short>::lowest();

  // compute result - kernel call
  layer5_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_4_output, d_cuda_layer_5_output, LOWEST);
  cudaCheckErrors("Kernel launch failure");

  // copy result from device to host
  cudaMemcpy(cuda_layer_5_output, d_cuda_layer_5_output, (3136*sizeof(signed short)), cudaMemcpyDeviceToHost);
  cudaCheckErrors("CUDA memcpy failure");

  // synchronize threads
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA synchronize failure");

  // free the memory
  cudaFree(d_cuda_layer_4_output);
  cudaFree(d_cuda_layer_5_output);
  cudaCheckErrors("cudaFree fail");

  // checksum
  int sum = 0;
  // for (int i = 0; i < 3136; i++) {
  //     sum += cuda_layer_5_output[i];
  //     // cout<<cuda_layer_2_output[i]<<" ";  
  // }
  return sum;
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

int layer8_gemm_cuda(unsigned long long * cuda_layer_7_output, signed short * cuda_layer_8_output){
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

  // compute result - kernel call
  layer8_gemm_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_7_output, d_layer_8_bias, d_cuda_layer_8_weight, d_cuda_layer_8_output);
  cudaCheckErrors("Kernel launch failure");

  // copy result from device to host
  cudaMemcpy(cuda_layer_8_output, d_cuda_layer_8_output, (2048*sizeof(signed short)), cudaMemcpyDeviceToHost);
  cudaCheckErrors("CUDA memcpy failure");

  // synchronize threads
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA synchronize failure");

  // free the memory
  cudaFree(d_cuda_layer_7_output);
  cudaFree(d_layer_8_bias);
  cudaFree(d_cuda_layer_8_weight);
  cudaFree(d_cuda_layer_8_output);
  cudaCheckErrors("cudaFree fail");

  // checksum
  int sum = 0;
  // for (int i = 0; i < 2048; i++) {
  //     sum += cuda_layer_8_output[i];
  //     // cout<<cuda_layer_4_output[i]<<" ";   
  // }
  return sum;
}

// Layer 10 - Gemm
__global__ void layer10_gemm_kernel(unsigned long long *d_cuda_layer_9_output, float *d_layer_10_bias, unsigned long long *d_cuda_layer_10_weight, signed short *d_cuda_layer_10_output){

  int d = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(d < 10){
    // original code always equals to 0, because the biases are float and output is signed short (?)
    d_cuda_layer_10_output[d] = d_layer_10_bias[d];
    for (int i = 0; i < 32; i++) {
      d_cuda_layer_10_output[d] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_10_weight[d*32+i] ^ d_cuda_layer_9_output[i])) - 64;
    }
  }
}

int layer10_gemm_cuda(unsigned long long * cuda_layer_9_output, signed short * cuda_layer_10_output){
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

  // compute result - kernel call
  layer10_gemm_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_9_output, d_layer_10_bias, d_cuda_layer_10_weight, d_cuda_layer_10_output);
  cudaCheckErrors("Kernel launch failure");

  // copy result from device to host
  cudaMemcpy(cuda_layer_10_output, d_cuda_layer_10_output, (10*sizeof(signed short)), cudaMemcpyDeviceToHost);
  cudaCheckErrors("CUDA memcpy failure");

  // synchronize threads
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA synchronize failure");

  // free the memory
  cudaFree(d_cuda_layer_9_output);
  cudaFree(d_layer_10_bias);
  cudaFree(d_cuda_layer_10_weight);
  cudaFree(d_cuda_layer_10_output);
  cudaCheckErrors("cudaFree fail");

  // checksum
  int sum = 0;
  // for (int i = 0; i < 10; i++) {
  //     sum += cuda_layer_10_output[i];
  //     // cout<<cuda_layer_4_output[i]<<" ";   
  // }
  return sum;
}

int predict_NeuralNet_Cuda(unsigned char * const x, float * pred) {

  layer1_conv_cuda(x, cuda_layer_1_output);
  layer2_maxpool_cuda(cuda_layer_1_output, cuda_layer_2_output);

  // unsigned long long *cuda_layer_3_output = (unsigned long long *) layer_3_output;
  // layer3_step_cuda(cuda_layer_2_output, cuda_layer_3_output);
    /* Layer 3 does not work because:
      - if run without the line 'layer_3_output[h][w][c / 64] |= (1ULL << (63 - c % 64));' every element will be 0
      - for ABSOLUTELY NO REASON if the line is present AFTER (!!) the cout/calculation with l30, the correct answer will be calculated
    */
    for (int h = 0; h < 14; h++) {
      for (int w = 0; w < 14; w++) {
        for (int c = 0; c < 64; c++) {
          if (cuda_layer_2_output[index3D(h,w,c,14,64)] > layer_3_threshold[c]) {
            // if(h==1)
            //   cout<<layer_3_output[h][w][c]<<" ";
            layer_3_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            // if(h==1)
            //   cout<<"~"<<layer_3_output[h][w][c]<<" ";
            layer_3_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
        // cout<<endl;
      }
      // cout<<endl;
    }
    // flatten layer_3_output into cuda_layer_3_output for further usage
    for(int i=0;i<14;i++){
      for(int j=0;j<14;j++){
        for(int k=0;k<64;k++){
          cuda_layer_3_output[index3D(i,j,k,14,64)] = layer_3_output[i][j][k];
        }
      }
    }
    // unsigned long long *cuda_layer_3_output = (unsigned long long *) layer_3_output;

  layer4_conv_cuda(cuda_layer_3_output, cuda_layer_4_output);
  layer5_maxpool_cuda(cuda_layer_4_output, cuda_layer_5_output);

  // layer6_step_cuda(cuda_layer_5_output, cuda_layer_6_output);
  /*
    Same as layer3_step_cuda
  */
    for (int h = 0; h < 7; h++) {
      for (int w = 0; w < 7; w++) {
        for (int c = 0; c < 64; c++) {
          if (cuda_layer_5_output[index3D(h,w,c,7,64)] > layer_6_threshold[c]) {
            layer_6_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_6_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }
    // flatten layer_6_output into cuda_layer_6_output for further usage
    // for(int i=0;i<7;i++){
    //   for(int j=0;j<7;j++){
    //     for(int k=0;k<64;k++){
    //       cuda_layer_6_output[index3D(i,j,k,7,64)] = layer_6_output[i][j][k];
    //     }
    //   }
    // }

  // Layer 7 is flattening layer -> cuda_layer_6_output skipped
  unsigned long long *layer_7_output = (unsigned long long *) layer_6_output; // size = 49
  
  layer8_gemm_cuda(layer_7_output, cuda_layer_8_output);

  // layer9_step_cuda(cuda_layer_8_output, cuda_layer_9_output);
  /*
    Same as layer6_step
  */
    for (int d = 0; d < 2048; d++) {
      if (cuda_layer_8_output[d] > layer_9_threshold[d]) {
        layer_9_output[d / 64] |= (1ULL << (63 - d % 64));
      } else {
        layer_9_output[d / 64] &= ~(1ULL << (63 - d % 64));
      }
    }
    unsigned long long *cuda_layer_9_output = (unsigned long long *) layer_9_output;
  
  // worth it for 10 iterations? not really
  layer10_gemm_cuda(cuda_layer_9_output, cuda_layer_10_output);
  
  for (int i = 0; i < 10; i++) {
    pred[i] += cuda_layer_10_output[i];
  }

  return 0;
}
