#include <iostream>
#include <chrono>
#include <tuple>

#include "cuda_net.h"
#include "netW.hpp"

using namespace std;

float predict_NeuralNet(unsigned char * const x, float * output) { // std::tuple<float,float,float,float,float> 
  
  // counter that keeps track of current layer number inside cuda functions and kernels
  size_t nr_layer = 0;
  // possibly not valid c++ code:
  // unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;
  nr_layer++;

  // add all kernel_time s
  float kernel_time = 0;
  kernel_time += layer1_conv(x, cuda_layer_1_output, nr_layer);
  nr_layer++;

  float l1_kernel_time = kernel_time;
  kernel_time += layer2_maxpool(cuda_layer_1_output, cuda_layer_2_output, nr_layer);
  nr_layer++;
  

  /*
      to run without the outputs from line 58-126: 
        - uncomment lines 19-50
        - comment lines 58-126
  */
  
  // layer3_step(cuda_layer_2_output, cuda_layer_3_output);
  
  /* Layer 3 does not work because:
    - if run without the line 'layer_3_output[h][w][c / 64] |= (1ULL << (63 - c % 64));' every element will be 0
    - for ABSOLUTELY NO REASON if the line is present AFTER (!!) the cout/calculation with l30, the correct answer will be calculated
  */

  // declare constants for current layer
  size_t OSh = output_shape[nr_layer][1];      // 28
  size_t OSw = output_shape[nr_layer][2];      // 28
  size_t OSm = output_shape[nr_layer][0];      // 64
  size_t offset = std::min(int(OSm), BINARY_WORD_SIZE);  // 64

  // auto start = std::chrono::high_resolution_clock::now();
  for(int b=0; b < BATCH_SIZE; b++){
    for (int h = 0; h < OSh; h++) {
      for (int w = 0; w < OSw; w++) {
        for (int c = 0; c < OSm; c++) {
          if (cuda_layer_2_output[index4D(b,h,w,c,OSh,OSw,OSm)] > layer_3_threshold[c]) {
            layer_3_output[b][h][w][c / offset] |= (1ULL << ((BINARY_WORD_SIZE-1) - c % offset));
          } else {
            layer_3_output[b][h][w][c / offset] &= ~(1ULL << ((BINARY_WORD_SIZE-1) - c % offset));
          }
        }
      // cout<<endl;
      }
    }
    // cout<<endl;
  }

  // flatten layer_3_output into cuda_layer_3_output for further usage
  for (int h = 0; h < OSh; h++) {
    for (int w = 0; w < OSw; w++) {
      for(int b=0; b < BATCH_SIZE; b++){
        for (int c = 0; c < OSm; c++) {
          cuda_layer_3_output[index4D(b,h,w,c,OSh,OSw,OSm)] = layer_3_output[b][h][w][c];
        }
      }
    }
  }
  nr_layer++;
  // auto end = std::chrono::high_resolution_clock::now();
  // auto l3_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

  // the method below for flattening does not lead to the correct result
  // unsigned long long *cuda_layer_3_output = (unsigned long long *) layer_3_output;

  /*
      to run the simple "debug" from line 58-126:
        - uncomment lines 58-126
        - comment lines 19-50
  */

  // cout<<"--------------------------------begin--------------------------------"<<endl;
  // cout<<endl;
  // cout<<"--------------------------------begin-CPU----------------------------"<<endl;

  // int sum1,sum2;
  // unsigned long long *cuda_layer_3_output = (unsigned long long *) layer_3_output;

  // for (int h = 0; h < 14; h++) {
  //   for (int w = 0; w < 14; w++) {
  //     for (int c = 0; c < 64; c++) {
  //       // printf("1 ");
  //       if (cuda_layer_2_output[index3D(h,w,c,14,64)] > layer_3_threshold[c]) {
  //         // printf("%d ", layer_3_threshold[c]);
  //         // cout<<"("<<layer_3_output[h][w][c/64]<<" - ";
  //         layer_3_output[h][w][c/64] |= (1ULL << (63 - c % 64));
  //         // cout<<layer_3_output[h][w][c/64]<<") ";
  //       } else {
  //         // printf("%d ", layer_3_threshold[c]);
  //         // cout<<"("<<layer_3_output[h][w][c/64]<<" - ";
  //         layer_3_output[h][w][c/64] &= ~(1ULL << (63 - c % 64));
  //         // cout<<layer_3_output[h][w][c/64]<<") ";
  //       }
  //     }
  //     // cout<<endl;
  //   }
  //   // cout<<endl;
  // }

  // // cout<<"layer3_output[14][14]: "<<endl;
  // // for (int h = 0; h < 14; h++) {
  // //   for (int w = 0; w < 14; w++) {
  // //       cout<<layer_3_output[h][w]<<" ";
  // //   }
  // //   cout<<endl;
  // // }
  // // cout<<endl;

  // sum1 = 0;
  // // summation of ULL values leads to overflow -> sum up only the last digit
  // for (int h = 0; h < 14; h++) {
  //   for (int w = 0; w < 14; w++) {
  //     for (int m = 0; m < 64; m++) {
  //       sum1 += layer_3_output[h][w][m]%10;
  //       // cout<<layer_3_output[h][w][m]<<" ";
  //     }
  //   }
  //   // cout<<endl;
  // }
  // cout<<endl;
  // // cout<<endl<<sum1<<endl;

  // cout<<"---------------------------------end-CPU-----------------------------"<<endl;
  // cout<<endl;
  // cout<<"--------------------------------begin-GPU----------------------------"<<endl;
  // sum2 = 0;
  // sum2 = layer3_step(cuda_layer_2_output, cuda_layer_3_output);
  // // cout<<endl<<sum2<<endl;
  // cout<<endl;
  // cout<<"---------------------------------end-GPU-----------------------------"<<endl;

  // if(sum1!=sum2){
  //   cout<<"FAIL"<<endl;
  // }else{
  //   cout<<"PASS"<<endl;
  // }
  // printf("cpp: %d - cuda: %d\n",sum1,sum2);

  // cout<<"---------------------------------end---------------------------------"<<endl;
  // cout<<endl<<endl;


  kernel_time += layer4_conv(cuda_layer_3_output, cuda_layer_4_output, nr_layer);
  nr_layer++;

  kernel_time += layer5_maxpool(cuda_layer_4_output, cuda_layer_5_output, nr_layer);
  nr_layer++;

  // layer6_step(cuda_layer_5_output, cuda_layer_6_output);
  /*
    Same as layer3_step
  */

  // declare constants for current layer
  OSh = output_shape[nr_layer][1];      // 7
  OSw = output_shape[nr_layer][2];      // 7
  OSm = output_shape[nr_layer][0];      // 64
  offset = std::min(int(OSm), BINARY_WORD_SIZE);  // 64

  // start = std::chrono::high_resolution_clock::now();
  for(int b=0; b < BATCH_SIZE; b++){
    for (int h = 0; h < OSh; h++) {
      for (int w = 0; w < OSw; w++) {
        for (int c = 0; c < OSm; c++) {
          if (cuda_layer_5_output[index4D(b,h,w,c,OSh,OSw,OSm)] > layer_6_threshold[c]) {
            layer_6_output[b][h][w][c / offset] |= (1ULL << ((BINARY_WORD_SIZE-1) - c % offset));
          } else {
            layer_6_output[b][h][w][c / offset] &= ~(1ULL << ((BINARY_WORD_SIZE-1) - c % offset));
          }
        }
      }
    }
  }
  // end = std::chrono::high_resolution_clock::now();
  // auto l6_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  nr_layer++;

  // Layer 7 is flattening layer -> cuda_layer_6_output skipped
  unsigned long long *layer_7_output = (unsigned long long *) layer_6_output; // size = 3136 = 64*7*7
  nr_layer++;
  
  kernel_time += layer8_gemm(layer_7_output, cuda_layer_8_output, nr_layer);
  nr_layer++;

  // layer9_step(cuda_layer_8_output, cuda_layer_9_output);
  /*
    Same as layer9_step
  */

  // declare constants for current layer
  OSm = output_shape[nr_layer][0];      // 2048
  offset = std::min(int(OSm), BINARY_WORD_SIZE);  // 64

  // start = std::chrono::high_resolution_clock::now();
  for(int b=0; b < BATCH_SIZE; b++){
    for (int d = 0; d < OSm; d++) {
      if (cuda_layer_8_output[b*OSm + d] > layer_9_threshold[d]) {
        layer_9_output[b][d / offset] |= (1ULL << ((BINARY_WORD_SIZE-1) - d % offset));
      } else {
        layer_9_output[b][d / offset] &= ~(1ULL << ((BINARY_WORD_SIZE-1) - d % offset));
      }
    }
  }
  // end = std::chrono::high_resolution_clock::now();
  // auto l9_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  unsigned long long *cuda_layer_9_output = (unsigned long long *) layer_9_output;
  nr_layer++;

  // worth it for 10 iterations? not really
  kernel_time += layer10_gemm(cuda_layer_9_output, cuda_layer_10_output, nr_layer);
  nr_layer++;

  for(int b=0;b<BATCH_SIZE;b++){
    for (int i = 0; i < 10; i++) {
      output[b*10 + i] += cuda_layer_10_output[b*10 + i];
    }
  }

  // l3_time+=l6_time + l9_time;

  return kernel_time;

}
