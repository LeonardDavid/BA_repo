#include <iostream>

#include "cuda_net.h"
#include "netW.hpp"

using namespace std;

float predict_NeuralNet(unsigned char * const x, float * output) {
  // possibly not valid c++ code:
  // unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

  // add all kernel_time s
  float kernel_time = 0;
  kernel_time += layer1_conv(x, cuda_layer_1_output);
  float l1_kernel_time = kernel_time;
  kernel_time += layer2_maxpool(cuda_layer_1_output, cuda_layer_2_output);

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
 
  for(int b=0;b<BATCH_SIZE;b++){
    for (int h = 0; h < 14; h++) {
      for (int w = 0; w < 14; w++) {
          for (int c = 0; c < 64; c++) {
          if (cuda_layer_2_output[index4D(b,h,w,c,14,14,64)] > layer_3_threshold[c]) {
            layer_3_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_3_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      // cout<<endl;
        }
      }
    // cout<<endl;
  }

  // flatten layer_3_output into cuda_layer_3_output for further usage
  for(int i=0;i<14;i++){
    for(int j=0;j<14;j++){
      for(int b=0;b<BATCH_SIZE;b++){
        for(int k=0;k<64;k++){
          cuda_layer_3_output[index4D(b,i,j,k,14,14,64)] = layer_3_output[b][i][j][k];
        }
      }
    }
  }

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


  kernel_time += layer4_conv(cuda_layer_3_output, cuda_layer_4_output);
  kernel_time += layer5_maxpool(cuda_layer_4_output, cuda_layer_5_output);

  // layer6_step(cuda_layer_5_output, cuda_layer_6_output);
  /*
    Same as layer3_step
  */
  for(int b=0;b<BATCH_SIZE;b++){
    for (int h = 0; h < 7; h++) {
      for (int w = 0; w < 7; w++) {
        for (int c = 0; c < 64; c++) {
          if (cuda_layer_5_output[index4D(b,h,w,c,7,7,64)] > layer_6_threshold[c]) {
            layer_6_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_6_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
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
  
  kernel_time += layer8_gemm(layer_7_output, cuda_layer_8_output);

  // layer9_step(cuda_layer_8_output, cuda_layer_9_output);
  /*
    Same as layer9_step
  */
  for(int b=0;b<BATCH_SIZE;b++){
    for (int d = 0; d < 2048; d++) {
      if (cuda_layer_8_output[b*2048 + d] > layer_9_threshold[d]) {
        layer_9_output[b][d / 64] |= (1ULL << (63 - d % 64));
      } else {
        layer_9_output[b][d / 64] &= ~(1ULL << (63 - d % 64));
      }
    }
  }
  
  unsigned long long *cuda_layer_9_output = (unsigned long long *) layer_9_output;
  
  // worth it for 10 iterations? not really
  kernel_time += layer10_gemm(cuda_layer_9_output, cuda_layer_10_output);

  for(int b=0;b<BATCH_SIZE;b++){
    for (int i = 0; i < 10; i++) {
      output[b*10 + i] += cuda_layer_10_output[b*10 + i];
    }
  }

  return kernel_time;

}
