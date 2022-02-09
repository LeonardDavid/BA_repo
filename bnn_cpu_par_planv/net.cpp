#include <iostream>
#include <chrono>

#include "cuda_net.h"
#include "netW.hpp"

using namespace std;

float predict_NeuralNet(unsigned char * const x, float * output) {

  // add all kernel_time s
  float kernel_time = 0;

  /* Layer 1 CPU */

  // initialize layer_0_output where x is the input image
  unsigned char (*layer_0_output)[BATCH_SIZE][28][1] = (unsigned char (*)[BATCH_SIZE][28][1]) x;

  // flatten layer_0_output
  unsigned char *cuda_layer_0_output = (unsigned char *) layer_0_output;

  // Layer 1: Conv @ cpp.NHWC {% else %} /{% if pads == [0, 0, 0, 0] %}
  for(int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 28; h++) {
      for (int w = 0; w < 28; w++) {
        for (int m = 0; m < 64; m++) {
          cuda_layer_1_output[index4D(b,h,w,m,28,28,64)] = layer_1_bias[m];
        }
        for (int kH = 0; kH < 3; kH++) {
          int iH = h * 1 + kH - 1;
          if (iH >= 0 && iH < 28) {
            for (int kW = 0; kW < 3; kW++) {
              int iW = w * 1 + kW - 1;
              if (iW >= 0 && iW < 28) {
                for (int c = 0; c < 1; c++) {
                  for (int m = 0; m < 64; m++) {
                    cuda_layer_1_output[index4D(b,h,w,m,28,28,64)] += layer_1_weight[kH][kW][c][m] * cuda_layer_0_output[index4D(b,iH,iW,c,28,28,1)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /* Layer 2 CPU */

  // Layer 2: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
  for(int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 14; h++) {
      for (int w = 0; w < 14; w++) {
        for (int c = 0; c < 64; c++) {
          cuda_layer_2_output[index4D(b,h,w,c,14,14,64)] = std::numeric_limits<float>::lowest();
        }
        for (int kH = 0; kH < 2; kH++) {
          for (int kW = 0; kW < 2; kW++) {
            for (int c = 0; c < 64; c++) {
              cuda_layer_2_output[index4D(b,h,w,c,14,14,64)] = std::max(cuda_layer_1_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,28,28,64)], cuda_layer_2_output[index4D(b,h,w,c,14,14,64)]);
            }
          }
        }
      }
    }
  }

  /* Layer 3 CPU */
  
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
      }
    }
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

  /* Layer 4 GPU */
  
  kernel_time += layer4_conv(cuda_layer_3_output, cuda_layer_4_output);

  /* Layer 5 CPU */
  
  // Layer 5: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
  for(int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 7; h++) {
      for (int w = 0; w < 7; w++) {
        for (int c = 0; c < 64; c++) {
          cuda_layer_5_output[index4D(b,h,w,c,7,7,64)] = std::numeric_limits<signed short>::lowest();
        }
        for (int kH = 0; kH < 2; kH++) {
          for (int kW = 0; kW < 2; kW++) {
            for (int c = 0; c < 64; c++) {
              cuda_layer_5_output[index4D(b,h,w,c,7,7,64)] = std::max(cuda_layer_4_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,14,14,64)], cuda_layer_5_output[index4D(b,h,w,c,7,7,64)]);
            }
          }
        }
      }
    }
  }

  /* Layer 6 CPU */
  
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

  // Layer 7 is flattening layer -> cuda_layer_6_output skipped
  unsigned long long *layer_7_output = (unsigned long long *) layer_6_output; // size = 49
  
  /* Layer 8 GPU */
  
  kernel_time += layer8_gemm(layer_7_output, cuda_layer_8_output);
  
  /* Layer 9 CPU */
  
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
  
  /* Layer 10 CPU */
  
  // Layer 10: Gemm @ cpp.binary
  for(int b = 0; b < BATCH_SIZE; b++){
    for (int d = 0; d < 10; d++) {
      cuda_layer_10_output[b*10 + d] = layer_10_bias[d];
    }
    for (int d = 0; d < 10; d++) {
      for (int i = 0; i < 32; i++) {
        cuda_layer_10_output[b*10 + d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_10_weight[d][i] ^ cuda_layer_9_output[i])) - 64;
      }
    }
  }

  for(int b=0;b<BATCH_SIZE;b++){
    for (int i = 0; i < 10; i++) {
      output[b*10 + i] += cuda_layer_10_output[b*10 + i];
    }
  }

  return kernel_time;
}
