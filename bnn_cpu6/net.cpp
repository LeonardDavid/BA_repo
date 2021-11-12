#include "netW.hpp"
#include "cuda_net.h"
#include <iostream>

int index(const int x, const int y, const int z) {
     return x * DATAYSIZE * DATAZSIZE + y * DATAZSIZE + z;
}

  void predict_NeuralNet(unsigned char * const x, float * pred, int c) {
		unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

      // Layer 1: Conv
      for (int h = 0; h < 28; h++) {
        for (int w = 0; w < 28; w++) {
          for (int m = 0; m < 64; m++) {
            layer_1_output[h][w][m] = layer_1_bias[m];
          }
          for (int kH = 0; kH < 3; kH++) {
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 28) {
              for (int kW = 0; kW < 3; kW++) {
                int iW = w * 1 + kW - 1;
                if (iW >= 0 && iW < 28) {
                  for (int c = 0; c < 1; c++) {
                    for (int m = 0; m < 64; m++) {
                      layer_1_output[h][w][m] += layer_1_weight[kH][kW][c][m] * layer_0_output[iH][iW][c];
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Layer 2: MaxPool
      for (int h = 0; h < 14; h++) {
        for (int w = 0; w < 14; w++) {
          for (int c = 0; c < 64; c++) {
            layer_2_output[h][w][c] = std::numeric_limits<float>::lowest();
          }
          for (int kH = 0; kH < 2; kH++) {
            for (int kW = 0; kW < 2; kW++) {
              for (int c = 0; c < 64; c++) {
                layer_2_output[h][w][c] = std::max(layer_1_output[h * 2 + kH][w * 2 + kW][c], layer_2_output[h][w][c]);
              }
            }
          }
        }
      }

      // Layer 3: Step
      for (int h = 0; h < 14; h++) {
        for (int w = 0; w < 14; w++) {
          for (int c = 0; c < 64; c++) {
            if (layer_2_output[h][w][c] >layer_3_threshold[c]) {
              layer_3_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
            } else {
              layer_3_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
            }
          }
        }
      }

      // Layer 4: Conv
      for (int h = 0; h < 14; h++) {
        for (int w = 0; w < 14; w++) {
          for (int m = 0; m < 64; m++) {
            layer_4_output[h][w][m] = layer_4_bias[m];
          }
          for (int kH = 0; kH < 3; kH++) {
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 14) {
              for (int kW = 0; kW < 3; kW++) {
                int iW = w * 1 + kW - 1;
                if (iW >= 0 && iW < 14) {
                  for (int m = 0; m < 64; m++) {
                    for (int c = 0; c < 1; c++) {
                      layer_4_output[h][w][m] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_4_weight[kH][kW][m][c] ^ layer_3_output[iH][iW][c])) - 64;
                    }
                  }
                }
              }
            }
          }
        }
      }

      // CUDA kernel call
      // 
      // flatten 3D layer_4_output -> 1D xb
      // CUDA kernel call -> x10 each element
      // deflatten 1D xb -> 3D layer_4_output
      // 


      // std::cout<<"----------begin----------"<<std::endl;

      // DO NOT DECLARE LIKE THIS (STACK SMASHING ERROR): short xb[DATAMUL];

      // * 2 because of error: double free or corruption (!prev)
      signed short* xb = (signed short *) malloc(sizeof(signed short) * (DATAMUL * 2));

      //populate xa/flatten xa
      // std::cout<<"flatten l_4_0 to xb:"<<std::endl<<std::endl;
      for(int i=0; i<DATAXSIZE; i++){
        for(int j=0; j<DATAYSIZE; j++){
          for(int k=0; k<DATAZSIZE; k++){
            xb[index(i,j,k)] = layer_4_output[i][j][k];
          }
        }
      }

      //check xa cout
      // std::cout<<"check l_4_o:"<<std::endl;
      // for(int i=0; i<DATAXSIZE; i++){
      //   for(int j=0; j<DATAYSIZE; j++){
      //     for(int k=0; k<DATAZSIZE; k++){
      //       std::cout<<layer_4_output[i][j][k]<<" ";
      //     }
      //     std::cout<<std::endl;
      //   }
      //   std::cout<<std::endl;
      // }

      // std::cout<<"cuda call:"<<std::endl;
      cuda_k(xb);
      // std::cout<<std::endl;

      // deflatten xb to xc
      // std::cout<<"deflatten xb to l_4_0:"<<std::endl;
      for(int i=0; i<DATAXSIZE; i++){
        for(int j=0; j<DATAYSIZE; j++){
          for(int k=0; k<DATAZSIZE; k++){
            layer_4_output[i][j][k] = xb[index(i,j,k)];
          }
        }
      }

      // std::cout<<"check xb:"<<std::endl;
      // for(int i=0;i<DATAMUL;i++){
      //   std::cout<<xb[i]<<" ";
      // }
      // std::cout<<std::endl;

      free(xb);

      // std::cout<<"-----------end-----------"<<std::endl<<std::endl;

      // Layer 5: MaxPool
      for (int h = 0; h < 7; h++) {
        for (int w = 0; w < 7; w++) {
          for (int c = 0; c < 64; c++) {
            layer_5_output[h][w][c] = std::numeric_limits<signed short>::lowest();
          }
          for (int kH = 0; kH < 2; kH++) {
            for (int kW = 0; kW < 2; kW++) {
              for (int c = 0; c < 64; c++) {
                layer_5_output[h][w][c] = std::max(layer_4_output[h * 2 + kH][w * 2 + kW][c], layer_5_output[h][w][c]);
              }
            }
          }
        }
      }

      // Layer 6: Step
      for (int h = 0; h < 7; h++) {
        for (int w = 0; w < 7; w++) {
          for (int c = 0; c < 64; c++) {
            if (layer_5_output[h][w][c] >layer_6_threshold[c]) {
              layer_6_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
            } else {
              layer_6_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
            }
          }
        }
      }

      // output layer 6 results
      // for (int h = 6; h < 7; h++) {
      //   for (int w = 6; w < 7; w++) {
      //     std::cout << layer_6_output[h][w][0] << std::endl;
      //   }
      // }

      // Layer 7: Flatten
      unsigned long long *layer_7_output = (unsigned long long *) layer_6_output;

      // Layer 8: Gemm
      for (int d = 0; d < 2048; d++) {
        layer_8_output[d] = layer_8_bias[d];
      }
      for (int d = 0; d < 2048; d++) {
        for (int i = 0; i < 49; i++) {
          layer_8_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_8_weight[d][i] ^ layer_7_output[i])) - 64;
        }
      }

      // Layer 9: Step
      for (int d = 0; d < 2048; d++) {
        if (layer_8_output[d] >layer_9_threshold[d]) {
          layer_9_output[d / 64] |= (1ULL << (63 - d % 64));
        } else {
          layer_9_output[d / 64] &= ~(1ULL << (63 - d % 64));
        }
      }

      // Layer 10: Gemm
      for (int d = 0; d < 10; d++) {
        layer_10_output[d] = layer_10_bias[d];
      }
      for (int d = 0; d < 10; d++) {
        for (int i = 0; i < 32; i++) {
          layer_10_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_10_weight[d][i] ^ layer_9_output[i])) - 64;
        }
      }

    for (int i = 0; i < 10; i++) {
      pred[i] += layer_10_output[i];
    }

  }
