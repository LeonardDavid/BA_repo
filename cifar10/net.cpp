#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <tuple>

#include "cuda_net.h"
#include "net.hpp"
#include "netW.hpp"

using namespace std;

float predict_NeuralNet(unsigned char x[][32][32][3], float * pred) {
  //unsigned char (*layer_0_output)[32][3] = (unsigned char (*)[32][3]) x;

    float kernel_time = 0;

    // Layer 1: Conv @ cpp.NHWC {% else %} /{% if pads == [0, 0, 0, 0] %}
    // for (int b = 0; b < BATCH_SIZE; b++){
    //   for (int h = 0; h < 32; h++) {
    //     for (int w = 0; w < 32; w++) {
    //       for (int m = 0; m < 128; m++) {
    //         layer_1_output[b][h][w][m] = layer_1_bias[m];
    //       }
    //       for (int kH = 0; kH < 3; kH++) {
    //         int iH = h * 1 + kH - 1;
    //         if (iH >= 0 && iH < 32) {
    //           for (int kW = 0; kW < 3; kW++) {
    //             int iW = w * 1 + kW - 1;
    //             if (iW >= 0 && iW < 32) {
    //               for (int c = 0; c < 3; c++) {
    //                 for (int m = 0; m < 128; m++) {
    //                   layer_1_output[b][h][w][m] += layer_1_weight[kH][kW][c][m] * x[b][iH][iW][c];
    //                 }
    //               }
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }
    // }

    // float sum_cpu = 0;
    // ofstream g("layer1/orig.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum_cpu = 0;
    //     for (int h = 0; h < 32; h++) {
    //       for (int w = 0; w < 32; w++) {
    //         for (int m = 0; m < 128; m++) {
    //           sum_cpu += layer_1_output[b][h][w][m];
    //           g<<layer_1_output[b][h][w][m]<<" ";  
    //         }
    //       }
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum_cpu<<endl;
    // }

    kernel_time += layer1_conv(x, cuda_layer_1_output);

    // float sum_gpu = 0;
    // ofstream gg("layer1/par.out");
    // for(int b=0;b<BATCH_SIZE;b++){
    //     sum_gpu = 0;
    //     for(int i=b*32*32*128;i<(b+1)*32*32*128;i++){
    //         sum_gpu += cuda_layer_1_output[i];
    //         gg<<cuda_layer_1_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"batch "<<b<<": "<<sum_gpu<<endl;
    // }

    // Layer 2: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
    for (int b = 0; b < BATCH_SIZE; b++){
      for (int h = 0; h < 32; h++) {
        for (int w = 0; w < 32; w++) {
          for (int c = 0; c < 128; c++) {
            if (cuda_layer_1_output[index4D(b,h,w,c,32,32,128)] > layer_2_threshold[c]) {
              layer_2_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
            } else {
              layer_2_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
            }
          }
        }
      }
    }

    // Layer 3: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
    for (int h = 0; h < 32; h++) {
      for (int w = 0; w < 32; w++) {
        for (int m = 0; m < 128; m++) {
          layer_3_output[h][w][m] = layer_3_bias[m];
        }
        for (int kH = 0; kH < 3; kH++) {
          int iH = h * 1 + kH - 1;
          if (iH >= 0 && iH < 32) {
            for (int kW = 0; kW < 3; kW++) {
              int iW = w * 1 + kW - 1;
              if (iW >= 0 && iW < 32) {
                for (int m = 0; m < 128; m++) {
                  for (int c = 0; c < 2; c++) {
                    layer_3_output[h][w][m] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_3_weight[kH][kW][m][c] ^ layer_2_output[iH][iW][c])) - 64;
                  }
                }
              }
            }
          }
        }
      }
    }

    // // Layer 4: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}

    // for (int h = 0; h < 16; h++) {
    //   for (int w = 0; w < 16; w++) {
    //     for (int c = 0; c < 2; c++) {
    //       layer_4_output[h][w][c] = 0;
    //     }
    //     for (int kH = 0; kH < 2; kH++) {
    //       for (int kW = 0; kW < 2; kW++) {
    //         for (int c = 0; c < 2; c++) {
    //           layer_4_output[h][w][c] |= layer_3_output[h * 2 + kH][w * 2 + kW][c];
    //         }
    //       }
    //     }
    //   }
    // }
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int c = 0; c < 128; c++) {
          layer_4_output[h][w][c] = std::numeric_limits<float>::lowest();
        }
        for (int kH = 0; kH < 2; kH++) {
          for (int kW = 0; kW < 2; kW++) {
            for (int c = 0; c < 128; c++) {
              layer_4_output[h][w][c] = std::max(layer_3_output[h * 2 + kH][w * 2 + kW][c], layer_4_output[h][w][c]);
            }
          }
        }
      }
    }

    // Layer 5: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int c = 0; c < 128; c++) {
          if (layer_4_output[h][w][c] >layer_5_threshold[c]) {
            layer_5_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_5_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }

    // Layer 6: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int m = 0; m < 256; m++) {
          layer_6_output[h][w][m] = layer_6_bias[m];
        }
        for (int kH = 0; kH < 3; kH++) {
          int iH = h * 1 + kH - 1;
          if (iH >= 0 && iH < 16) {
            for (int kW = 0; kW < 3; kW++) {
              int iW = w * 1 + kW - 1;
              if (iW >= 0 && iW < 16) {
                for (int m = 0; m < 256; m++) {
                  for (int c = 0; c < 2; c++) {
                    layer_6_output[h][w][m] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_6_weight[kH][kW][m][c] ^ layer_5_output[iH][iW][c])) - 64;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Layer 7: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int c = 0; c < 256; c++) {
          if (layer_6_output[h][w][c] >layer_7_threshold[c]) {
            layer_7_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_7_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }

    // Layer 8: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int m = 0; m < 256; m++) {
          layer_8_output[h][w][m] = layer_8_bias[m];
        }
        for (int kH = 0; kH < 3; kH++) {
          int iH = h * 1 + kH - 1;
          if (iH >= 0 && iH < 16) {
            for (int kW = 0; kW < 3; kW++) {
              int iW = w * 1 + kW - 1;
              if (iW >= 0 && iW < 16) {
                for (int m = 0; m < 256; m++) {
                  for (int c = 0; c < 4; c++) {
                    layer_8_output[h][w][m] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_8_weight[kH][kW][m][c] ^ layer_7_output[iH][iW][c])) - 64;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Layer 9: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int c = 0; c < 256; c++) {
          layer_9_output[h][w][c] = std::numeric_limits<float>::lowest();
        }
        for (int kH = 0; kH < 2; kH++) {
          for (int kW = 0; kW < 2; kW++) {
            for (int c = 0; c < 256; c++) {
              layer_9_output[h][w][c] = std::max(layer_8_output[h * 2 + kH][w * 2 + kW][c], layer_9_output[h][w][c]);
            }
          }
        }
      }
    }

    // Layer 10: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int c = 0; c < 256; c++) {
          if (layer_9_output[h][w][c] >layer_10_threshold[c]) {
            layer_10_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_10_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }

    // Layer 11: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int m = 0; m < 512; m++) {
          layer_11_output[h][w][m] = layer_11_bias[m];
        }
        for (int kH = 0; kH < 3; kH++) {
          int iH = h * 1 + kH - 1;
          if (iH >= 0 && iH < 8) {
            for (int kW = 0; kW < 3; kW++) {
              int iW = w * 1 + kW - 1;
              if (iW >= 0 && iW < 8) {
                for (int m = 0; m < 512; m++) {
                  for (int c = 0; c < 4; c++) {
                    layer_11_output[h][w][m] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_11_weight[kH][kW][m][c] ^ layer_10_output[iH][iW][c])) - 64;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Layer 12: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int c = 0; c < 512; c++) {
          if (layer_11_output[h][w][c] >layer_12_threshold[c]) {
            layer_12_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_12_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }

    // Layer 13: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int m = 0; m < 512; m++) {
          layer_13_output[h][w][m] = layer_13_bias[m];
        }
        for (int kH = 0; kH < 3; kH++) {
          int iH = h * 1 + kH - 1;
          if (iH >= 0 && iH < 8) {
            for (int kW = 0; kW < 3; kW++) {
              int iW = w * 1 + kW - 1;
              if (iW >= 0 && iW < 8) {
                for (int m = 0; m < 512; m++) {
                  for (int c = 0; c < 8; c++) {
                    layer_13_output[h][w][m] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_13_weight[kH][kW][m][c] ^ layer_12_output[iH][iW][c])) - 64;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Layer 14: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
    for (int h = 0; h < 4; h++) {
      for (int w = 0; w < 4; w++) {
        for (int c = 0; c < 512; c++) {
          layer_14_output[h][w][c] = std::numeric_limits<float>::lowest();
        }
        for (int kH = 0; kH < 2; kH++) {
          for (int kW = 0; kW < 2; kW++) {
            for (int c = 0; c < 512; c++) {
              layer_14_output[h][w][c] = std::max(layer_13_output[h * 2 + kH][w * 2 + kW][c], layer_14_output[h][w][c]);
            }
          }
        }
      }
    }

    // Layer 15: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
    for (int h = 0; h < 4; h++) {
      for (int w = 0; w < 4; w++) {
        for (int c = 0; c < 512; c++) {
          if (layer_14_output[h][w][c] >layer_15_threshold[c]) {
            layer_15_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_15_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }

    // Layer 16: Flatten @ cpp.NHWC:reshape.j2 
    unsigned long long *layer_16_output = (unsigned long long *) layer_15_output;

    // Layer 17: Gemm @ cpp.binary
    for (int d = 0; d < 1024; d++) {
      layer_17_output[d] = layer_17_bias[d];
    }
    for (int d = 0; d < 1024; d++) {
      for (int i = 0; i < 128; i++) {
        layer_17_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_17_weight[d][i] ^ layer_16_output[i])) - 64;
      }
    }

    // Layer 18: Step @ cpp.binary {% else %} /{% if layer.output_shape|length > 2 %}
    for (int d = 0; d < 1024; d++) {
      if (layer_17_output[d] >layer_18_threshold[d]) {
        layer_18_output[d / 64] |= (1ULL << (63 - d % 64));
      } else {
        layer_18_output[d / 64] &= ~(1ULL << (63 - d % 64));
      }
    }

    // Layer 19: Gemm @ cpp.binary
    for (int d = 0; d < 10; d++) {
      layer_19_output[d] = layer_19_bias[d];
    }
    for (int d = 0; d < 10; d++) {
      for (int i = 0; i < 16; i++) {
        layer_19_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_19_weight[d][i] ^ layer_18_output[i])) - 64;
      }
    }

  for (int i = 0; i < 10; i++) {
    pred[i] += layer_19_output[i];
  }

  return kernel_time;

}
