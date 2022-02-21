#include "netW.hpp"
#include <iostream>
#include <fstream>

using namespace std;

void predict_NeuralNet(unsigned char * const x, float * pred, int threshold) {
  unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

    // Layer 0: Binarization

    // for(int i=0;i<28;i++){
    //   for(int j=0;j<28;j++){
    //     for(int k=0;k<64;k++){
    //       if(int(layer_0_output[i][j][k]) > threshold){
    //         layer_0_output[i][j][k] = 1;
    //       }else{
    //         layer_0_output[i][j][k] = -1;
    //       }
    //       // cout<<int(layer_0_output[i][j][k])<<" ";
    //     }
    //     // cout<<endl;
    //   }
    //   // cout<<endl;
    // }
    float sum = 0;

    // Layer 1: Conv @ cpp.NHWC {% else %} /{% if pads == [0, 0, 0, 0] %}
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

    // sum = 0;
    // ofstream g1("layer_1_orig.out");
    // for (int h = 0; h < 28; h++) {
    //   for (int w = 0; w < 28; w++) {
    //     for (int m = 0; m < 64; m++) {
    //       sum += layer_1_output[h][w][m];
    //       g1<<layer_1_output[h][w][m]<<" ";
    //     }
    //     g1<<endl;
    //   }
    // }
    // cout<<fixed<<"layer 1: "<<sum<<endl;

    // float sum = 0;
    // for (int h = 0; h < 28; h++) {
    //   for (int w = 0; w < 28; w++) {
    //     for (int m = 0; m < 64; m++) {
    //       sum += layer_1_output[h][w][m];
    //       cout<< layer_1_output[h][w][m]<<" ";
    //     }
    //   }
    // }
    // cout<<endl<<endl<<sum<<endl;

    // Layer 2: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
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

    // sum = 0;
    // ofstream g2("layer_2_orig.out");
    // for (int h = 0; h < 14; h++) {
    //   for (int w = 0; w < 14; w++) {
    //     for (int m = 0; m < 64; m++) {
    //       sum += layer_2_output[h][w][m];
    //       g2<<layer_2_output[h][w][m]<<" ";
    //     }
    //   }
    // }
    // cout<<fixed<<"layer 2: "<<sum<<endl;

    // Layer 3: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
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

    // Layer 4: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
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

    // sum = 0;
    // ofstream g4("layer_4_orig.out");
    // for (int h = 0; h < 14; h++) {
    //   for (int w = 0; w < 14; w++) {
    //     for (int m = 0; m < 64; m++) {
    //       sum += layer_4_output[h][w][m];
    //       g4<<layer_4_output[h][w][m]<<" ";
    //     }
    //   }
    // }
    // cout<<fixed<<"layer 4: "<<sum<<endl;

    // Layer 5: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
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

    // sum = 0;
    // ofstream g5("layer_5_orig.out");
    // for (int h = 0; h < 7; h++) {
    //   for (int w = 0; w < 7; w++) {
    //     for (int m = 0; m < 64; m++) {
    //       sum += layer_5_output[h][w][m];
    //       g5<<layer_5_output[h][w][m]<<" ";
    //     }
    //   }
    // }
    // cout<<fixed<<"layer 5: "<<sum<<endl;

    // Layer 6: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
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

    // Layer 7: Flatten @ cpp.NHWC:reshape.j2 
    unsigned long long *layer_7_output = (unsigned long long *) layer_6_output;

    // Layer 8: Gemm @ cpp.binary 
    for (int d = 0; d < 2048; d++) {
      layer_8_output[d] = layer_8_bias[d];
    }
    for (int d = 0; d < 2048; d++) {
      for (int i = 0; i < 49; i++) {
        layer_8_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_8_weight[d][i] ^ layer_7_output[i])) - 64;
      }
    }

    // sum = 0;
    // ofstream g8("layer_8_orig.out");
    // for (int d = 0; d < 2048; d++) {
    //   sum += layer_8_output[d];
    //   g8<<layer_8_output[d]<<" ";
    // }
    // cout<<fixed<<"layer 8: "<<sum<<endl;

    // Layer 9: Step @ cpp.binary {% else %} /{% if layer.output_shape|length > 2 %}
    for (int d = 0; d < 2048; d++) {
      if (layer_8_output[d] >layer_9_threshold[d]) {
        layer_9_output[d / 64] |= (1ULL << (63 - d % 64));
      } else {
        layer_9_output[d / 64] &= ~(1ULL << (63 - d % 64));
      }
    }

    // Layer 10: Gemm @ cpp.binary
    for (int d = 0; d < 10; d++) {
      layer_10_output[d] = layer_10_bias[d];
    }
    for (int d = 0; d < 10; d++) {
      for (int i = 0; i < 32; i++) {
        layer_10_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_10_weight[d][i] ^ layer_9_output[i])) - 64;
      }
    }

    // sum = 0;
    // ofstream g10("layer_10_orig.out");
    // for (int d = 0; d < 10; d++) {
    //   sum += layer_10_output[d];
    //   g10<<layer_10_output[d]<<" ";
    // }
    // cout<<fixed<<"layer 10: "<<sum<<endl;

  for (int i = 0; i < 10; i++) {
    pred[i] += layer_10_output[i];
  }

}
