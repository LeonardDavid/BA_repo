#include "netW.hpp"
#include <iostream>

/*
  just don't look after line 190 :')
*/

int predict_NeuralNet(unsigned char * const x, int thread, int label, float * pred) { 

  /*
    these declarations have to be made locally, not globally in order to avoid multiple threads writing in the same area
  */
  static float layer_1_output[28][28][64];
  static float layer_2_output[14][14][64];
  static unsigned long long layer_3_output[14][14][1];
  static signed short layer_4_output[14][14][64];
  static signed short layer_5_output[7][7][64];
  static unsigned long long layer_6_output[7][7][1];

  static signed short layer_8_output[2048];
  static unsigned long long layer_9_output[32];
  static signed short layer_10_output[10];

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
    // layer_7_output = (unsigned long long *) layer_6_output;

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
    
  float out[10] = {0};
  std::fill(out, out+10, 0);

  for (int i = 0; i < 10; i++) {
    out[i] += layer_10_output[i];
  }

  int match = 0;
  float max = out[0];
  int argmax = 0;
  for (int j = 1; j < 10; j++) {
      if (out[j] > max) {
          max = out[j];
          argmax = j;
      }
  }

  if (argmax == label) {
      match++;
  }

  return match;


}

int predict_NeuralNet0(unsigned char * const x, int thread, int label, float * pred) { 

  static float layer_1_output[28][28][64];
  static float layer_2_output[14][14][64];
  static unsigned long long layer_3_output[14][14][1];
  static signed short layer_4_output[14][14][64];
  static signed short layer_5_output[7][7][64];
  static unsigned long long layer_6_output[7][7][1];

  static signed short layer_8_output[2048];
  static unsigned long long layer_9_output[32];
  static signed short layer_10_output[10];

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
    // layer_7_output = (unsigned long long *) layer_6_output;

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
    
  float out[10] = {0};
  std::fill(out, out+10, 0);

  for (int i = 0; i < 10; i++) {
    out[i] += layer_10_output[i];
  }

  int match = 0;
  float max = out[0];
  int argmax = 0;
  for (int j = 1; j < 10; j++) {
      if (out[j] > max) {
          max = out[j];
          argmax = j;
      }
  }

  if (argmax == label) {
      match++;
  }

  return match;


}

int predict_NeuralNet1(unsigned char * const x, int thread, int label, float * pred) { 

  static float layer_1_output[28][28][64];
  static float layer_2_output[14][14][64];
  static unsigned long long layer_3_output[14][14][1];
  static signed short layer_4_output[14][14][64];
  static signed short layer_5_output[7][7][64];
  static unsigned long long layer_6_output[7][7][1];

  static signed short layer_8_output[2048];
  static unsigned long long layer_9_output[32];
  static signed short layer_10_output[10];

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
    // layer_7_output = (unsigned long long *) layer_6_output;

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
    
  float out[10] = {0};
  std::fill(out, out+10, 0);

  for (int i = 0; i < 10; i++) {
    out[i] += layer_10_output[i];
  }

  int match = 0;
  float max = out[0];
  int argmax = 0;
  for (int j = 1; j < 10; j++) {
      if (out[j] > max) {
          max = out[j];
          argmax = j;
      }
  }

  if (argmax == label) {
      match++;
  }

  return match;


}

int predict_NeuralNet2(unsigned char * const x, int thread, int label, float * pred) { 

  static float layer_1_output[28][28][64];
  static float layer_2_output[14][14][64];
  static unsigned long long layer_3_output[14][14][1];
  static signed short layer_4_output[14][14][64];
  static signed short layer_5_output[7][7][64];
  static unsigned long long layer_6_output[7][7][1];

  static signed short layer_8_output[2048];
  static unsigned long long layer_9_output[32];
  static signed short layer_10_output[10];

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
    // layer_7_output = (unsigned long long *) layer_6_output;

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
    
  float out[10] = {0};
  std::fill(out, out+10, 0);

  for (int i = 0; i < 10; i++) {
    out[i] += layer_10_output[i];
  }

  int match = 0;
  float max = out[0];
  int argmax = 0;
  for (int j = 1; j < 10; j++) {
      if (out[j] > max) {
          max = out[j];
          argmax = j;
      }
  }

  if (argmax == label) {
      match++;
  }

  return match;


}

int predict_NeuralNet3(unsigned char * const x, int thread, int label, float * pred) { 

  static float layer_1_output[28][28][64];
  static float layer_2_output[14][14][64];
  static unsigned long long layer_3_output[14][14][1];
  static signed short layer_4_output[14][14][64];
  static signed short layer_5_output[7][7][64];
  static unsigned long long layer_6_output[7][7][1];

  static signed short layer_8_output[2048];
  static unsigned long long layer_9_output[32];
  static signed short layer_10_output[10];

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
    // layer_7_output = (unsigned long long *) layer_6_output;

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
    
  float out[10] = {0};
  std::fill(out, out+10, 0);

  for (int i = 0; i < 10; i++) {
    out[i] += layer_10_output[i];
  }

  int match = 0;
  float max = out[0];
  int argmax = 0;
  for (int j = 1; j < 10; j++) {
      if (out[j] > max) {
          max = out[j];
          argmax = j;
      }
  }

  if (argmax == label) {
      match++;
  }

  return match;


}

int predict_NeuralNet4(unsigned char * const x, int thread, int label, float * pred) { 

  static float layer_1_output[28][28][64];
  static float layer_2_output[14][14][64];
  static unsigned long long layer_3_output[14][14][1];
  static signed short layer_4_output[14][14][64];
  static signed short layer_5_output[7][7][64];
  static unsigned long long layer_6_output[7][7][1];

  static signed short layer_8_output[2048];
  static unsigned long long layer_9_output[32];
  static signed short layer_10_output[10];

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
    // layer_7_output = (unsigned long long *) layer_6_output;

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
    
  float out[10] = {0};
  std::fill(out, out+10, 0);

  for (int i = 0; i < 10; i++) {
    out[i] += layer_10_output[i];
  }

  int match = 0;
  float max = out[0];
  int argmax = 0;
  for (int j = 1; j < 10; j++) {
      if (out[j] > max) {
          max = out[j];
          argmax = j;
      }
  }

  if (argmax == label) {
      match++;
  }

  return match;


}
