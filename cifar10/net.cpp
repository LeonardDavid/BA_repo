#include "net.hpp"
#include "netW.hpp"
#include<algorithm>

static float layer_1_output[32][32][128];
static unsigned long long layer_2_output[32][32][2];
static float layer_3_output[32][32][128];
static float layer_4_output[16][16][128];
static unsigned long long layer_5_output[16][16][2];
static float layer_6_output[16][16][256];
static unsigned long long layer_7_output[16][16][4];
static float layer_8_output[16][16][256];
static float layer_9_output[8][8][256];
static unsigned long long layer_10_output[8][8][4];
static float layer_11_output[8][8][512];
static unsigned long long layer_12_output[8][8][8];
static float layer_13_output[8][8][512];
static float layer_14_output[4][4][512];
static unsigned long long layer_15_output[4][4][8];
static float layer_17_output[1024];
static unsigned long long layer_18_output[16];
static float layer_19_output[10];

  void predict_NeuralNet(unsigned char x[][32][3], float * pred) {
		//unsigned char (*layer_0_output)[32][3] = (unsigned char (*)[32][3]) x;

      // Layer 1: Conv
      for (int h = 0; h < 32; h++) {
        for (int w = 0; w < 32; w++) {
          for (int m = 0; m < 128; m++) {
            layer_1_output[h][w][m] = layer_1_bias[m];
          }
          for (int kH = 0; kH < 3; kH++) {
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 32) {
              for (int kW = 0; kW < 3; kW++) {
                int iW = w * 1 + kW - 1;
                if (iW >= 0 && iW < 32) {
                  for (int c = 0; c < 3; c++) {
                    for (int m = 0; m < 128; m++) {
                      layer_1_output[h][w][m] += layer_1_weight[kH][kW][c][m] * x[iH][iW][c];
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Layer 2: Step
      for (int h = 0; h < 32; h++) {
        for (int w = 0; w < 32; w++) {
          for (int c = 0; c < 128; c++) {
            if (layer_1_output[h][w][c] >layer_2_threshold[c]) {
              layer_2_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
            } else {
              layer_2_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
            }
          }
        }
      }

      // Layer 3: Conv
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

      // // Layer 4: MaxPool
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

      // Layer 5: Step
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

      // Layer 6: Conv
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

      // Layer 7: Step
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

      // Layer 8: Conv
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

      // Layer 9: MaxPool
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

      // Layer 10: Step
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

      // Layer 11: Conv
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

      // Layer 12: Step
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

      // Layer 13: Conv
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

      // Layer 14: MaxPool
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

      // Layer 15: Step
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

      // Layer 16: Flatten
      unsigned long long *layer_16_output = (unsigned long long *) layer_15_output;

      // Layer 17: Gemm
      for (int d = 0; d < 1024; d++) {
        layer_17_output[d] = layer_17_bias[d];
      }
      for (int d = 0; d < 1024; d++) {
        for (int i = 0; i < 128; i++) {
          layer_17_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_17_weight[d][i] ^ layer_16_output[i])) - 64;
        }
      }

      // Layer 18: Step
      for (int d = 0; d < 1024; d++) {
        if (layer_17_output[d] >layer_18_threshold[d]) {
          layer_18_output[d / 64] |= (1ULL << (63 - d % 64));
        } else {
          layer_18_output[d / 64] &= ~(1ULL << (63 - d % 64));
        }
      }

      // Layer 19: Gemm
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

  }
