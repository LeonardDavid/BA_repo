#include "netW.hpp"
#include "cuda_net.h"
#include <iostream>

using namespace std;

float predict_NeuralNet(unsigned char * const x, float * pred) {
  unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

  // add all kernel_time s
  float kernel_time = 0;
  kernel_time += layer1_conv(x, cuda_layer_1_output);
  
  kernel_time += layer2_maxpool(cuda_layer_1_output, cuda_layer_2_output);

  /*
      to run without the outputs from line 58-126: 
        - uncomment lines 19-50
        - comment lines 58-126
  */
  // unsigned long long *cuda_layer_3_output = (unsigned long long *) layer_3_output;
  // layer3_step(cuda_layer_2_output, cuda_layer_3_output);
  
  /* Layer 3 does not work because:
    - if run without the line 'layer_3_output[h][w][c / 64] |= (1ULL << (63 - c % 64));' every element will be 0
    - for ABSOLUTELY NO REASON if the line is present AFTER (!!) the cout/calculation with l30, the correct answer will be calculated
  */

 /*
    - kernel calculates for layer_3_output[h][w][0] correctly, but from 0-64 not
 */ 
 
  for (int h = 0; h < 14; h++) {
    for (int w = 0; w < 14; w++) {
      for (int c = 0; c < 64; c++) {
        if (cuda_layer_2_output[index3D(h,w,c,14,64)] > layer_3_threshold[c]) {
          layer_3_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
        } else {
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

  // int sum1;
  // float sum2;
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
  // cout<<endl;

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
  //     // sum1 += layer_3_output[h][w][0]%10;
  //     // cout<<layer_3_output[h][w][0]<<" ";
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

  // // for (int h = 0; h < 14; h++) {
  // //   for (int w = 0; w < 14; w++) {
  // //     for (int c = 0; c < 64; c++) {
  // //       // printf("1 ");
  // //       if (cuda_layer_2_output[index3D(h,w,c,14,64)] > layer_3_threshold[c]) {
  // //         // printf("%d ", layer_3_threshold[c]);
  // //         // cout<<"("<<layer_3_output[h][w][c/64]<<" - ";
  // //         slayer_3_output[h][w][c/64] = 1;// |= (1ULL << (63 - c % 64));
  // //         // cout<<layer_3_output[h][w][c/64]<<") ";
  // //       } else {
  // //         // printf("%d ", layer_3_threshold[c]);
  // //         // cout<<"("<<layer_3_output[h][w][c/64]<<" - ";
  // //         slayer_3_output[h][w][c/64] = -1; //&= ~(1ULL << (63 - c % 64));
  // //         // cout<<layer_3_output[h][w][c/64]<<") ";
  // //       }
  // //     }
  // //     // cout<<endl;
  // //   }
  // //   // cout<<endl;
  // // }

  // // for (int h = 0; h < 14; h++) {
  // //   for (int w = 0; w < 14; w++) {
  // //     for (int m = 0; m < 64; m++) {
  // //       sum2 += slayer_3_output[h][w][m]%10;
  // //       cout<<slayer_3_output[h][w][m]<<" ";
  // //     }
  // //   }
  // //   cout<<endl;
  // // }
  // // cout<<endl;

  // // cout<<endl<<sum2<<endl;
  // cout<<endl;
  // cout<<"---------------------------------end-GPU-----------------------------"<<endl;

  // if(sum1!=sum2){
  //   cout<<"FAIL"<<endl;
  // }else{
  //   cout<<"PASS"<<endl;
  // }
  // printf("cpp: %d - cuda: %f\n",sum1,sum2);

  // cout<<"---------------------------------end---------------------------------"<<endl;
  // cout<<endl<<endl;


  kernel_time += layer4_conv(cuda_layer_3_output, cuda_layer_4_output);
  kernel_time += layer5_maxpool(cuda_layer_4_output, cuda_layer_5_output);

  // layer6_step(cuda_layer_5_output, cuda_layer_6_output);
  /*
    Same as layer3_step
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
  
  kernel_time += layer8_gemm(layer_7_output, cuda_layer_8_output);

  // layer9_step(cuda_layer_8_output, cuda_layer_9_output);
  /*
    Same as layer9_step
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
  kernel_time += layer10_gemm(cuda_layer_9_output, cuda_layer_10_output);
  
  for (int i = 0; i < 10; i++) {
    pred[i] += cuda_layer_10_output[i];
  }

  return kernel_time;


  /*
    after this line: previous step by step implementations and checking of layers

  */

// cout<<"--------------------------------begin--------------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-CPU----------------------------"<<endl;



// float sum1 = 0;
// for (int h = 0; h < 14; h++) {
//   for (int w = 0; w < 14; w++) {
//     for (int m = 0; m < 64; m++) {
//       sum1 += layer_2_output[h][w][m];
//      // cout<<layer_2_output[h][w][m]<<" ";
//     }
//   }
// }
// // cout<<sum1<<endl;

// cout<<"---------------------------------end-CPU-----------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-GPU----------------------------"<<endl;

// float sum2 = 0;
// sum2 = cuda();
// // cout<<sum2<<endl;

// cout<<"---------------------------------end-GPU-----------------------------"<<endl;

// if(sum1!=sum2){
//   cout<<"FAIL"<<endl;
// }else{
//   cout<<"PASS"<<endl;
// }
// printf("cpp: %f - cuda: %f\n",sum1,sum2);

// cout<<"---------------------------------end---------------------------------"<<endl;
// cout<<endl<<endl;




// cout<<"--------------------------------begin--------------------------------"<<endl;
// cout<<endl;
// // cout<<"---------------------------------end---------------------------------"<<endl;

// // Layer 1: Conv
// cout<<"--------------------------------begin-CPU----------------------------"<<endl;

// for (int h = 0; h < 28; h++) {
//   for (int w = 0; w < 28; w++) {
//     for (int m = 0; m < 64; m++) {
//       layer_1_output[h][w][m] = layer_1_bias[m];
//     }
//     for (int kH = 0; kH < 3; kH++) {
//       int iH = h * 1 + kH - 1;
//       if (iH >= 0 && iH < 28) {
//         for (int kW = 0; kW < 3; kW++) {
//           int iW = w * 1 + kW - 1;
//           if (iW >= 0 && iW < 28) {
//             for (int c = 0; c < 1; c++) {
//               for (int m = 0; m < 64; m++) {
//                 layer_1_output[h][w][m] += layer_1_weight[kH][kW][c][m] * layer_0_output[iH][iW][c];
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

// float sum1 = 0;
// for (int h = 0; h < 28; h++) {
//   for (int w = 0; w < 28; w++) {
//     for (int m = 0; m < 64; m++) {
//       sum1 += layer_1_output[h][w][m];
//       // cout<<layer_1_output[h][w][m]<<" ";
//     }
//   }
// }
// // cout<<sum1<<endl;
// cout<<"---------------------------------end-CPU-----------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-GPU----------------------------"<<endl;
// float sum2 = 0;
// sum2 = layer1_conv(x);
// // cout<<sum2<<endl;
// cout<<"---------------------------------end-GPU-----------------------------"<<endl;

// if(sum1!=sum2){
//   cout<<"FAIL"<<endl;
// }else{
//   cout<<"PASS"<<endl;
// }
// printf("cpp: %f - cuda: %f\n",sum1,sum2);

// cout<<"---------------------------------end---------------------------------"<<endl;
// cout<<endl<<endl;


// cout<<"cuda_layer_1_output"<<endl;
// for(int i=0;i<50176;i++){
//   cout<<cuda_layer_1_output[i]<<" ";
// }


// Layer 2: MaxPool
// cout<<"--------------------------------begin--------------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-CPU----------------------------"<<endl;
// for (int h = 0; h < 14; h++) {
//   for (int w = 0; w < 14; w++) {
//     for (int c = 0; c < 64; c++) {
//       layer_2_output[h][w][c] = std::numeric_limits<float>::lowest();
//     }
//     for (int kH = 0; kH < 2; kH++) {
//       for (int kW = 0; kW < 2; kW++) {
//         for (int c = 0; c < 64; c++) {
//           layer_2_output[h][w][c] = std::max(cuda_layer_1_output[index3D((h * 2 + kH),(w * 2 + kW),c,28,64)], layer_2_output[h][w][c]);
//         }
//       }
//     }
//   }
// }

// float sum1 = 0;
// for (int h = 0; h < 14; h++) {
//   for (int w = 0; w < 14; w++) {
//     for (int m = 0; m < 64; m++) {
//       sum1 += layer_2_output[h][w][m];
//       cout<<layer_2_output[h][w][m]<<" ";
//     }
//   }
// }
// // cout<<sum1<<endl;

// cout<<"---------------------------------end-CPU-----------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-GPU----------------------------"<<endl;

// float sum2 = 0;
// sum2 = layer2_maxpool(cuda_layer_1_output, cuda_layer_2_output);
// // cout<<sum2<<endl;

// cout<<"---------------------------------end-GPU-----------------------------"<<endl;

// if(sum1!=sum2){
//   cout<<"FAIL"<<endl;
// }else{
//   cout<<"PASS"<<endl;
// }
// printf("cpp: %f - cuda: %f\n",sum1,sum2);

// cout<<"---------------------------------end---------------------------------"<<endl;
// cout<<endl<<endl;


// TODO Layer 3: Step WORK IN PROGRESS


// Layer 4: Conv
// cout<<"--------------------------------begin--------------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-CPU----------------------------"<<endl;

// // cout<<"layer3_output[14][14][c]: "<<endl;
// //   for (int h = 0; h < 14; h++) {
// //     for (int w = 0; w < 14; w++) {
// //       for (int c = 0; c < 64; c++) {
// //         if(h==1 && w==1)
// //           cout<<layer_3_output[h][w][c]<<" ";
// //         // cout<<"("<<c<<" - "<<c/64<<")";
// //       }
// //       // cout<<endl;
// //     }
// //     // cout<<endl;
// //   }
// //   cout<<endl;


//   // flatten layer_3_output into cuda_layer_3_output for further usage
//   for(int i=0;i<14;i++){
//     for(int j=0;j<14;j++){
//       for(int k=0;k<64;k++){
//         cuda_layer_3_output[index3D(i,j,k,14,64)] = layer_3_output[i][j][k];
//         // if(i==1 && j==1)
//         //   cout<<cuda_layer_3_output[index3D(i,j,k,14,64)]<<" ";
//       }
//     }
//   }
//   // cout<<endl;


//   for (int h = 0; h < 14; h++) {
//     for (int w = 0; w < 14; w++) {
//       for (int m = 0; m < 64; m++) {
//         // if(h==0 && w==0 && m==0)
//         //   cout<<"before1: "<<layer_4_output[h][w][m];
//         layer_4_output[h][w][m] = 0; //layer_4_bias[m];
//         // if(h==0 && w==0 && m==0)
//         //   cout<<" after1: "<<layer_4_output[h][w][m]<<endl;
//       }
//       for (int kH = 0; kH < 3; kH++) {
//         int iH = h * 1 + kH - 1;
//         if (iH >= 0 && iH < 14) {
//           for (int kW = 0; kW < 3; kW++) {
//             int iW = w * 1 + kW - 1;
//             if (iW >= 0 && iW < 14) {
//               for (int m = 0; m < 64; m++) {
//                 for (int c = 0; c < 1; c++) {
//                   // if(h==0 && w==0 && m==0){
//                   //   cout<<"--------------------"<<endl;
//                   //   cout<<"before2: "<<layer_4_output[h][w][m];}
//                   layer_4_output[h][w][m] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_4_weight[kH][kW][m][c] ^ cuda_layer_3_output[index3D(iH,iW,c,14,64)])) - 64;
//                   // if(h==0 && w==0 && m==0)
//                   //   cout<<" after2: "<<layer_4_output[h][w][m]<<" = 2*ppc("<<layer_4_weight[kH][kW][m][c]<<" ^ "<<cuda_layer_3_output[index3D(iH,iW,c,14,64)]<<")-64"<<endl;
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }


// int sum1 = 0;
// for (int h = 0; h < 14; h++) {
//   for (int w = 0; w < 14; w++) {
//     for (int m = 0; m < 64; m++) {
//       sum1 += layer_4_output[h][w][m];
//      // cout<<layer_4_output[h][w][m]<<" ";
//     }
//   }
// }
// // cout<<sum1<<endl;

// cout<<"---------------------------------end-CPU-----------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-GPU----------------------------"<<endl;

// int sum2 = 0;
// sum2 = layer4_conv(cuda_layer_3_output, cuda_layer_4_output);
// // // cout<<sum2<<endl;

// cout<<"---------------------------------end-GPU-----------------------------"<<endl;

// if(sum1!=sum2){
//   cout<<"FAIL"<<endl;
// }else{
//   cout<<"PASS"<<endl;
// }
// printf("cpp: %d - cuda: %d\n",sum1,sum2);

// cout<<"---------------------------------end---------------------------------"<<endl;
// cout<<endl<<endl;


// Layer 5: MaxPool
// cout<<"--------------------------------begin--------------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-CPU----------------------------"<<endl;

//   for (int h = 0; h < 7; h++) {
//     for (int w = 0; w < 7; w++) {
//       for (int c = 0; c < 64; c++) {
//         layer_5_output[h][w][c] = std::numeric_limits<signed short>::lowest();
//       }
//       for (int kH = 0; kH < 2; kH++) {
//         for (int kW = 0; kW < 2; kW++) {
//           for (int c = 0; c < 64; c++) {
//             layer_5_output[h][w][c] = std::max(cuda_layer_4_output[index3D((h * 2 + kH),(w * 2 + kW),c,14,64)], layer_5_output[h][w][c]);
//           }
//         }
//       }
//     }
//   }

// int sum1 = 0;
// for (int h = 0; h < 7; h++) {
//   for (int w = 0; w < 7; w++) {
//     for (int m = 0; m < 64; m++) {
//       sum1 += layer_5_output[h][w][m];
//      // cout<<layer_2_output[h][w][m]<<" ";
//     }
//   }
// }
// // cout<<sum1<<endl;

// cout<<"---------------------------------end-CPU-----------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-GPU----------------------------"<<endl;

// int sum2 = 0;
// sum2 = layer5_maxpool(cuda_layer_4_output, cuda_layer_5_output);
// // cout<<sum2<<endl;

// cout<<"---------------------------------end-GPU-----------------------------"<<endl;

// if(sum1!=sum2){
//   cout<<"FAIL"<<endl;
// }else{
//   cout<<"PASS"<<endl;
// }
// printf("cpp: %d - cuda: %d\n",sum1,sum2);

// cout<<"---------------------------------end---------------------------------"<<endl;
// cout<<endl<<endl;



// Layer 6: Step
// for (int h = 0; h < 7; h++) {
//   for (int w = 0; w < 7; w++) {
//     for (int c = 0; c < 64; c++) {
//       if (cuda_layer_5_output[index3D(h,w,c,7,64)] > layer_6_threshold[c]) {
//         layer_6_output[h][w][c / 64] |= (1ULL << (63 - c % 64));
//       } else {
//         layer_6_output[h][w][c / 64] &= ~(1ULL << (63 - c % 64));
//       }
//     }
//   }
// }

// output layer 6 results
// for (int h = 6; h < 7; h++) {
//   for (int w = 6; w < 7; w++) {
//     std::cout << layer_6_output[h][w][0] << std::endl;
//   }
// }

// Layer 7: Flatten
// unsigned long long *layer_7_output = (unsigned long long *) layer_6_output;


// Layer 8: Gemm
// cout<<"--------------------------------begin--------------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-CPU----------------------------"<<endl;

// for (int d = 0; d < 2048; d++) {
//   layer_8_output[d] = layer_8_bias[d];
// }
// for (int d = 0; d < 2048; d++) {
//   for (int i = 0; i < 49; i++) {
//     layer_8_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_8_weight[d][i] ^ layer_7_output[i])) - 64;
//   }
// }


// int sum1 = 0;
// for (int d = 0; d < 2048; d++) {
//       sum1 += layer_8_output[d];
//      // cout<<layer_2_output[h][w][m]<<" ";
// }

// // cout<<sum1<<endl;

// cout<<"---------------------------------end-CPU-----------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-GPU----------------------------"<<endl;

// int sum2 = 0;
// // unsigned long long* cuda_layer_8_weight = (unsigned long long *)malloc(sizeof(unsigned long long)*100352); // (2048*49 -> unsigned long long)
// //   for(int i=0;i<2048;i++){
// //       for(int j=0;j<49;j++){
// //           cuda_layer_8_weight[i*49+j] = layer_8_weight[i][j];
// //       }
// //   }
// sum2 = layer8_gemm(layer_7_output, cuda_layer_8_output);
// // cout<<sum2<<endl;

// cout<<"---------------------------------end-GPU-----------------------------"<<endl;

// if(sum1!=sum2){
//   cout<<"FAIL"<<endl;
// }else{
//   cout<<"PASS"<<endl;
// }
// printf("cpp: %d - cuda: %d\n",sum1,sum2);

// cout<<"---------------------------------end---------------------------------"<<endl;
// cout<<endl<<endl;


// Layer 9: Step
// for (int d = 0; d < 2048; d++) {
//   if (cuda_layer_8_output[d] >layer_9_threshold[d]) {
//     layer_9_output[d / 64] |= (1ULL << (63 - d % 64));
//   } else {
//     layer_9_output[d / 64] &= ~(1ULL << (63 - d % 64));
//   }
// }

// Layer 10: Gemm
// cout<<"--------------------------------begin--------------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-CPU----------------------------"<<endl;

// for (int d = 0; d < 10; d++) {
//     layer_10_output[d] = layer_10_bias[d];
//   }
//   for (int d = 0; d < 10; d++) {
//     for (int i = 0; i < 32; i++) {
//       layer_10_output[d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_10_weight[d][i] ^ layer_9_output[i])) - 64;
//     }
//   }

// int sum1 = 0;
// for (int h = 0; h < 10; h++) {
//   sum1+=layer_10_output[h];
// }

// // cout<<sum1<<endl;

// cout<<"---------------------------------end-CPU-----------------------------"<<endl;
// cout<<endl;
// cout<<"--------------------------------begin-GPU----------------------------"<<endl;

// int sum2 = 0;
// sum2 = layer10_gemm(cuda_layer_9_output, cuda_layer_10_output);
// // cout<<sum2<<endl;

// cout<<"---------------------------------end-GPU-----------------------------"<<endl;

// if(sum1!=sum2){
//   cout<<"FAIL"<<endl;
// }else{
//   cout<<"PASS"<<endl;
// }
// printf("cpp: %d - cuda: %d\n",sum1,sum2);

// cout<<"---------------------------------end---------------------------------"<<endl;
// cout<<endl<<endl;


// for (int i = 0; i < 10; i++) {
//   pred[i] += layer_10_output[i];
// }

}