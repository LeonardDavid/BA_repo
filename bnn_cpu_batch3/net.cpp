#include "netW.hpp"
#include "cuda_net.h"
#include <iostream>

using namespace std;

void predict_NeuralNet(unsigned char * const x, float * pred) {
  // unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

  layer1_conv(x, cuda_layer_1_output);

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

}