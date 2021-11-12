#include <iostream>
#include "kernel_main.h"

int main(){

    // short* arr = (short *) malloc(sizeof(short) * 5);
    short arr[5];
    for (int i=0; i<5; ++i)
        arr[i] = i+1;
    
    cuda_k(arr);

    std::cout<<"main:"<<std::endl;
    for (int i=0; i<5; ++i)
        std::cout<<arr[i]<<" ";
    std::cout<<std::endl;

    // free(arr);
    return 0;
}