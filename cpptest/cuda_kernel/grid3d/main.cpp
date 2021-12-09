#include <iostream>
#include "kernel_main.h"
#include "utils.h"

using namespace std;

int main(){

    // short* arr = (short *) malloc(sizeof(short) * 5);
    short arr[25];
    for (int i=0; i<5; ++i){
        for(int j=0;j<5;j++){
            arr[i*5+j] = i+j;
        }
    }

    cout<<"main:"<<endl;
    for (int i=0; i<25; i++){
        if(i%5==0)
            cout<<endl;
        cout<<arr[i]<<" ";
    }
    cout<<endl<<endl;

    cuda_k(arr);

    cout<<"cuda:"<<endl;
    for (int i=0; i<25; i++){
        if(i%5==0)
            cout<<endl;
        cout<<arr[i]<<" ";
    }
    cout<<endl<<endl;

    // free(arr);
    return 0;
}