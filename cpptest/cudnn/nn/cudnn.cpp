/*
    run with:
    $ make
    $ ./cu.o
*/  

#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <fstream>
#include "MNISTLoader.h"

/**
 * Minimal example to apply sigmoid activation on a tensor 
 * using cuDNN.
 **/
int main(int argc, char** argv)
{    
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0); // use GPU0
    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl<<std::endl;

    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;
    int n = 1, h = 28, w = 28, c = 1;
    int NUM_ELEMENTS = n*h*w*c;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, h, w, c);

    // create the tensor
    unsigned char *x;
    cudaMallocManaged(&x, NUM_ELEMENTS * sizeof(unsigned char));
    std::ifstream f ("C:/Users/leona/Projects/TU-Dortmund/BA_repo/bnn_cpu_batch4/data.out");
    for(int i=0;i<NUM_ELEMENTS;i++){
        f>>x[i];
    }

    std::cout << "Original array: "<<std::endl; 
    for(int i=0;i<h;i++) {
        for(int j=0;j<w;j++){
            printf("%d ", x[i*28+j]);
        }
        printf("\n");
    }

    // create activation function descriptor
    float alpha[1] = {1};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    cudnnActivationForward(
        handle_,
        sigmoid_activation,
        alpha,
        x_desc,
        x,
        beta,
        x_desc,
        x
    );

    cudnnDestroy(handle_);
    std::cout << std::endl << "Destroyed cuDNN handle." << std::endl<<std::endl;
    std::cout << "New array: "<<std::endl;
    for(int i=0;i<h;i++) {
        for(int j=0;j<w;j++){
            printf("%d ", x[i*28+j]);
        }
        printf("\n");
    }
    std::cout << std::endl;
    cudaFree(x);
    return 0;
}
