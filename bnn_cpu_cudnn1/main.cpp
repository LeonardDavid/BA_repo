/*
    Run with: 
    $ make
    $ ./batch_par_gpu.o
*/

#include <iostream>
#include <chrono>
#include <algorithm>
#include <tuple>
#include <fstream>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "MNISTLoader.h"
#include "utils.h"

#ifdef BINARY
#define INPUT_FEATURE char
// #include "net.hpp"
#elif INT16
#define INPUT_FEATURE int
// #include "net.hpp"
#else
#define INPUT_FEATURE float
// #include "net.hpp"
#endif

using namespace std;

auto benchmark(vector<MNISTLoader> &loaderx, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[BATCH_SIZE*10] = {0};
#else
    float output[BATCH_SIZE*10] = {0};
    std::fill(output, output+10*BATCH_SIZE, 0);
#endif

    auto startcpu = std::chrono::high_resolution_clock::now();

    ofstream g ("original.out");
    ofstream gg ("new.out");

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    cout << "Found " << numGPUs << " GPUs" << endl;
    cudaSetDevice(0); // use GPU0

    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    cout << "Compute capability: " << devProp.major << "." << devProp.minor << endl << endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    cout << "Created cuDNN handle" << endl << endl;

    int factor = 1;
    int matches[BATCH_SIZE] = {0};
    int const imgsize = 28*28;
    int lsize = 1904/BATCH_SIZE; //loaderx[0].size();
    // max lsize for 1 batch: 1904
    // max lsize for n batches: floor(1904/n)
    printf("lsize: %d\n", lsize);
    float total_kernel_time = 0;

    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;
    int n = lsize, h = 28, w = 28, c = 1;
    int NUM_ELEMENTS = n*c*h*w;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, h, w, c);
    
    unsigned char * img; //img[BATCH_SIZE*imgsize];
    img = (unsigned char*) malloc (NUM_ELEMENTS*BATCH_SIZE*imgsize);
    cudaMallocManaged(&img, NUM_ELEMENTS * BATCH_SIZE * imgsize * sizeof(unsigned char));

    // load label i of corresponding image from every batch in an array
    int label[BATCH_SIZE];

    for(int b=0; b<BATCH_SIZE; b++){        // b := # batch
        for(int i=0; i<lsize; i+=factor){         // i := # image
            for(int p=0; p<imgsize; p++){   // p := # pixel
                img[index3D(b,i,p,lsize,imgsize)] = loaderx[b].images(i)[p]; //  loaderx[b].images(i)[p];
            }
            // label[b] = loaderx[b].labels(i); // loaderx[b].labels(i);
        }
    }
    
    // display img array
    // cout<<"Original image: "<<endl;
    for(int b=0;b<BATCH_SIZE;b++){
        // printf("batch: %d, label %d:\n",b,label[b]);
        for(int im=0;im<lsize;im+=factor){
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    // printf("%d ", img[index4D(b,im,i,j,lsize,28,28)]);
                    g<<int(img[index4D(b,im,i,j,lsize,28,28)])<<" ";
                }
                // printf("\n");
                g<<"\n";
            }
            // printf("\n\n");
            g<<"\n\n";
        }
    }

    // create activation function descriptor
    float alpha[1] = {1};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    auto startgpu = std::chrono::high_resolution_clock::now();
    cudnnActivationForward(
        handle_,
        sigmoid_activation,
        alpha,
        x_desc,
        img,
        beta,
        x_desc,
        img
    );
    auto endgpu = std::chrono::high_resolution_clock::now();

    // handle has to be destroyed before accesing img
    cudnnDestroy(handle_);
    cout << endl << "Destroyed cuDNN handle." << endl << endl;

    // cout<<"New image: "<<endl;
    for(int b=0;b<BATCH_SIZE;b++){
        // printf("batch: %d, label %d:\n",b,label[b]);
        for(int im=0;im<lsize;im+=factor){
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    // printf("%d ", img[index4D(b,im,i,j,lsize,28,28)]);
                    gg<<int(img[index4D(b,im,i,j,lsize,28,28)])<<" ";
                }
                // printf("\n");
                gg<<"\n";
            }
            // printf("\n\n");
            gg<<"\n\n";
        }
    }
    
    // free the memory after done using img array 
    cudaFree(img);
    
    // total_kernel_time += predict_NeuralNet(img, output);
    
    // for(int b = 0; b < BATCH_SIZE; b++){
    //     float max = output[b*10];
    //     int argmax = 0;
    //     for (int j = 1; j < 10; j++) {
    //         if (output[b*10 + j] > max) {
    //             max = output[b*10 + j];
    //             argmax = b*10 + j;
    //         }
    //     }

    //     if (argmax == label[b]) {
    //         matches[b]++;
    //     }
    // }
    auto endcpu = std::chrono::high_resolution_clock::now();
    
    float accuracy[BATCH_SIZE];
    // if(BATCH_SIZE>1){
    //     printf("Note: Current build gives a correct accuracy only for BATCH_SIZE=1\nFor more batches it only calculates the first layer correctly in parallel.\n");
    // }
    // for(int b = 0; b < BATCH_SIZE; b++){
    //     accuracy[b] = static_cast<float>(matches[b]) / (lsize/factor) * 100.f;
    //     printf("Accuracy batch %d: %.1f%\n", b, accuracy[b]);
    // }

    auto total_gpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(endgpu-startgpu).count());
    auto gpu_time = static_cast<float>(total_gpu_time) / (lsize/factor) / BATCH_SIZE;
    
    auto total_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(endcpu-startcpu).count());
    auto total_cpu_time = total_time - total_gpu_time;
    auto cpu_time = static_cast<float>(total_cpu_time) / (lsize/factor) / BATCH_SIZE;

    return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, gpu_time);
}

int main() {

    auto start = std::chrono::high_resolution_clock::now();
    // load batches in a vector
    std::vector<MNISTLoader> loaderx(BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; i++){
        printf("Loading batch %d...",i);
        loaderx[i] = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
        printf("loaded\n");
    }
    printf("\n");
    auto end = std::chrono::high_resolution_clock::now();
    auto batch_loading_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    printf("Batch loading time: %.2f [s] => Latency: %.4f [s/batch]\n", batch_loading_time/1000.0f, batch_loading_time/BATCH_SIZE/1000.0f);
    printf("\n");

    auto results = benchmark(loaderx);

    /*
        For some reason, printing the accuracy here always leads to "0.0%"
        Therefore it is printed in benchmark()
        (if it is printed both in benchmark and here, both print the correct accuracy)
    */
    // for(int b = 0; b < BATCH_SIZE; b++){
    //     printf("Accuracy batch %d: %.1f%\n", b, std::get<0>(results)[b]);
    // }


    printf("\n");
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));

    return 0;
}
