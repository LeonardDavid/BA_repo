/*
    Run with: 
    $ make
    $ ./parz.o
*/

#include <iostream>
#include <chrono>
#include <algorithm>
#include <tuple>

#include "MNISTLoader.h"
#include "utils.h"

#ifdef BINARY
#define INPUT_FEATURE char
#include "net.hpp"
#elif INT16
#define INPUT_FEATURE int
#include "net.hpp"
#else
#define INPUT_FEATURE float
#include "net.hpp"
#endif

using namespace std;

auto benchmark(vector<MNISTLoader> &loaderx, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[BATCH_SIZE*10] = {0};
#else
    float output[BATCH_SIZE*10] = {0};
#endif

    int factor = 1;
    int matches[BATCH_SIZE] = {0};
    int const imgsize = 28*28;
    int lsize = loaderx[0].size();
    float total_kernel_time = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < lsize; i+=factor) { // i := # image
        std::fill(output, output+10*BATCH_SIZE, 0);
       
        unsigned char * img;
        img = (unsigned char*) malloc (BATCH_SIZE*imgsize);

        // load label i of corresponding image from every batch in an array
        int label[BATCH_SIZE];

        for(int b=0; b<BATCH_SIZE; b++){    // b := # batch
            for(int p=0; p<imgsize; p++){   // p := # pixel
                img[b*imgsize+p] = loaderx[b].images(i)[p];
            }
            label[b] = loaderx[b].labels(i);
        }
        
        // // display img array (remove for before)
        // for(int b=0;b<BATCH_SIZE;b++){
        //     printf("batch: %d, label %d:\n",b,label[b]);
        //     for (int i = 0; i < 28; i++)
        //     {
        //         for (int j = 0; j < 28; j++)
        //         {
        //             // img[i*28 + j] < 128 ? img[i*28 + j] = 0 : img[i*28 + j] = 255;
        //             printf("%d ", img[index3D(b,i,j,28,28)]);
        //             // printf("%d ", img[i*28+j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n\n");
        // }

        total_kernel_time += predict_NeuralNet(img, output);
        
        for(int b = 0; b < BATCH_SIZE; b++){
            float max = output[b*10];
            int argmax = 0;
            for (int j = 1; j < 10; j++) {
                if (output[b*10 + j] > max) {
                    max = output[b*10 + j];
                    argmax = j;
                }
            }

            if (argmax == label[b]) {
                matches[b]++;
            }
        }
        
    }
    auto end = std::chrono::high_resolution_clock::now();

    float accuracy[BATCH_SIZE];
    if(BATCH_SIZE>1){
        printf("Note: Current build gives a correct accuracy only for BATCH_SIZE=1\nFor more batches it only calculates the first layer correctly in parallel.\n");
    }
    for(int b = 0; b < BATCH_SIZE; b++){
        accuracy[b] = static_cast<float> (matches[b]) / (lsize/factor) * 100.f;
        printf("Accuracy batch %d: %.1f%\n", b, accuracy[b]);
    }

    auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    total_cpu_time -= total_kernel_time;
    auto cpu_time = static_cast<float>(total_cpu_time) / (lsize/factor) / BATCH_SIZE;
    auto kernel_time = static_cast<float>(total_kernel_time) / (lsize/factor) / BATCH_SIZE;

    return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time);
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
