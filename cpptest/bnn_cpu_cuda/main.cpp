/*
    Run with: 
    $ make
    $ ./cuda_net
*/

#include <iostream>
#include <chrono>
#include <algorithm>
#include <tuple>

#include "MNISTLoader.h"
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

auto benchmark(MNISTLoader &loader, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[10] = {0};
#else
    float output[10] = {0};
#endif
    int factor = 1;
    int matches = 0;
    float total_kernel_time = 0;
    auto start = std::chrono::high_resolution_clock::now();
    /*
        for better readibility in the console: 
            - comment out the for loop
            - replace loader.images(i) with loader.images(0)
            - replace loader.labels(i) with loader.labels(0)
    */
    for (unsigned int i = 0; i < loader.size(); i+=factor) {
        std::fill(output, output+10, 0);

        unsigned char * const  img = loader.images(i);
        int label = loader.labels(i);

        // unsigned char * const  img = loader.images(0);
        // int label = loader.labels(0);

        // for (int i = 0; i < 28; i++)
        // {
        //     for (int j = 0; j < 28; j++)
        //     {
        //         img[i*28 + j] < 128 ? img[i*28 + j] = 0 : img[i*28 + j] = 255;
        //         //printf("%d", img[i*28 + j]);
        //     }
        //     //printf("\n");
        //
        // }

        total_kernel_time += predict_NeuralNet(img, output);
        float max = output[0];
        int argmax = 0;
        for (int j = 1; j < 10; j++) {
            if (output[j] > max) {
                max = output[j];
                argmax = j;
            }
        }

        if (argmax == label) {
            matches++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    float accuracy = static_cast<float>(matches) / (loader.size()/factor) * 100.f;
    auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    auto cpu_time = static_cast<float>(total_cpu_time) / (loader.size()/factor);
    auto kernel_time = static_cast<float>(total_kernel_time) / (loader.size()/factor);

    return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time);
}

int main() {
    MNISTLoader loader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

    auto results = benchmark(loader);
    std::cout << "Accuracy: " << std::get<0>(results) << " %" << std::endl;
    // the difference between total_cpu_time-total_kernel_time is the loading time
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));

    return 0;
}
