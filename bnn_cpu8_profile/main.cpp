/*
    Run with: 
    $ make
    $ ./cuda_net.o
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
    float l1_time = 0, l2_time = 0, l3_time = 0, l4_time = 0, l5_time = 0, l6_time = 0, l8_time = 0, l9_time = 0, l10_time = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < loader.size(); i+=factor) { // i < loader.size()
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

        float a,b,c,d,e,f,g,h,ii;
        std::tie(a,b,c,d,e,f,g,h,ii) = predict_NeuralNet(img, output);
        l1_time += a;
        l2_time += b;
        l3_time += c;
        l4_time += d;
        l5_time += e;
        l6_time += f;
        l8_time += g;
        l9_time += h;
        l10_time += ii;

        float max = output[0];
        int argmax = 0;
        for (int j = 0; j < 10; j++) {
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

    return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time, l1_time,l2_time,l3_time,l4_time,l5_time,l6_time,l8_time,l9_time,l10_time);
}

int main() {
    MNISTLoader loader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

    auto results = benchmark(loader);
    std::cout << "Accuracy: " << std::get<0>(results) << " %" << std::endl;
    // the difference between total_cpu_time-total_kernel_time is the loading time
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));

    float l1_time = std::get<5>(results)/1000000.0f; // ns / 1e6 -> ms
    float l2_time = std::get<6>(results)/1000000.0f; // ns / 1e6 -> ms
    float l3_time = std::get<7>(results)/1000000.0f; // ns / 1e6 -> ms
    float l4_time = std::get<8>(results)/1000000.0f; // ns / 1e6 -> ms
    float l5_time = std::get<9>(results)/1000000.0f; // ns / 1e6 -> ms
    float l6_time = std::get<10>(results)/1000000.0f; // ns / 1e6 -> ms
    float l8_time = std::get<11>(results)/1000000.0f; // ns / 1e6 -> ms
    float l9_time = std::get<12>(results)/1000000.0f; // ns / 1e6 -> ms
    float l10_time = std::get<13>(results)/1000000.0f; // ns / 1e6 -> ms

    float sum_l = l1_time + l2_time + l3_time + l4_time + l5_time + l6_time + l8_time + l9_time + l10_time;

    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 1 time:", l1_time, "Ratio:", (l1_time/sum_l)*100);
    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 2 time:", l2_time, "Ratio:", (l2_time/sum_l)*100);
    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 3 time:", l3_time, "Ratio:", (l3_time/sum_l)*100);
    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 4 time:", l4_time, "Ratio:", (l4_time/sum_l)*100);
    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 5 time:", l5_time, "Ratio:", (l5_time/sum_l)*100);
    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 6 time:", l6_time, "Ratio:", (l6_time/sum_l)*100);
    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 8 time:", l8_time, "Ratio:", (l8_time/sum_l)*100);
    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 9 time:", l9_time, "Ratio:", (l9_time/sum_l)*100);
    printf("%-15s %-10.2f [ms], %-10s %-5.2f%\n", "Layer 10 time:", l10_time, "Ratio:", (l10_time/sum_l)*100);
    printf("\n");
    printf("%-15s %.2f [ms]\n", "Total time:", sum_l);

    return 0;
}
