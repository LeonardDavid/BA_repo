#include <iostream>
#include <chrono>
#include <algorithm>

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

using namespace std;

auto benchmark(MNISTLoader &loader, int tsh, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[10] = {0};
#else
    float output[10] = {0};
#endif
    int factor = 1;
    int matches = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < loader.size(); i+=factor) { // loader.size()
        std::fill(output, output+10, 0);
        unsigned char * const  img = loader.images(i); //i
        int label = loader.labels(i); //i

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

        predict_NeuralNet(img, output, tsh);
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
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (loader.size()/factor);
    float accuracy = static_cast<float>(matches) / (loader.size()/factor) * 100.f;
   
    return std::make_pair(accuracy, runtime);
}

int main() {
    

    MNISTLoader loader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    // benchmark(loader);
    // cout<<"----------------------------------------------"<<endl;

    int tsh;
    // cout<<"threshold: ";
    // cin>>tsh;
    // cout<<"threshold: "<<tsh<<endl;

    auto results = benchmark(loader, tsh);

    std::cout << "Accuracy: " << results.first << " %" << std::endl;
    std::cout << "Latency: " << results.second << " [ms/elem]" << std::endl;
    
    return 0;
}
