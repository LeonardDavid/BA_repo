#include <iostream>
#include <chrono>
#include <algorithm>
#include <omp.h>

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

auto benchmark(MNISTLoader &loader, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[10] = {0};
#else
    float output[10] = {0};
#endif
    int factor = 1;
    int matches = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < loader.size(); i+=factor) {
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

        predict_NeuralNet(img, output);
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
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (loader.size()/factor);
   
    return std::make_pair(accuracy, runtime);
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

    /*
        with this implementation, the accuracy is still ~110/BATCH_SIZE %
    */

    auto startnn = std::chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(BATCH_SIZE)
    {
        int tid = omp_get_thread_num();

        start = std::chrono::high_resolution_clock::now();
        auto results = benchmark(loaderx[tid]);
        end = std::chrono::high_resolution_clock::now();

        cout << endl << "Batch " << tid << ": " << endl;
        std::cout << "Accuracy: " << results.first << " %" << std::endl;
        std::cout << "Latency: " << results.second << " [ms/elem]" << std::endl;
        cout << "Batch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0f << " [s]" << endl;
        cout<<endl;
    }
    auto endnn = std::chrono::high_resolution_clock::now();

    cout << "Total time (excl. loading): " << std::chrono::duration_cast<std::chrono::milliseconds>(endnn-startnn).count()/1000.0f << " [s]" << endl;

    return 0;
}
