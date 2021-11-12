#include <iostream>
#include <chrono>
#include <algorithm>
#include <omp.h>

#include "netW.hpp"

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

auto benchmark(MNISTLoader &loader, int thread, int matches, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[10] = {0};
#else
    float output[10] = {0};
#endif
    int factor = 1;
    auto startnn = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < loader.size(); i+=factor) { // loader.size()
        // std::fill(output, output+10, 0);
        unsigned char * const  img = loader.images(i);
        int label = loader.labels(i);

        /* 
        very ugly solution, especially in net.hpp and net.cpp
        maybe sometime can come up with a better one
        */

        switch(thread){
            case 0: matches += predict_NeuralNet0(img, thread, label, output);
                    break;
            case 1: matches += predict_NeuralNet1(img, thread, label, output);
                    break;
            case 2: matches += predict_NeuralNet2(img, thread, label, output);
                    break;
            case 3: matches += predict_NeuralNet3(img, thread, label, output);
                    break;
            case 4: matches += predict_NeuralNet4(img, thread, label, output);
                    break;
        }

        // for (int i = 0; i < 28; i++)
        // {
        //     for (int j = 0; j < 28; j++)
        //     {
        //         // img[i*28 + j] < 128 ? img[i*28 + j] = 0 : img[i*28 + j] = 255;
        //         printf("%d ", img[i*28 + j]);
        //     }
        //     printf("\n");
        
        // }
        // printf("\n");

        // #pragma omp task
        // matches += predict_NeuralNet(img, thread, label, output); //, output
        // #pragma omp taskwait

        /*
            put the following lines of code in net.cpp in case of shared variables
            (although I'm pretty sure it works fine the OG way)
        */
        // float max = output[0];
        // int argmax = 0;
        // for (int j = 1; j < 10; j++) {
        //     if (output[j] > max) {
        //         max = output[j];
        //         argmax = j;
        //     }
        // }

        // if (argmax == label) {
        //     matches++;
        // }
    }
    auto endnn = std::chrono::high_resolution_clock::now();
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(endnn-startnn).count()) / (loader.size()/factor);
    float accuracy = static_cast<float>(matches) / (loader.size()/factor) * 100.f;
    return std::make_pair(accuracy, runtime);
}

int main() {
    // BATCH_SIZE and MAX_THREADS defined in net.hpp
    
    MNISTLoader loader0("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
            loader1("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
            loader2("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
            loader3("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
            loader4("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"); 

    /*  
        *to use MNISTLoader arrays*
        - For some reason, batches have to be declared in a separate for loop
        AND batch 0 cannot be used -> leads to segmentation fault.
        threfore, declare batch size n+1 and benchmark from 1 to n+1

        - Also, at the end an error appears "double free or corruption (out) - Aborted"

        -- solution is above this comment, very ugly for now.
    */
    // MNISTLoader loaderx[BATCH_SIZE];
    // for(int i=0;i<BATCH_SIZE;i++){
    //     cout<<"Loading batch "<<i<<"...";
    //     loaderx[i] = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    //     cout<<"loaded"<<endl;
    // }    

    /* 
        very ugly solution, especially in net.hpp and net.cpp
        maybe sometime can come up with a better one
    */
    pair<float,float> results;
    int matches;
    auto startx = std::chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(BATCH_SIZE) private(matches)
    {
        int tid = omp_get_thread_num();
        
        auto start = std::chrono::high_resolution_clock::now();
        results = benchmark(loader0,tid,0);
        auto end = std::chrono::high_resolution_clock::now();

        cout << endl << "Batch " << tid << ": " << endl;
        cout << "Accuracy: " << results.first << " %" << endl;
        cout << "Latency: " << results.second << " [ms/elem]" << endl;
        cout << "Batch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0f << " [s]" << endl;
        cout<<endl;

        /*
            tried implementing benchmark directly in here, but the function is still not executed in each thread only once
        */
        
        // float output[10] = {0};
        // int factor = 1;
        // matches = 0;
        // auto startnn = std::chrono::high_resolution_clock::now();
        // for (unsigned int i = 0; i < loader0.size(); i+=factor) { 
        //     // std::fill(output, output+10, 0);
        //     unsigned char * const  x = loader0.images(i);
        //     int label = loader0.labels(i);

        //     matches += predict_NeuralNet(); 
        // }
        
        // auto endnn = std::chrono::high_resolution_clock::now();
        // auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(endnn-startnn).count()) / (loader0.size()/factor);
        // float accuracy = static_cast<float>(matches) / (loader0.size()/factor) * 100.f;

        // cout << endl << "Batch " << tid << ": " << endl;
        // cout << "Accuracy: " << accuracy << " %" << endl;
        // cout << "Latency: " << runtime << " [ms/elem]" << endl;
        // cout << "Batch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endnn-startnn).count()/1000.0f << " [s]" << endl;

    }
    auto endx = std::chrono::high_resolution_clock::now();
    cout<<endl;
    cout << "Total time (excl. loading): " << std::chrono::duration_cast<std::chrono::milliseconds>(endx-startx).count()/1000.0f << " [s]" << endl;


    

    /*
        tried parallel for
    */
    // // look into "mutating parameters in functions" (via https://stackoverflow.com/a/31914880)
    // auto startx = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel num_threads(BATCH_SIZE-1) //private(layer_1_output, layer_2_output, layer_3_output, layer_4_output, layer_5_output, layer_6_output, layer_8_output, layer_9_output, layer_10_output)
    // {
    //     int tid = omp_get_thread_num();
        
    //     // auto results = benchmark(loader0,i,matches);
    //     pair<float,float> results;
    //     int matches;
    //     #pragma omp for private(matches, results)
    //     for(int i=1;i<BATCH_SIZE;i++){
    //         matches = 0;
    //         auto start = std::chrono::high_resolution_clock::now();
    //         results = benchmark(loaderx[i], i, matches);
    //         auto end = std::chrono::high_resolution_clock::now();

    //         cout << endl << "Batch " << tid << ": " << endl;
    //         cout << "Accuracy: " << results.first << " %" << endl;
    //         cout << "Latency: " << results.second << " [ms/elem]" << endl;
    //         cout << "Batch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0f << " [s]" << endl;
    //         cout<<endl;
    //     }
    //     // switch(i){
    //     //     case 0: results = benchmark(loader0, i, matches);
    //     //             break;
    //     //     case 1: results = benchmark(loader1, i, matches);
    //     //             break;
    //     //     case 2: results = benchmark(loader2, i, matches);
    //     //             break;
    //     //     case 3: results = benchmark(loader3, i, matches);
    //     //             break;
    //     //     case 4: results = benchmark(loader4, i, matches);
    //     //             break;
    //     // }
    
    //     }
    // auto endx = std::chrono::high_resolution_clock::now();
    // cout<<endl;
    // cout << "Total time (excl. loading): " << std::chrono::duration_cast<std::chrono::milliseconds>(endx-startx).count()/1000.0f << " [s]" << endl;

    /*
        initial implementation which should have worked, but life is not that beautiful (:
    */

    // auto startx = std::chrono::high_resolution_clock::now();
    // pair<float,float> results;
    // #pragma omp parallel num_threads(BATCH_SIZE-1) private(results)
    // {
    //     int tid = omp_get_thread_num();
    
    //     auto start = std::chrono::high_resolution_clock::now();
    //     results = benchmark(loaderx[tid],tid,0);
    //     auto end = std::chrono::high_resolution_clock::now();

    //     cout <<endl<<"Batch "<<tid<<": "<<endl;
    //     cout << "Accuracy: " << results.first << " %" << endl;
    //     cout << "Latency: " << results.second << " [ms/elem]" << endl;
    //     cout << "Batch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0f << " [s]" << endl;
    //     cout<<endl;
    // }
    // auto endx = std::chrono::high_resolution_clock::now();
    // cout<<endl;
    // cout << "Total time (excl. loading): " << std::chrono::duration_cast<std::chrono::milliseconds>(endx-startx).count()/1000.0f << " [s]" << endl;

}