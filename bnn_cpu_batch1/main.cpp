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

// I. multiple benchmarks in main()
auto benchmark(MNISTLoader &loader, bool verbose = false) {
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
        unsigned char * const  img = loader.images(i);
        int label = loader.labels(i);

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
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (loader.size()/factor);
    float accuracy = static_cast<float>(matches) / (loader.size()/factor) * 100.f;
    return std::make_pair(accuracy, runtime);
}

// I. multiple benchmarks in main()
int main(){

    MNISTLoader loader0("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
                loader1("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
                loader2("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
                loader3("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
                loader4("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");  

    // loader0 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    // loader1 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    // loader2 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    // loader3 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    // loader4 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"); 
    /*  
        - For some reason, batches have to be declared in a separate for loop
        AND batch 0 cannot be used -> leads to segmentation fault.
        threfore, declare batch size n+1 and benchmark from 1 to n+1

        - Also, at the end an error appears "double free or corruption (out) - Aborted"
    */
    // for(int i=0;i<1;i++){
    //     cout<<"Loading batch "<<i<<"...";
    //     switch(i){
    //         case 0: loader0 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    //                 break;
    //         case 1: loader1 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    //                 break;
    //         case 2: loader2 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    //                 break;
    //         case 3: loader3 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    //                 break;
    //         case 4: loader4 = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    //                 break;
    //     }
    //     cout<<"loaded"<<endl;
    // }    

    auto startx = std::chrono::high_resolution_clock::now();
    // for(int i=0;i<1;i++){
        // cout<<"Batch "<<i<<": "<<endl;

        // pair<float,float> results;
        auto start = std::chrono::high_resolution_clock::now();
        // switch(i){
        //     case 0: results = benchmark(loader0);
        //             break;
        //     case 1: results = benchmark(loader1);
        //             break;
        //     case 2: results = benchmark(loader2);
        //             break;
        //     case 3: results = benchmark(loader3);
        //             break;
        //     case 4: results = benchmark(loader4);
        //             break;
        // }
        auto results = benchmark(loader0);
        auto end = std::chrono::high_resolution_clock::now();

        cout << "Accuracy: " << results.first << " %" << endl;
        cout << "Latency: " << results.second << " [ms/elem]" << endl;
        cout << "Batch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0f << " [s]" << endl;
        cout<<endl;
    // }
    auto endx = std::chrono::high_resolution_clock::now();
    cout<<endl;
    cout << "Total time (excl. loading): " << std::chrono::duration_cast<std::chrono::milliseconds>(endx-startx).count()/1000.0f << " [s]" << endl;

    // MNISTLoader loaderx[6];   
    // /*  
    //     - For some reason, batches have to be declared in a separate for loop
    //     AND batch 0 cannot be used -> leads to segmentation fault.
    //     threfore, declare batch size n+1 and benchmark from 1 to n+1

    //     - Also, at the end an error appears "double free or corruption (out) - Aborted"
    // */
    // for(int i=0;i<6;i++){
    //     cout<<"Loading batch "<<i<<"...";
    //     loaderx[i] = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    //     cout<<"loaded"<<endl;
    // }    

    // auto startx = std::chrono::high_resolution_clock::now();
    // for(int i=1;i<6;i++){
    //     cout<<"Batch "<<i<<": "<<endl;
    
    //     auto start = std::chrono::high_resolution_clock::now();
    //     auto results = benchmark(loaderx[i]);
    //     auto end = std::chrono::high_resolution_clock::now();

    //     cout << "Accuracy: " << results.first << " %" << endl;
    //     cout << "Latency: " << results.second << " [ms/elem]" << endl;
    //     cout << "Batch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0f << " [s]" << endl;
    //     cout<<endl;
    // }
    // auto endx = std::chrono::high_resolution_clock::now();
    // cout<<endl;
    // cout << "Total time (excl. loading): " << std::chrono::duration_cast<std::chrono::milliseconds>(endx-startx).count()/1000.0f << " [s]" << endl;
}

// // II. multiple predict_NeuralNet() in benchmark()
// void benchmark(MNISTLoader &loader, bool verbose = false) {
// #if defined BINARY || defined INT16
//     int output[10] = {0};
// #else
//     float output[10] = {0};
// #endif
//     auto startx = std::chrono::high_resolution_clock::now();
//     for(int j=0;j<5;j++){
//         printf("Batch: %d\n",j);
//         auto start = std::chrono::high_resolution_clock::now();
//         int factor = 1;
//         int matches = 0;
//         for (unsigned int i = 0; i < loader.size(); i+=factor) {

//             std::fill(output, output+10, 0);
//             unsigned char * const  img = loader.images(i);
//             int label = loader.labels(i);

//             // for (int i = 0; i < 28; i++)
//             // {
//             //     for (int j = 0; j < 28; j++)
//             //     {
//             //         img[i*28 + j] < 128 ? img[i*28 + j] = 0 : img[i*28 + j] = 255;
//             //         //printf("%d", img[i*28 + j]);
//             //     }
//             //     //printf("\n");
//             //
//             // }

//             predict_NeuralNet(img, output);
//             float max = output[0];
//             int argmax = 0;
//             for (int j = 1; j < 10; j++) {
//                 if (output[j] > max) {
//                     max = output[j];
//                     argmax = j;
//                 }
//             }

//             if (argmax == label) {
//                 matches++;
//             }
//         }

//         auto end = std::chrono::high_resolution_clock::now();
//         float accuracy = static_cast<float>(matches) / (loader.size()/factor) * 100.f;
//         auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (loader.size()/factor);

//         printf("Accuracy: %.2f%\n",accuracy);
//         printf("Latency: %.4f [ms/elem]\n", runtime);
//         printf("Batch time: %.3f [s]\n\n", (std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0f));
//     }
//     auto endx = std::chrono::high_resolution_clock::now();
//     printf("Total time (excl. loading): %.3fs\n",  (std::chrono::duration_cast<std::chrono::milliseconds>(endx-startx).count()/1000.0f));
//     // return std::make_pair(accuracy, runtime);
// }

// // II. multiple predict_NeuralNet() in benchmark()
// int main() {
    
//     MNISTLoader loader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

//     benchmark(loader);

//     return 0;
// }

// // III. original 
// auto benchmark(MNISTLoader &loader, bool verbose = false) {
// #if defined BINARY || defined INT16
//     int output[10] = {0};
// #else
//     float output[10] = {0};
// #endif
//     int factor = 1;
//     int matches = 0;
//     auto start = std::chrono::high_resolution_clock::now();
//     for (unsigned int i = 0; i < loader.size(); i+=factor) {
//         std::fill(output, output+10, 0);
//         unsigned char * const  img = loader.images(i);
//         int label = loader.labels(i);

//         // for (int i = 0; i < 28; i++)
//         // {
//         //     for (int j = 0; j < 28; j++)
//         //     {
//         //         img[i*28 + j] < 128 ? img[i*28 + j] = 0 : img[i*28 + j] = 255;
//         //         //printf("%d", img[i*28 + j]);
//         //     }
//         //     //printf("\n");
//         //
//         // }

//         predict_NeuralNet(img, output);
//         float max = output[0];
//         int argmax = 0;
//         for (int j = 1; j < 10; j++) {
//             if (output[j] > max) {
//                 max = output[j];
//                 argmax = j;
//             }
//         }

//         if (argmax == label) {
//             matches++;
//         }
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (loader.size()/factor);
//     float accuracy = static_cast<float>(matches) / (loader.size()/factor) * 100.f;
//     return std::make_pair(accuracy, runtime);
// }

// // III. original
// int main() {
//     MNISTLoader loader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

//     auto results = benchmark(loader);
//     std::cout << "Accuracy: " << results.first << " %" << std::endl;
//     std::cout << "Latency: " << results.second << " [ms/elem]" << std::endl;

//     return 0;
// }