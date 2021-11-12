/*
Run with: 
$ make
$ ./cuda_net
*/

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

/*
    the following are included in cuda_kernel.h and should be put later in another utilities header to also be available in main
*/
#define BATCH_SIZE 4

inline int index3D(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

inline int index4D(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}

auto benchmark(MNISTLoader &loader0, MNISTLoader &loader1, MNISTLoader &loader2, MNISTLoader &loader3, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[BATCH_SIZE*10] = {0};
#else
    float output[BATCH_SIZE*10] = {0};
#endif
    int factor = 1;
    // int matches = 0;
    int lsize = loader0.size();
    auto start = std::chrono::high_resolution_clock::now();
    // for (unsigned int i = 0; i < lsize; i+=factor) {
        std::fill(output, output+10*BATCH_SIZE, 0);
        
        // load (flattened) image i of every batch in 1 array, at a distance of imgsize
        int imgsize = 28*28;
        unsigned char * img0 = loader0.images(0); //i
        unsigned char * img1 = loader1.images(0); //i
        unsigned char * img2 = loader2.images(0); //i
        unsigned char * img3 = loader3.images(0); //i

        // make all of these declarations pretier later, maybe with pointers
        unsigned char img[BATCH_SIZE*imgsize];
        for(int b=0;b<BATCH_SIZE;b++){
            for(int i=0;i<imgsize;i++){
                switch(b){
                    case 0: img[b*imgsize+i] = img0[i];
                            break;
                    case 1: img[b*imgsize+i] = img1[i];
                            break;
                    case 2: img[b*imgsize+i] = img2[i];
                            break;
                    case 3: img[b*imgsize+i] = img3[i];
                            break;
                }
            }
        }
        
        // load label i of corresponding image from every batch in an array
        int label[4];
        label[0] = loader0.labels(0); //i 
        label[1] = loader1.labels(0); //i 
        label[2] = loader2.labels(0); //i 
        label[3] = loader3.labels(0); //i  

        // for(int b=0;b<BATCH_SIZE;b++){
        //     printf("batch: %d, label %d:\n",b,label[b]);
        //     for (int i = 0; i < 28; i++)
        //     {
        //         for (int j = 0; j < 28; j++)
        //         {
        //             // img[i*28 + j] < 128 ? img[i*28 + j] = 0 : img[i*28 + j] = 255;
        //             printf("%d ", img[index3D(b,i,j,28,28)]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n\n");
        // }
        
        predict_NeuralNet(img, output);

        /* 
            skip matching for now
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
    // }
    auto end = std::chrono::high_resolution_clock::now();
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (loader0.size()/factor);
    // float accuracy = static_cast<float>(matches) / (loader.size()/factor) * 100.f;
    // return std::make_pair(accuracy, runtime);
    return 0;

}

int main() {

    // later initialize all batches in an array of size BATCH_SIZE
    MNISTLoader loader0("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
            loader1("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
            loader2("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"), 
            loader3("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

    auto results = benchmark(loader0, loader1, loader2, loader3);

    return 0;
}
