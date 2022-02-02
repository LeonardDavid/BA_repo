<<<<<<< HEAD
// constant variables
#define BATCH_SIZE 1
// #define NR_NEURONS 64   // DEPRECATION because it is different for each layer
#define IMG_HEIGHT 28       // original input image
#define IMG_WIDTH 28        // original input image
#define OUT_SIZE 10         // number of classes
#define NR_LAYERS 10        // number of layers
#define BINARY_WORD_SIZE 64 // size of binary word

// constant arrays 
// TODO also declared in utils.cuh -> implement a read from file
const size_t output_shape[NR_LAYERS+1][3] = {
    {64,28,28}, {64,28,28}, {64,14,14}, {64,14,14}, {64,14,14}, {64,7,7},
    {64,7,7}, {3136,1,1}, {2048,1,1}, {2048,1,1}, {10,1,1}
};

const size_t input_shape[NR_LAYERS+1][3] = {
    {64,28,28}, {64,28,28}, {64,28,28}, {64,14,14}, {64,14,14}, {64,14,14},
    {64,7,7}, {64,7,7}, {3136,1,1}, {2048,1,1}, {2048,1,1}
};

const size_t bias_size[NR_LAYERS+1] = {0, 64, 0, 0, 64, 0, 0, 0, 2048, 0, 10};
const size_t weight_size[NR_LAYERS+1] = {0, 576, 0, 0, 576, 0, 0, 0, 100352, 0, 320};

const size_t kernel_shape[NR_LAYERS+1][3] = {
    {0,0,0}, {1,3,3}, {0,2,2}, {0,0,0}, {0,3,3}, {0,2,2},
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0} 
};

const size_t strides[NR_LAYERS+1][3] = {
    {0,0}, {1,1}, {2,2}, {0,0}, {1,1}, {2,2}, 
    {0,0}, {0,0}, {0,0}, {0,0}, {0,0}
};

const size_t pads[NR_LAYERS+1][3] = {
    {0,0}, {1,1}, {0,0}, {0,0}, {1,1}, {0,0}, 
    {0,0}, {0,0}, {0,0}, {0,0}, {0,0}
};
=======
#define BATCH_SIZE 2
#define NR_NEURONS 64
>>>>>>> parent of 14d997b (introduced some constants)

#pragma once 
inline int index3D(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

#pragma once 
inline int index4D(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}