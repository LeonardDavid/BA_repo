 // for cuda error checking
 #define cudaCheckErrors(msg) \
 do { \
     cudaError_t __err = cudaGetLastError(); \
     if (__err != cudaSuccess) { \
         fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
             msg, cudaGetErrorString(__err), \
             __FILE__, __LINE__); \
         fprintf(stderr, "*** FAILED - ABORTING\n"); \
         return 1; \
     } \
 } while (0)

// constant arrays to be used in device code.
// TODO also declared in utils.h -> implement a read from file
// __device__ const size_t output_shape[NR_LAYERS+1][3] = {
//     {64,28,28}, {64,28,28}, {64,14,14}, {64,14,14}, {64,14,14}, {64,7,7},
//     {64,7,7}, {3136,1,1}, {2048,1,1}, {2048,1,1}, {10,1,1}
// };

// __device__ const size_t input_shape[NR_LAYERS+1][3] = {
//     {64,28,28}, {64,28,28}, {64,28,28}, {64,14,14}, {64,14,14}, {64,14,14},
//     {64,7,7}, {64,7,7}, {3136,1,1}, {2048,1,1}, {2048,1,1}
// };

// __device__ const size_t bias_size[NR_LAYERS+1] = {0, 64, 0, 0, 64, 0, 0, 0, 2048, 0, 10};
// __device__ const size_t weight_size[NR_LAYERS+1] = {0, 576, 0, 0, 576, 0, 0, 0, 100352, 0, 320};

// __device__ const size_t kernel_shape[NR_LAYERS+1][3] = {
//     {0,0,0}, {1,3,3}, {0,2,2}, {0,0,0}, {0,3,3}, {0,2,2},
//     {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0} 
// };

// __device__ const size_t strides[NR_LAYERS+1][3] = {
//     {0,0}, {1,1}, {2,2}, {0,0}, {1,1}, {2,2}, 
//     {0,0}, {0,0}, {0,0}, {0,0}, {0,0}
// };

// __device__ const size_t pads[NR_LAYERS+1][3] = {
//     {0,0}, {1,1}, {0,0}, {0,0}, {1,1}, {0,0}, 
//     {0,0}, {0,0}, {0,0}, {0,0}, {0,0}
// };

// TODO https://codeyarns.com/tech/2011-03-14-cuda-common-function-for-both-host-and-device-code.html#:~:text=If%20a%20function%20needs%20to,host__%20__device__%20.

__device__ int index3D_cuda(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

__device__ int index4D_cuda(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}

// use the second GPU on Uni-server because the first is used most of the time
void setUniGPU(){
    int devices;
    cudaGetDeviceCount(&devices);
    if(devices>1){
        cudaSetDevice(1);
    }
}