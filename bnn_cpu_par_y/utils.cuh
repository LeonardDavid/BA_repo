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

__device__ int index3D_cuda(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

__device__ int index4D_cuda(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}

/*
    atomicMax for float does not exist. This is the workaround:

    The function is community written and taken from here:
    https://stackoverflow.com/a/17401122/13184775

    For float atomicMin replace fmaxf with fminf
*/

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/*
    short atomicMax does not exist anywhere.
*/

// __device__ short atomicMaxShort(short* address, short val){

// }

/*
    There are no CUDA atomic intrinsics for unsigned short and unsigned char data types, or any data type smaller than 32 bits

    The function is comunity written and taken from here:
    https://forums.developer.nvidia.com/t/how-to-use-atomiccas-to-implement-atomicadd-short-trouble-adapting-programming-guide-example/22712
*/
__device__ short atomicAddShort(short* address, short val){

    unsigned int *base_address = (unsigned int *) ((char *)address - ((size_t)address & 2));	//tera's revised version (showtopic=201975)

    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;

    unsigned int long_old = atomicAdd(base_address, long_val);

    if((size_t)address & 2) {

        return (short)(long_old >> 16);

    } else {

        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

        if (overflow)

            atomicSub(base_address, overflow);

        return (short)(long_old & 0xffff);

    }

}