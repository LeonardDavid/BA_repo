inline int index3D(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

inline int index4D(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}

float layer1_conv_cuda(unsigned char * const x, float * layer);

float layer2_maxpool_cuda(float * layer1, float * layer2);

float layer3_step_cuda(float * layer1, unsigned long long * layer2);

float layer4_conv_cuda(unsigned long long * layer1, signed short * layer2);

float layer5_maxpool_cuda(signed short * layer1, signed short * layer2);

float layer8_gemm_cuda(unsigned long long * layer1, signed short * layer2);

float layer10_gemm_cuda(unsigned long long * layer1, signed short * layer2);
