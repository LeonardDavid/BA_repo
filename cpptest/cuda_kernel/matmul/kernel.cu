#include <iostream>
#include "kernel.h"

using namespace std;

__device__ int index_cuda(int i, int j){
    return i*SIZEX + j;
}

// int index_cuda_host(int i, int j){
//     return i*SIZEX + j;
// }

__global__ void  matrix_mult_kernel(int *d_x, int *d_y, int *d_z){
    
    int j = blockIdx.x * blockDim.x + threadIdx.x; //col
    int i = blockIdx.y * blockDim.y + threadIdx.y; //row
    
    if (i < SIZEX && j < SIZEY) {
        // printf("i: %d, j: %d\n",i,j);
        for (int k = 0; k < SIZEY; k++) {
            d_z[index_cuda(i,j)] += d_x[index_cuda(i,k)] * d_y[index_cuda(k,j)];
        }
    }
}

int matrix_mult_cuda(int *x, int *y, int *z){

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE);

    int *d_x, *d_y, *d_z;
    cudaMalloc((void **) &d_x, SIZEMUL*sizeof(int));
    cudaMalloc((void **) &d_y, SIZEMUL*sizeof(int));
    cudaMalloc((void **) &d_z, SIZEMUL*sizeof(int));

    cudaMemcpy(d_x, x, (SIZEMUL*sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, (SIZEMUL*sizeof(int)), cudaMemcpyHostToDevice);

    // cout<<endl<<"x: "<<endl;
    // cout<<x[SIZEMUL-1]<<endl;
    // for(int i=0;i<SIZEX;i++){
    //     for(int j=0;j<SIZEY;j++){
    //         cout<<x[index(i,j)]<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl<<"y: "<<endl;
    // cout<<y[SIZEMUL-1]<<endl;
    // for(int i=0;i<SIZEX;i++){
    //     for(int j=0;j<SIZEY;j++){
    //         cout<<y[index(i,j)]<<" ";
    //     }
    //     cout<<endl;
    // }

    matrix_mult_kernel<<<numBlocks,threadsPerBlock>>>(d_x, d_y, d_z);

    cudaMemcpy(z, d_z, (SIZEMUL*sizeof(int)), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // cout<<endl<<"z: "<<endl;
    // cout<<z[SIZEMUL-1]<<endl;
    // for(int i=0;i<SIZEX;i++){
    //     for(int j=0;j<SIZEY;j++){
    //         cout<<z[index(i,j)]<<" ";
    //     }
    //     cout<<endl;
    // }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    std::cout<<"Cuda done!"<<std::endl;

    return 0;
}