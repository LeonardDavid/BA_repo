#include <iostream>
#include "kernel.h"

using namespace std;

__device__ int index_cuda(int i, int j){
    return i*SIZEX + j;
}

// int index_cuda_host(int i, int j){
//     return i*SIZEX + j;
// }

template <typename scalar_t>
__global__ void  matrix_mult_kernel(torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
                                    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> y,
                                    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z){
    
    int j = blockIdx.x * blockDim.x + threadIdx.x; //col
    int i = blockIdx.y * blockDim.y + threadIdx.y; //row
    
    if (i < SIZEX && j < SIZEY) {
        // printf("i: %d, j: %d\n",i,j);
        for (int k = 0; k < SIZEY; k++) {
            z[i][j] += x[i][k] * y[k][j];
        }
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor z){

    // int *d_x, *d_y, *d_z;
    // cudaMalloc((void **) &d_x, SIZEMUL*sizeof(int));
    // cudaMalloc((void **) &d_y, SIZEMUL*sizeof(int));
    // cudaMalloc((void **) &d_z, SIZEMUL*sizeof(int));

    // cudaMemcpy(d_x, x, (SIZEMUL*sizeof(int)), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_y, y, (SIZEMUL*sizeof(int)), cudaMemcpyHostToDevice);

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
    int64_t shape_len = x.dim();
    std::vector<int64_t> shape_original;
    for (int i = 0; i < shape_len; i++)
    {
        shape_original.push_back(x.size(i));
    }

    if (shape_len == 1)
    {
        x = x.reshape({x.size(0),1});
        y = y.reshape({x.size(0),1});
        y = y.reshape({x.size(0),1});
    }

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE);

    AT_DISPATCH_ALL_TYPES(x,type(), "matrix_mult_cuda",([&] {
        matrix_mult_kernel<scalar_t><<<numBlocks,threadsPerBlock>>>(x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                                                    y.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                                                    z.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>())
    }));

    // matrix_mult_kernel<<<numBlocks,threadsPerBlock>>>(d_x, d_y, d_z);

    // cudaMemcpy(z, d_z, (SIZEMUL*sizeof(int)), cudaMemcpyDeviceToHost);
    
    // cudaDeviceSynchronize();

    cout<<endl<<"z: "<<endl;
    // cout<<z[SIZEMUL-1]<<endl;
    for(int i=0;i<SIZEMUL;i++){
        // for(int j=0;j<SIZEY;j++){
        //     cout<<z[index(i,j)]<<" ";
        // }
        if(i%3==0)
            cout<<endl;
        cout<<z[i]<<" ";
    }

    // cudaFree(d_x);
    // cudaFree(d_y);
    // cudaFree(d_z);

    std::cout<<"Cuda done!"<<std::endl;

    return z;
}