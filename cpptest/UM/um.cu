// #include <iostream>
// #include <math.h>
// #include <vector>
 
// using namespace std;

// static constexpr signed char layer_1_weight[3][3][1][64] = {{{{-1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1}}, {{-1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1}}, {{1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1}}}, {{{-1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1}}, {{1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1}}, {{1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1}}}, {{{-1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1}}, {{1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1}}, {{1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1}}}};
// static float layer_1_bias[64] = {0.13352546095848083, -0.12036499381065369, 0.24544695019721985, -0.02730405330657959, 0.20062103867530823, -0.1533348262310028, -0.31367605924606323, 0.32493308186531067, -0.020914167165756226, 0.011539816856384277, -0.23307013511657715, -0.2848169803619385, 0.2976394593715668, 0.28303393721580505, 0.12131848931312561, -0.030313611030578613, 0.18565580248832703, -0.125528022646904, -0.26768624782562256, -0.2897748351097107, -0.04102277755737305, -0.12822549045085907, 0.24084368348121643, -0.20761868357658386, -0.18164309859275818, -0.18892285227775574, -0.006639987230300903, 0.22329166531562805, -0.1719619482755661, -0.03761908411979675, -0.20930270850658417, -0.08441802859306335, -0.31097275018692017, 0.13221105933189392, -0.2714305520057678, 0.2563836872577667, -0.13719308376312256, -0.22940699756145477, 0.16428956389427185, -0.29203593730926514, -0.008505433797836304, -0.14707009494304657, 0.09783756732940674, 0.2632528245449066, -0.323319673538208, -0.13313846290111542, -0.057888299226760864, 0.2701054513454437, 0.08538663387298584, 0.3096083104610443, 0.08091691136360168, -0.09096133708953857, -0.25001415610313416, 0.07449612021446228, -0.11684815585613251, -0.18291668593883514, 0.3229244649410248, -0.29078277945518494, -0.07493731379508972, -0.1492733657360077, 0.25669369101524353, -0.31583523750305176, 0.20943573117256165, 0.2097615897655487};

// // CUDA kernel to add elements of two arrays
// __global__ void add(float *x, float *y)
// {
//     int tid = threadIdx.x; // = h

//     if(tid<64){
//         printf("%f ", x[tid]);
//     }
        
//     //   if(index == 0 || index == 63){
//     //       printf("device: x[%d]: %f, y[%d]: %f\n",index, x[index],index,y[index]);
//     //   }
//     //   for (int i = index; i < n; i += stride)
//     //     y[i] = x[i] + y[i];
// }
 
// int main(void)
// {
//     // unsigned char *d_cuda_layer_0_output; 
//     // float *d_layer_1_bias; 
//     // signed char *d_cuda_layer_1_weight; 
//     // float *d_cuda_layer_1_output; 

//     // cudaMallocManaged(&d_cuda_layer_0_output, 784*sizeof(unsigned char));
//     // cudaMallocManaged(&d_layer_1_bias, 64*sizeof(float));
//     // cudaMallocManaged(&d_cuda_layer_1_weight, 576*sizeof(signed char));
//     // cudaMallocManaged(&d_cuda_layer_1_output, 50176*sizeof(float));

//     // d_cuda_layer_0_output = cuda_layer_0_output;
//     // d_layer_1_bias = layer_1_bias;
//     // d_cuda_layer_1_weight = cuda_layer_1_weight;
   
//     float *y;
//     float *x;
    
//     // Allocate Unified Memory -- accessible from CPU or GPU
//     cudaMallocManaged((void **)&x, 64*sizeof(float));
//     cudaMallocManaged((void **)&y, 64*sizeof(float));
    
//     // initialize x and y arrays on the host
//     for (int i = 0; i < 64; i++) {
//         y[i] = 2.0f;
//     }

//     x = layer_1_bias;
//     //   std::copy(std::begin(x), std::end(x), std::begin(layer_1_bias));
//     //   for(int i=0;i<64;i++){
//     //       x[i] = layer_1_bias[i];
//     //   }
//     cout<<"host: "<<endl;
//     cout<<x[0]<<" "<<x[63]<<endl;
//     cout<<y[0]<<" "<<y[63]<<endl;
//     cout<<endl;

//     // Launch kernel on 1M elements on the GPU
//     const int BLKXSIZE = 64;
//     const int BLKYSIZE = 1;
//     const int GRIDXSIZE = 1;
//     const int GRIDYSIZE = 1;
//     const int GRIDZSIZE = 1;

//     const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE); // the 2 for loops 28 iterations each
//     const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

//     add<<<numBlocks, threadsPerBlock>>>(x, y);
 
//     // Wait for GPU to finish before accessing on host
//     cudaDeviceSynchronize();
    
//     // Check for errors (all values should be 3.0f)
//     float maxError = 0.0f;
//     for (int i = 0; i < 64; i++)
//         maxError = fmax(maxError, fabs(y[i]-3.0f));
//     std::cout << "Max error: " << maxError << std::endl;
    
//     // Free memory
//     cudaFree(x);
//     cudaFree(y);
 
//     return 0;
// }

#include <string.h>
#include <stdio.h>

struct DataElement
{
  char *name;
  int value;
};

__global__ 
void Kernel(DataElement *elem) {
  printf("On device: name=%s, value=%d\n", elem->name, elem->value);

  elem->name[0] = 'd';
  elem->value++;
}

void launch(DataElement *elem) {
  Kernel<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

int main(void)
{
  DataElement *e;
  cudaMallocManaged((void**)&e, sizeof(DataElement));

  e->value = 10;
  cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1) );
  strcpy(e->name, "hello");

  launch(e);

  printf("On host: name=%s, value=%d\n", e->name, e->value);

  cudaFree(e->name);
  cudaFree(e);

  cudaDeviceReset();
}