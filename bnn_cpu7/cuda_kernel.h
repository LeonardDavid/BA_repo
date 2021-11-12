// set a 3D volume
// To compile it with nvcc execute: nvcc -O2 -o cuda_matadd.out cuda_matadd.cu

 // define the data set size (cubic volume)
 #define DATAXSIZE 14
 #define DATAYSIZE 14
 #define DATAZSIZE 64
 #define DATAMUL DATAXSIZE*DATAYSIZE*DATAZSIZE
 // define the chunk sizes that each threadblock will work on
 // change them and eventually check performance by timing different sizes
 #define BLKXSIZE 32
 #define BLKYSIZE 32
//  #define BLKZSIZE 4
#define GRIDXSIZE DATAMUL / BLKXSIZE
#define GRIDYSIZE DATAMUL / BLKYSIZE

int predict_NeuralNet_Cuda(unsigned char * const x, float * pred);