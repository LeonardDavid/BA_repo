// set a 3D volume
// To compile it with nvcc execute: nvcc -O2 -o cuda_matadd.out cuda_matadd.cu

 // define the data set size (cubic volume)
 #define DATAXSIZE 2//64
 #define DATAYSIZE 2//14
 #define DATAZSIZE 2//14
 // define the chunk sizes that each threadblock will work on
 // change them and eventually check performance by timing different sizes
 #define BLKXSIZE 32
 #define BLKYSIZE 4
 #define BLKZSIZE 4

int matadd(signed short (*a)[DATAYSIZE][DATAXSIZE]);