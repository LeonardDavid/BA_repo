#define SIZEX 1024
#define SIZEY 1024
#define SIZEMUL SIZEX*SIZEY

#define BLKXSIZE 32
#define BLKYSIZE 32
// max number of threads/block: 32x32 = 1024

#define GRIDXSIZE SIZEMUL/SIZEX
#define GRIDYSIZE SIZEMUL/SIZEY

int matrix_mult_cuda(int *x, int *y, int *z);

inline int index(int i, int j){
    return i*SIZEX + j;
}