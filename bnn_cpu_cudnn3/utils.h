/*
    Use batch sizes which divide evenly with 10000: 1, 2, 5, 10 ... Just to be sure
*/
#define BATCH_SIZE 4

#pragma once 
inline int index3D(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

#pragma once 
inline int index4D(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}