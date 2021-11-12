/*
    run with:
    $ g++ -fopenmp -o hello hello_world.cpp
    (optional) $  export OMP_NUM_THREADS=6 
    $ ./hello
*/

#include <iostream>
#include <stdio.h>
#include <omp.h>

using namespace std;

int main(void)
{

    // int numthreads = 12;
    // #pragma omp parallel num_threads(numthreads)
    // {
    //     int tid = omp_get_thread_num();
    //     cout<<tid<<": Hello World!"<<endl;
    //     cout<<"cya"<<endl;
    // }

    int numthreads = omp_get_max_threads();
    // cout<<"num: "<<numthreads<<endl;
    #pragma omp parallel for
    for(int i=0;i<numthreads;i++){
        cout<<i<<": Hello world!"<<endl;
        cout<<"cya"<<endl;
    }


}