#include <iostream>
#include <algorithm>
#include <chrono>
#include "kernel_main.h"


using namespace std;

// int index(int i, int j){
//     return i*SIZEX + j;
// }

int main(){

    cout<<"Allocating memory"<<endl;    
    int* a = (int *)malloc(sizeof(int)*SIZEMUL);
    int* b = (int *)malloc(sizeof(int)*SIZEMUL);
    int* c = (int *)malloc(sizeof(int)*SIZEMUL);
    int* d = (int *)malloc(sizeof(int)*SIZEMUL);

    cout<<"Populating arrays"<<endl;
    // int ka = 0;
    // int kb = 10;
    for(int i=0;i<SIZEMUL;i++){
        a[i] = 6;
        b[i] = 9;
    }

    // cout<<"a: "<<endl;
    // cout<<a[SIZEMUL-1]<<endl;
    // // for(int i=0;i<SIZEX;i++){
    // //     for(int j=0;j<SIZEY;j++){
    // //         cout<<a[index(i,j)]<<" ";
    // //     }
    // //     cout<<endl;
    // // }
    // cout<<endl<<"b: "<<endl;
    // cout<<b[SIZEMUL-1]<<endl;
    // // for(int i=0;i<SIZEX;i++){
    // //     for(int j=0;j<SIZEY;j++){
    // //         cout<<b[index(i,j)]<<" ";
    // //     }
    // //     cout<<endl;
    // // }

    // GPU implementation
    cout<<endl<<"Begin GPU calculation"<<endl;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    c = matrix_mult(a, b, c);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout<<"End GPU calculation"<<endl;

    // cout << "Time difference = " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << "[ns]" <<endl;
    // cout << "Time difference = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" <<endl;    
    cout << "[GPU] Time difference = " << chrono::duration_cast<chrono::milliseconds> (end - begin).count() << "[ms]" <<endl;   
    cout << "[GPU] Time difference = " << chrono::duration_cast<chrono::seconds> (end - begin).count() << "[s]" <<endl;   

    // CPU implementation
    cout<<endl<<"Begin CPU calculation"<<endl;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    for(int i=0;i<SIZEX;i++){
        for(int j=0;j<SIZEY;j++){
            for(int k=0;k<SIZEY;k++){
                d[index(i,j)] += a[index(i,k)]*b[index(k,j)];
            }
        }
    }
    chrono::steady_clock::time_point stop = chrono::steady_clock::now();
    cout<<"End CPU calculation"<<endl;

    // cout << "[CPU] Time difference = " << chrono::duration_cast<chrono::nanoseconds>(stop - start).count() << "[ns]" <<endl;
    // cout << "[CPU] Time difference = " << chrono::duration_cast<chrono::microseconds>(stop - start).count() << "[µs]" <<endl;    
    cout << "[CPU] Time difference = " << chrono::duration_cast<chrono::milliseconds> (stop - start).count() << "[ms]" <<endl;
    cout << "[CPU] Time difference = " << chrono::duration_cast<chrono::seconds> (stop - start).count() << "[s]" <<endl;   

    cout<<endl<<"c: "<<endl;
    // cout<<c[0]<<endl;
    // cout<<c[SIZEMUL-1]<<endl;
    for(int i=0;i<SIZEMUL;i++){
        // for(int j=0;j<SIZEY;j++){
        //     cout<<z[index(i,j)]<<" ";
        // }
        if(i%3==0)
            cout<<endl;
        cout<<c[i]<<" ";
    }

    cout<<endl<<"Begin check"<<endl;
    bool check = true;
    int target = a[0]*b[0]*SIZEX;
    for(int i=0;i<SIZEMUL;i++){
        // if(c[i]!=d[i]){
        if(c[i]!=target){
            check = false;
            break;
        }
    }

    if(check){
        cout<<"Results pass!"<<endl;
    }else{
        cout<<"Results fail!"<<endl;
    }
    
    free(a);
    free(b);
    free(c);
}