#include <iostream>

#define Z 2
#define Y 3
#define X 3

using namespace std;

int index(const int x, const int y, const int z) {
     return x * Y * Z + y * Z + z;
}

int main(){

    int data[X][Y][Z];
    const int N = X*Y*Z;
    int flat[N];

    for(int i=0;i<X;i++){
        for(int j=0;j<Y;j++){
            for(int k=0;k<Z;k++){
                data[i][j][k] = i+j+k;
                flat[index(i,j,k)] = data[i][j][k];
            }
        }
    }

    cout<<"data: "<<endl;
    for(int i=0;i<X;i++){
        for(int j=0;j<Y;j++){
            for(int k=0;k<Z;k++){
                std::cout<<data[i][j][k]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    cout<<"flattened: "<<endl;
    for(int i=0;i<N;i++){
        cout<<flat[i]<<" ";
    }
    cout<<endl;



    return 0;
}