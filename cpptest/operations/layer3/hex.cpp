#include <iostream>

using namespace std;

int main(){
    unsigned long long x[5][1] = {{0x28204204c0728340}, {0x94dd1e257b816cda}, {0x4e2be201d575bfbc}, {0x6830e0efd1762bb6}, {0x160bf2d38d86270a}};

    unsigned long long y[5] = {0x557149ab6d80, 0x5616738b7d88, 0x5616738b7d90, 0x5616738b7d98, 0x5616738b7da0, 0x5616738b7da8 0x5616738b7db0 0x5616738b7db8 0x5616738b7dc0 0x5616738b7dc8 0x5616738b7dd0 0x5616738b7dd8 0x5616738b7de0 0x5616738b7de8};

    0x557149ab6d80
    93945055636864
10101010111000101001001101010110110110110000000

0x5621b24a9f60

    0x1089E20004870400
    1191732066120041472
1000010001001111000100000000000000100100001110000010000000000

    unsigned long long out;

    // for (int c = 0; c < 64; c++) {
    //     out = x | (1ULL << (63 - c % 64));
    //     cout<<out<<" ";
    // }

    cout<<"y: ";
    for(int i=0;i<5;i++){
        cout<<y[i]<<" ";
    }
    cout<<endl<<endl;

    // cout<<"x: ";
    // for(int i=0;i<5;i++){
    //     cout<<x[i]<<" ";
    // }
    // cout<<endl<<endl;

    unsigned long long x64[320];

    cout<<"x[i][0]: "<<endl;
    for(int i=0;i<5;i++){
        for(int c = 0; c<64;c++){
            x64[i*64+c] = x[i][c];
            cout<<x[i][c]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    cout<<"x64[320]: "<<endl;
    for(int i=0;i<320;i++){
        cout<<x64[i]<<" ";
        if(i%64==0){
            cout<<endl;
        }
    }

    // cout<<"x[i][c/64]: ";
    // for(int i=0;i<5;i++){
    //     for(int c = 0; c<64;c++){
    //         cout<<x[i][c/64]<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;
}