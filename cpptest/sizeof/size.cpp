#include <iostream>

using namespace std;

int main(){
    int batch_size;
    cout<<"BATCH_SIZE: ";
    cin>>batch_size;

    cout<<"dcl0o: "<<batch_size*784*sizeof(unsigned char)<<endl;
    cout<<"dl1b: "<<64*sizeof(float)<<endl;
    cout<<"dcl1w: "<<576*sizeof(signed char)<<endl;
    cout<<"dcl1o: "<<batch_size*50176*sizeof(float)<<endl;

    cout<<"total: "<<batch_size*784*sizeof(unsigned char)+64*sizeof(float)+576*sizeof(signed char)+batch_size*50176*sizeof(float)<<endl;
}