/*
    run with
    $ g++ add%10.cpp -o add.o
    $ ./add.o
*/

#include<iostream>
#include<fstream>

using namespace std;

int main(){
    ifstream f("numbers.txt");
    int count = 0;
    unsigned long long sum = 0;
    unsigned long long a;
    while(!f.eof()){
        f>>a;
        sum += a%10; 
        count++;
    }
    cout<<fixed<<"elements: "<<count<<endl<<"sum: "<<sum<<endl;
}