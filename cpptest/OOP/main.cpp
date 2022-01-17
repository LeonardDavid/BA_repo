#include <iostream>
#include "test_object.h"

using namespace std;

int main(){

    Test* test = new Test(69,'x');
    cout<<test->getNumber()<<" "<<test->getCharacter()<<endl;
    return 0;
}