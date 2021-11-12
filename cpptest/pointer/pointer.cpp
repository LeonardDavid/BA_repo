#include <iostream>

int main(){

    const int Ncols = 5;
    int* a = (int *)malloc(sizeof(int)*Ncols);

    for (int i=0; i<Ncols; ++i)
        a[i] = i+1;
    

    for (int i=0; i<Ncols; ++i)
        std::cout<<a[i]<<" ";
    std::cout<<std::endl;

    free(a);
    std::cout<<std::endl;

    const int Mrows = 2;
    // int* b[M] = { (int *)malloc(sizeof(int)*N) };

    static int b[Mrows][Ncols];
    int (*pb)[Ncols] = &b[0];

    for(int i=0;i<Mrows;i++){
        for(int j=0;j<Ncols;j++){
            b[i][j] = i+j;
        }
    }

    for(int i=0;i<Mrows;i++){
        for(int j=0;j<Ncols;j++){
            std::cout<<pb[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    // free(b);
    std::cout<<std::endl;

    const int Lmats = 2;
    static int c[Lmats][Mrows][Ncols];
    int (*pc)[Mrows][Ncols] = &c[0];

    for(int i=0;i<Lmats;i++){
        for(int j=0;j<Mrows;j++){
            for(int k=0;k<Ncols;k++){
                pc[i][j][k] = i+j+k;
            }
        }
    }

    for(int i=0;i<Lmats;i++){
        for(int j=0;j<Mrows;j++){
            for(int k=0;k<Ncols;k++){
                std::cout<<pc[i][j][k]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    // free(pc);
    std::cout<<std::endl;


    return 0;
}