intel:
	nvcc -c kernel.cu
	g++ -c -g -w -std=c++14 main.cpp -O3
	g++ -o cuda_array kernel.o main.o -L/usr/local/cuda/lib64 -lcudart
pcc:
	#powerpc-linux-gnu-g++ 
	powerpc-fsl-linux-g++ -g -w -std=c++14 -m32 -mhard-float -mcpu=e6500 --sysroot=/opt/fsl-qoriq/2.0/sysroots/ppce6500-fsl-linux --static main.cpp net.cpp -I net.hpp netW.hpp -O3
clean:
	rm -f $(binaries) *.o
