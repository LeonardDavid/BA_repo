intel:
	nvcc -g -G -c cuda_kernel.cu
	g++ -c -g -w -std=c++14 main.cpp net.cpp -I net.hpp netW.hpp -O3
	g++ -g -O0 -Q -o batch_par_gpu.o cuda_kernel.o net.o main.o -L/usr/local/cuda/lib64 -lcudart
pcc:
	#powerpc-linux-gnu-g++ 
	powerpc-fsl-linux-g++ -g -w -std=c++14 -m32 -mhard-float -mcpu=e6500 --sysroot=/opt/fsl-qoriq/2.0/sysroots/ppce6500-fsl-linux --static main.cpp net.cpp -I net.hpp netW.hpp -O3
clean:
	rm -f $(binaries) *.o
