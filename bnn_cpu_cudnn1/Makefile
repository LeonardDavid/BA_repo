intel:
	# nvcc -G -c cuda_kernel.cu
	# g++ -c -g -w -std=c++14 main.cpp net.cpp -I net.hpp netW.hpp -O3
	# g++ -o batch_par_gpu.o cuda_kernel.o net.o main.o -L/usr/local/cuda/lib64 -lcudart

	g++ -o cudnn.o -c main.cpp -I/usr/local/cuda-11.4/include -I/usr/local/cuda-11.4/targets/x86_64-linux/include 
	/usr/local/cuda-11.4/bin/nvcc -ccbin g++ -o cu.o cudnn.o -m64 -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda-11.4/include -I/usr/local/cuda-11.4/targets/x86_64-linux/include -L/usr/local/cuda-11.4/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -lcublasLt -lcudart -lcublas -lcudnn -lstdc++ -lm
pcc:
	#powerpc-linux-gnu-g++ 
	powerpc-fsl-linux-g++ -g -w -std=c++14 -m32 -mhard-float -mcpu=e6500 --sysroot=/opt/fsl-qoriq/2.0/sysroots/ppce6500-fsl-linux --static main.cpp net.cpp -I net.hpp netW.hpp -O3
clean:
	rm -f $(binaries) *.o
