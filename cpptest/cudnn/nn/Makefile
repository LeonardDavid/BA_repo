intel:
	g++ -o cudnn.o -c cudnn.cpp -I/usr/local/cuda-11.4/include -I/usr/local/cuda-11.4/targets/x86_64-linux/include 
	/usr/local/cuda-11.4/bin/nvcc -ccbin g++ -o cu.o cudnn.o -m64 -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda-11.4/include -I/usr/local/cuda-11.4/targets/x86_64-linux/include -L/usr/local/cuda-11.4/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -lcublasLt -lcudart -lcublas -lcudnn -lstdc++ -lm
