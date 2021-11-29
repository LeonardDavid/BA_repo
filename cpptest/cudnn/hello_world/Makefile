intel:
	g++ -I/usr/local/cuda-11.4/include -I/usr/local/cuda-11.4/targets/x86_64-linux/include -o hw.o -c hello_world.cpp
	/usr/local/cuda-11.4/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -o hwcu.o hw.o -I/usr/local/cuda-11.4/include -I/usr/local/cuda-11.4/targets/x86_64-linux/include -L/usr/local/cuda-11.4/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -lcublasLt -lcudart -lcublas -lcudnn -lstdc++ -lm
