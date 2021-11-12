intel:
	g++ -g -w -std=c++14 -fopenmp -o batch_par_cpu main.cpp net.cpp -I net.hpp netW.hpp -O3
pcc:
	#powerpc-linux-gnu-g++ 
	powerpc-fsl-linux-g++ -g -w -std=c++14 -m32 -mhard-float -mcpu=e6500 --sysroot=/opt/fsl-qoriq/2.0/sysroots/ppce6500-fsl-linux --static main.cpp net.cpp -I net.hpp netW.hpp -O3
clean:
	rm -f $(binaries) *.o
