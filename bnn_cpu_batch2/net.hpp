#define MAX_THREADS 12 	// MAX_THREADS for LDB-XPS: 12
#define BATCH_SIZE 4 	// BATCH_SIZE <= MAX_THREADS

/*
	we don't talk about this
*/

#pragma once
		int predict_NeuralNet(unsigned char * const x, int thread, int label, float * pred); //, float * pred
#pragma once
		int predict_NeuralNet0(unsigned char * const x, int thread, int label, float * pred); //, float * pred
#pragma once
		int predict_NeuralNet1(unsigned char * const x, int thread, int label, float * pred); //, float * pred
#pragma once
		int predict_NeuralNet2(unsigned char * const x, int thread, int label, float * pred); //, float * pred
#pragma once
		int predict_NeuralNet3(unsigned char * const x, int thread, int label, float * pred); //, float * pred
#pragma once
		int predict_NeuralNet4(unsigned char * const x, int thread, int label, float * pred); //, float * pred