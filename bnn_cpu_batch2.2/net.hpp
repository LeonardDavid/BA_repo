#define MAX_THREADS 12	// MAX_THREADS on LDB-XPS machine
#define BATCH_SIZE 4	// BATCH_SIZE < MAX THREADS!

#pragma once
		void predict_NeuralNet(unsigned char * const x, float * pred);
