#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define BLOCK_SIZE 1

__global__ void matrixMulCUDA(float *C, float *A, float *B, int widthA, int widthB)
{
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    const int aBegin = widthA * BLOCK_SIZE * block_y;
    const int aEnd   = aBegin + widthA - 1;
    const int aStep  = BLOCK_SIZE;
    const int bBegin = BLOCK_SIZE * block_x;
    const int bStep  = BLOCK_SIZE * widthB;

    float Csub = 0;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        As[thread_y][thread_x] = A[a + widthA * thread_y + thread_x];
        Bs[thread_y][thread_x] = B[b + widthB * thread_y + thread_x];

        __syncthreads();

		#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[thread_y][k] * Bs[k][thread_x];
        }

        __syncthreads();
    }

    int c = widthB * BLOCK_SIZE * block_y + BLOCK_SIZE * block_x;
    C[c + widthB * thread_y + thread_x] = Csub;
}


int matrixMultiply(int argc, char **argv, dim3 &dimsA, dim3 &dimsB)
{
	// Host
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int size_B = dimsB.x * dimsB.y;
    dim3 dimsC(dimsA.y, dimsB.x, 1);

    unsigned int mem_size_A = sizeof(float) * size_A;
    unsigned int mem_size_B = sizeof(float) * size_B;
    unsigned int mem_size_C = sizeof(float) * dimsA.y * dimsB.x;

    float *h_A = (float *)malloc(mem_size_A);
    float *h_B = (float *)malloc(mem_size_B);
    float *h_C = (float *)malloc(mem_size_C);

    for (int i = 0; i < size_A; i++) {
    	h_A[i] = 1.0;
    }

    for (int i = 0; i < dimsB.y; i++) {
    	for (int j = 0; j < dimsB.x; j++) {
    		h_B[i*dimsB.x + j] = i + 1.0;
    	}
    }

    // Device
    float *d_A, *d_B, *d_C;

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));


    // Threads and grids
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(dimsA.x / threads.x, dimsB.y / threads.y);

    // Start timer
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Multiply matrixes
    matrixMulCUDA<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Get elapsed time
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Print time and performance
    double ops = 2.0 * (double)dimsA.y * (double)dimsA.x * (double)dimsB.x;
    double gigaFlops = (ops * 1.0e-9f) / (msecTotal / 1000.0f);
    printf("Performance: %.2f GFlop/s, Time: %.3f msec", gigaFlops, msecTotal);

    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));


    bool correct = true;
    double eps = 1.e-10;

    double expected_result = 0;
    for (int i = 0; i < dimsA.x; i++) expected_result += i + 1;

    for (int i = 0; i < (int)(dimsC.y * dimsC.x); i++)
    {
        double abs_err = fabs(h_C[i] - expected_result);
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], expected_result, eps);
            correct = false;
        }
    }
    printf("Result: %s\n", correct ? "success" : "failure");

    cudaFree(d_A);
    free(h_A);
    cudaFree(d_B);
    free(h_B);
    cudaFree(d_C);
    free(h_C);

    return !correct;
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (argc != 4) {
        printf("Usage [height A] [width A = height B] [width B]\n");
        exit(0);
    }

    int devID = 0;
    checkCudaErrors(cudaGetDevice(&devID));

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);

    dim3 dimsA(n, m, 1); // A[m, n]
    dim3 dimsB(p, n, 1); // B[n, p]

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.y, dimsA.x, dimsB.y, dimsB.x);

    return matrixMultiply(argc, argv, dimsA, dimsB);
}
