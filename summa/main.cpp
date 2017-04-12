#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <mpi.h>
#include <algorithm>

using namespace std;

#define N_DIMS 2

// Utilizar o rank para preencher os blocos.
// Se não definido, é utilizado o algoritmo do primeiro trabalho.
//#define FILL_BLOCKS_RANK

// Matrizes: A[m,k]  B[k,n]  C[m,n]

struct Dimensions {
    size_t m;
    size_t k;
    size_t n;
    size_t aHeight;
    size_t aWidth;
    size_t bHeight;
    size_t bWidth;
    size_t cHeight;
    size_t cWidth;
    size_t aReservedSpace;
    size_t bReservedSpace;
};


int getBlockDimension(const int dimension, const int nProcsDim, const int coord) {
    int i = coord * dimension / nProcsDim;
    int next = (coord + 1) * dimension / nProcsDim;
    return next - i;
}

int getBlockMaxDimension(const int dimension, const int nProcsDim) {
    return dimension / nProcsDim + ((dimension % nProcsDim) > 0);
}

Dimensions getDimensions(const int m, const int k, const int n, const int dims[N_DIMS], const int coords[N_DIMS]) {
    Dimensions dimensions;

    dimensions.m = (size_t)m;
    dimensions.k = (size_t)k;
    dimensions.n = (size_t)n;

    dimensions.aHeight = (size_t)getBlockDimension(m, dims[0], coords[0]);
    dimensions.aWidth = (size_t)getBlockDimension(k, dims[1], coords[1]);

    dimensions.bHeight = (size_t)getBlockDimension(k, dims[0], coords[0]);
    dimensions.bWidth = (size_t)getBlockDimension(n, dims[1], coords[1]);

    dimensions.cHeight = (size_t)getBlockDimension(m, dims[0], coords[0]);
    dimensions.cWidth = (size_t)getBlockDimension(n, dims[1], coords[1]);

    dimensions.aReservedSpace = (size_t)getBlockMaxDimension(m, dims[0]) * getBlockMaxDimension(k, dims[1]);
    dimensions.bReservedSpace = (size_t)getBlockMaxDimension(k, dims[0]) * getBlockMaxDimension(n, dims[1]);

    return dimensions;
}

void createBlocks(const int rank,
                  const int coords[N_DIMS], const int dims[N_DIMS],
                  Dimensions dim,
                  double *&a, double *&b, double *&c)
{
    a = (double *)malloc(dim.aReservedSpace * sizeof(double));
    b = (double *)malloc(dim.bReservedSpace * sizeof(double));
    c = (double *)calloc((dim.cHeight * dim.cWidth), sizeof(double));

    for (int i = 0; i < dim.aHeight; i++) {
        for (int j = 0; j < dim.aWidth; j++) {
            double val;
#ifdef FILL_BLOCKS_RANK
            val = rank;
#else
            val = 1.0;
#endif
            a[i*dim.aWidth + j] = val;
        }
    }

#ifndef FILL_BLOCKS_RANK
    const size_t offset = coords[0] * dim.k / dims[0];
#endif

    for (int i = 0; i < dim.bHeight; i++) {
        for (int j = 0; j < dim.bWidth; j++) {
            double val;
#ifdef FILL_BLOCKS_RANK
            val = rank;
#else
            val = offset + i + 1.0;
#endif
            b[i*dim.bWidth + j] = val;
        }
    }
}

void matrixMultiply(const int m, const int k, const int n, const double *A, const double *B, double *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
            for (int l = 0; l < k; l++) {
                C[i*n + j] += A[i*k + l] * B[l*n + j];
            }
        }
    }
}

void addMatrixToMatrix(const int m, const int n, double *other, double *dest) {
    int length = m*n;
    for (int i = 0; i < length; i++) {
        dest[i] += other[i];
    }
}

void print_matrix(const int rows, const int cols, const double *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%1.2f\t", matrix[i*cols + j]);
        }
        printf("\n");
    }
}

void SUMMA(const int coords[N_DIMS], const int dims[N_DIMS],
           MPI_Comm comm_cart, const int rank,
           const Dimensions dimensions,
           const int m, const int k, const int n,
           double *a_block, double *b_block, double *c_block)
{
    int my_row = coords[0];
    int my_col = coords[1];

    MPI_Comm row_comm, col_comm;

    int remain_dims_row[N_DIMS] = {0, 1};
    int remain_dims_col[N_DIMS] = {1, 0};

    MPI_Cart_sub(comm_cart, remain_dims_row, &row_comm);
    MPI_Cart_sub(comm_cart, remain_dims_col, &col_comm);

    double *a_block_copy = (double *)malloc(dimensions.aReservedSpace*sizeof(double));
    double *b_block_copy = (double *)malloc(dimensions.bReservedSpace*sizeof(double));
    double *c_block_temp = (double *)calloc(dimensions.cHeight*dimensions.cWidth, sizeof(double));

    memcpy(a_block_copy, a_block, dimensions.aHeight*dimensions.aWidth*sizeof(double));
    memcpy(b_block_copy, b_block, dimensions.bHeight*dimensions.bWidth*sizeof(double));

    int nBlocks = dims[0];

    for (int l = 0; l < nBlocks; l++) {

        int mb = getBlockDimension(m, dims[0], my_row);
        int kb = getBlockDimension(k, dims[1], l);
        int nb = getBlockDimension(n, dims[1], my_col);

        if (my_col == l) {
            memcpy(a_block, a_block_copy, mb*kb*sizeof(double));
        }
        // broadcast A block within my row
        MPI_Bcast(a_block, mb*kb, MPI_DOUBLE, l, row_comm);

        if (my_row == l) {
            memcpy(b_block, b_block_copy, kb*nb*sizeof(double));
        }
        // broadcast B block within my column
        MPI_Bcast(b_block, kb*nb, MPI_DOUBLE, l, col_comm);

        // c_block_temp = a_block * b_block
        matrixMultiply(mb, kb, nb, a_block, b_block, c_block_temp);

        // c_block += c_block_temp
        addMatrixToMatrix(mb, nb, c_block_temp, c_block);
    }

    free(a_block_copy);
    free(b_block_copy);
    free(c_block_temp);
}

void gatherResult(const int rank, const int dims[N_DIMS],
                  const size_t mb, const size_t nb,
                  const double *block,
                  const int m, const int n,
                  double *total)
{
    double *recvbuf = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recvbuf = (double*)malloc(m * n * sizeof(double));
        int nProcs = dims[0] * dims[1];
        recvcounts = (int*)malloc(nProcs * sizeof(int));
        displs = (int*)malloc(nProcs * sizeof(int));

        for (int i = 0, curDispl = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                recvcounts[i * dims[1] + j] = getBlockDimension(m, dims[0], i) * getBlockDimension(n, dims[1], j);
                displs[i * dims[1] + j] = curDispl;
                curDispl += recvcounts[i * dims[1] + j];
            }
        }
    }

    MPI_Gatherv(block, mb*nb, MPI_DOUBLE, recvbuf, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) return;

    int idx = 0;
    for (int iBlocks = 0; iBlocks < dims[0]; iBlocks++) {
        for (int jBlocks = 0; jBlocks < dims[1]; jBlocks++) {

            const int blockFirstRow = iBlocks * m / dims[0];
            const int blockFirstCol = jBlocks * n / dims[1];
            const int blockHeight = (iBlocks + 1) * m / dims[0] - blockFirstRow;
            const int blockWidth = (jBlocks + 1) * n / dims[1] - blockFirstCol;

            for (int i = 0; i < blockHeight; ++i) {
                for (int j = 0; j < blockWidth; ++j) {
                    total[(blockFirstRow + i)*n + (blockFirstCol + j)] = recvbuf[idx];
                    idx++;
                }
            }
        }
    }

    free(recvbuf);
    free(recvcounts);
}


void readArgs(int argc, char *argv[], const int rank, int &m, int &k, int &n, int &print_result) {
    if (argc != 4 && argc != 5) {
        if (rank == 0) {
            cerr << "Usage:" << endl
                 << "mpirun -n <number of procs> ./summa <m> <k> <n> [print_result]" << endl
                 << "A[m, k] * B[k, n] = C[m, n]" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
    print_result = (argc == 4) ? 0 : atoi(argv[4]);

    if (m <= 0 || k <= 0 || n <= 0) {
        if (rank == 0) {
            cerr << "Error: Invalid matrices dimensions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char *argv[])
{
    int m, k, n, print_result, rank, numberProcesses;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    readArgs(argc, argv, rank, m, k, n, print_result);

    MPI_Comm_size(MPI_COMM_WORLD, &numberProcesses);

    if (min(m, min(k, n)) < numberProcesses) {
        if (rank == 0) {
            cerr << "Error: None of the matrices dimensions can be smaller than the number of processes." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_proc_rows = (int)sqrt(numberProcesses);
    int n_proc_cols = n_proc_rows;

    if (n_proc_cols * n_proc_rows != numberProcesses) {
        cerr << "Error: The number of processes must be a perfect square." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int dims[N_DIMS] = {n_proc_rows, n_proc_cols};
    const int periods[N_DIMS] = {0, 0};
    const int reorder = 0;
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, N_DIMS, dims, periods, reorder, &comm_cart);

    int coords[N_DIMS];
    MPI_Cart_coords(comm_cart, rank, N_DIMS, coords);

    const Dimensions dimensions = getDimensions(m, k, n, dims, coords);

    double *a_block, *b_block, *c_block;
    createBlocks(rank, coords, dims, dimensions, a_block, b_block, c_block);

    double timeStart, timeEnd;
    timeStart = MPI_Wtime();

    SUMMA(coords, dims, comm_cart, rank, dimensions, m, k, n, a_block, b_block, c_block);

    timeEnd = MPI_Wtime();

    double timeDif = timeEnd - timeStart;
    double maxTimeDiff;

    MPI_Reduce(&timeDif, &maxTimeDiff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Time: " << maxTimeDiff << " seconds." << endl;
    }

    if (print_result) {
        double * c_total = NULL;
        if (rank == 0) {
            c_total = (double *)malloc(m * n * sizeof(double));
        }

        gatherResult(rank, dims, dimensions.cHeight, dimensions.cWidth, c_block, m, n, c_total);

        if (rank == 0) {
            cout << endl << "Matrix C: " << endl;
            print_matrix(m, n, c_total);
            cout << endl;
            free(c_total);
        }
    }

    free(a_block);
    free(b_block);
    free(c_block);

    MPI_Finalize();
    return 0;
}

