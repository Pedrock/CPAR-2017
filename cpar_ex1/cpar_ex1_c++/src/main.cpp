#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <papi.h>

using namespace std;

#define SYSTEMTIME clock_t

void createMatrixes(int m_ar, int m_br, double*& pha, double*& phb, double*& phc) {
    pha = (double *)malloc(m_ar * m_br * sizeof(double));
    phb = (double *)malloc(m_br * m_ar * sizeof(double));
    phc = (double *)malloc(m_ar * m_ar * sizeof(double));

    int i, j;

    for (i = 0; i < m_ar; i++)
        for(j = 0; j < m_br; j++)
            pha[i*m_br + j] = (double)1.0;

    for (i = 0; i < m_br; i++)
        for(j = 0; j < m_ar; j++)
            phb[i*m_ar + j] = (double)(i+1);
}

void printAndFreeMatrixes(int m_ar, int m_br, double*& pha, double*& phb, double*& phc) {
    int j;
    cout << "Result matrix: " << endl;

    for (j = 0; j < min(10, m_ar); j++) {
        cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}

void OnMult(int m_ar, int m_br)
{
    SYSTEMTIME Time1, Time2;
    double temp;
    int i, j, k;
    double *pha, *phb, *phc;

    createMatrixes(m_ar, m_br, pha, phb, phc);

    Time1 = clock();

    for (i = 0; i < m_ar; i++)
    {	
        for (j = 0; j < m_ar; j++)
        {
            temp = 0;
            for (k = 0; k < m_br; k++)
            {
                temp += pha[i*m_br+k] * phb[k*m_ar+j];
            }
            phc[i*m_ar+j] = temp;
        }
    }

    Time2 = clock();
    printf("Time: %3.3f seconds\n", (double)(Time2 - Time1) / CLOCKS_PER_SEC);
    printAndFreeMatrixes(m_ar, m_br, pha, phb, phc);
}


void OnMultLine(int m_ar, int m_br)
{
    SYSTEMTIME Time1, Time2;
    int i, j, k;
    double *pha, *phb, *phc;

    createMatrixes(m_ar, m_br, pha, phb, phc);
    memset(phc, 0, m_ar * m_ar * sizeof(double));

    Time1 = clock();

    for (i = 0; i < m_ar; i++)
    {
        for (k = 0; k < m_br; k++)
        {
            for (j = 0; j < m_ar; j++)
            {
                phc[i*m_ar+j] += pha[i*m_br+k] * phb[k*m_ar+j];
            }
        }
    }

    Time2 = clock();
    printf("Time: %3.3f seconds\n", (double)(Time2 - Time1) / CLOCKS_PER_SEC);
    printAndFreeMatrixes(m_ar, m_br, pha, phb, phc);
}

void OnMultParallel(int m_ar, int m_br, int numThreads)
{
    double time1, time2;
    double *pha, *phb, *phc;

    createMatrixes(m_ar, m_br, pha, phb, phc);

    time1 = omp_get_wtime();

    #pragma omp parallel for num_threads(numThreads)
    for(int i = 0; i < m_ar; i++)
    {
        for(int j = 0; j < m_ar; j++)
        {
            double temp = 0;
            for(int k = 0; k < m_br; k++)
            {
                temp += pha[i*m_br+k] * phb[k*m_ar+j];
            }
            phc[i*m_ar+j] = temp;
        }
    }

    time2 = omp_get_wtime();
    printf("Time: %3.3f seconds\n", time2 - time1);
    printAndFreeMatrixes(m_ar, m_br, pha, phb, phc);
}

void OnMultLineParallel(const int m_ar, const int m_br, const int numThreads)
{
    double time1, time2;
    double *pha, *phb, *phc;

    createMatrixes(m_ar, m_br, pha, phb, phc);
    memset(phc, 0, m_ar * m_ar * sizeof(double));

    time1 = omp_get_wtime();

    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < m_ar; i++)
    {
        for (int k = 0; k < m_br; k++)
        {
            for (int j = 0; j < m_ar; j++)
            {
                phc[i*m_ar+j] += pha[i*m_br+k] * phb[k*m_ar+j];
            }
        }
    }

    time2 = omp_get_wtime();
    printf("Time: %3.3f seconds\n", time2 - time1);
    printAndFreeMatrixes(m_ar, m_br, pha, phb, phc);
}

int main (int argc, char *argv[])
{
    int lin, col;
    int op = 1, numThreads = 4;

    int EventSet = PAPI_NULL;
    long long values[2];
    int ret;


    ret = PAPI_library_init( PAPI_VER_CURRENT );
    if ( ret != PAPI_VER_CURRENT )
        std::cout << "FAIL" << endl;


    ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK) cout << "ERRO: create eventset" << endl;


    ret = PAPI_add_event(EventSet,PAPI_L1_DCM );
    if (ret != PAPI_OK) cout << "ERRO: PAPI_L1_DCM" << endl;


    ret = PAPI_add_event(EventSet,PAPI_L2_DCM);
    if (ret != PAPI_OK) cout << "ERRO: PAPI_L2_DCM" << endl;


    do {
        cout << endl << "1. Multiplication" << endl;
        cout << "2. Line Multiplication" << endl;
        cout << "3. Multiplication in Parallel" << endl;
        cout << "4. Line Multiplication in Parallel" << endl;
        cout << "Selection?: ";
        cin >> op;
        if (op == 0)
            break;
        printf("Dimensions: lins cols ? ");
        cin >> lin >> col;

        if (op == 3 || op == 4) {
            printf("Number of Threads ? ");
            cin >> numThreads;
        }

        // Start counting
        ret = PAPI_start(EventSet);
        if (ret != PAPI_OK) cout << "ERRO: Start PAPI" << endl;

        switch (op){
            case 1:
                OnMult(lin, col);
                break;
            case 2:
                OnMultLine(lin, col);
                break;
            case 3:
                OnMultParallel(lin, col, numThreads);
                break;
            case 4:
                OnMultLineParallel(lin, col, numThreads);
                break;
        }

        ret = PAPI_stop(EventSet, values);
        if (ret != PAPI_OK) cout << "ERRO: Stop PAPI" << endl;
        printf("L1 DCM: %lld \n",values[0]);
        printf("L2 DCM: %lld \n",values[1]);

        ret = PAPI_reset( EventSet );
        if ( ret != PAPI_OK )
            std::cout << "FAIL reset" << endl;



    } while (op != 0);

    ret = PAPI_remove_event( EventSet, PAPI_L1_DCM );
    if ( ret != PAPI_OK )
        std::cout << "FAIL remove event" << endl;

    ret = PAPI_remove_event( EventSet, PAPI_L2_DCM );
    if ( ret != PAPI_OK )
        std::cout << "FAIL remove event" << endl;

    ret = PAPI_destroy_eventset( &EventSet );
    if ( ret != PAPI_OK )
        std::cout << "FAIL destroy" << endl;

}
