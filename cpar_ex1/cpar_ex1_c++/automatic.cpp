#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <papi.h>
#include <fstream>

using namespace std;

#define SYSTEMTIME clock_t

void createMatrixes(int m_ar, int m_br, double*& pha, double*& phb, double*& phc) {
    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    int i, j;

    for(i=0; i<m_ar; i++)
        for(j=0; j<m_ar; j++)
            pha[i*m_ar + j] = (double)1.0;

    for(i=0; i<m_br; i++)
        for(j=0; j<m_br; j++)
            phb[i*m_br + j] = (double)(i+1);
}

void printAndFreeMatrixes(int m_ar, int m_br, double*& pha, double*& phb, double*& phc) {
    int i, j;
    cout << "Result matrix: " << endl;
    for(i=0; i<1; i++)
    {	for(j=0; j<min(10,m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}

double OnMult(int m_ar, int m_br)
{
    SYSTEMTIME Time1, Time2;
    double temp;
    int i, j, k;
    double *pha, *phb, *phc;

    createMatrixes(m_ar, m_br, pha, phb, phc);

    Time1 = clock();

    for(i=0; i<m_ar; i++)
    {	for( j=0; j<m_br; j++)
        {	temp = 0;
            for( k=0; k<m_ar; k++)
            {
                temp += pha[i*m_ar+k] * phb[k*m_br+j];
            }
            phc[i*m_ar+j]=temp;
        }
    }

    Time2 = clock();
    double t = (double)(Time2 - Time1) / CLOCKS_PER_SEC;
    printf("Time: %3.3f seconds\n", t);
    printAndFreeMatrixes(m_ar, m_br, pha, phb, phc);
    return t;
}


double OnMultLine(int m_ar, int m_br)
{
    SYSTEMTIME Time1, Time2;
    int i, j, k;
    double *pha, *phb, *phc;

    createMatrixes(m_ar, m_br, pha, phb, phc);
    memset(phc, 0, m_ar * m_br * sizeof(double));

    Time1 = clock();

    for (i = 0; i < m_ar; i++)
    {
        for (k = 0; k < m_br; k++)
        {
            for (j = 0; j < m_br; j++)
            {
                phc[i*m_ar+j] += pha[i*m_ar+k] * phb[k*m_br+j];
            }
        }
    }

    Time2 = clock();
    double t = (double)(Time2 - Time1) / CLOCKS_PER_SEC;
    printf("Time: %3.3f seconds\n", t);
    printAndFreeMatrixes(m_ar, m_br, pha, phb, phc);
    return t;
}

double OnMultParallel(int m_ar, int m_br, int numThreads)
{
    double time1, time2;
    double *pha, *phb, *phc;

    createMatrixes(m_ar, m_br, pha, phb, phc);

    time1 = omp_get_wtime();

    #pragma omp parallel for num_threads(numThreads)
    for(int i=0; i < m_ar; i++)
    {
        for(int j=0; j < m_br; j++)
        {
            double temp = 0;
            for(int k=0; k < m_ar; k++)
            {
                temp += pha[i*m_ar+k] * phb[k*m_br+j];
            }
            phc[i*m_ar+j] = temp;
        }
    }

    time2 = omp_get_wtime();
    printf("Time: %3.3f seconds\n", time2 - time1);
    printAndFreeMatrixes(m_ar, m_br, pha, phb, phc);
    return time2 - time1;
}

double OnMultLineParallel(const int m_ar, const int m_br, const int numThreads)
{
    double time1, time2;
    double *pha, *phb, *phc;

    createMatrixes(m_ar, m_br, pha, phb, phc);
    memset(phc, 0, m_ar * m_br * sizeof(double));

    time1 = omp_get_wtime();

    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < m_ar; i++)
    {
        for (int k = 0; k < m_br; k++)
        {
            for (int j = 0; j < m_br; j++)
            {
                phc[i*m_ar+j] += pha[i*m_ar+k] * phb[k*m_br+j];
            }
        }
    }

    time2 = omp_get_wtime();
    printf("Time: %3.3f seconds\n", time2 - time1);
    printAndFreeMatrixes(m_ar, m_br, pha, phb, phc);
    return time2 - time1;
}

int main (int argc, char *argv[])
{
    int lin;
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

    ofstream myfile;
    myfile.open("out.csv");
    myfile << "Selection;Dimensions;num Threads;Time;L1 DCM;L2 DCM" << endl;

    do {
        cout << endl << "1. Multiplication" << endl;
        cout << "2. Line Multiplication" << endl;
        cout << "3. Multiplication in Parallel" << endl;
        cout << "4. Line Multiplication in Parallel" << endl;
        cout << "Selection?: ";
        cin >> op;
        if (op == 0)
            break;
        printf("Dimensions: ? ");
        cin >> lin;
        numThreads = 0;
        if (op == 3 || op == 4) {
            printf("Number of Threads ? ");
            cin >> numThreads;
        }

        for (int i = 0; i < 3; i++) {
          // Start counting
          ret = PAPI_start(EventSet);
          if (ret != PAPI_OK) cout << "ERRO: Start PAPI" << endl;

          double t;
          switch (op){
              case 1:
                  t = OnMult(lin, lin);
                  break;
              case 2:
                  t = OnMultLine(lin, lin);
                  break;
              case 3:
                  t = OnMultParallel(lin, lin, numThreads);
                  break;
              case 4:
                  t = OnMultLineParallel(lin, lin, numThreads);
                  break;
          }

          ret = PAPI_stop(EventSet, values);
          if (ret != PAPI_OK) cout << "ERRO: Stop PAPI" << endl;
          printf("L1 DCM: %lld \n",values[0]);
          printf("L2 DCM: %lld \n",values[1]);

          myfile << op << ";" << lin << ";" << numThreads << ";" << t << ";" << values[0] << ";" << values[1] << endl;

          ret = PAPI_reset( EventSet );
          if ( ret != PAPI_OK )
              std::cout << "FAIL reset" << endl;
        }

    } while (op != 0);

    myfile.close();

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
