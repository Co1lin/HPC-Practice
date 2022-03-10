#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include "mpi.h"
#include "omp.h"

using namespace std;

void rand_fill(int* a, const int len) {
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < len; i++) {
        a[i] = rand();
    }
}

void sum(int *a, const int* b, const int len) {
#pragma omp parallel for // num_threads()
    for (int i = 0; i < len; i++) {
        a[i] += b[i];
    }
}

int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int size = MPI::COMM_WORLD.Get_size();
    const int arr_len = atoi(argv[1]);
    if (rank == 0)
        cout << "arr_len: " << arr_len << endl;

    #pragma omp parallel
    {
        #pragma omp master
        {
            cout << "omp default thread num: " << omp_get_num_threads() << endl;
        }
    }
    
    int* local_data = new int[arr_len];
    rand_fill(local_data, arr_len);
    int* total_data = nullptr;
    if (rank == 0) {
        total_data = new int[arr_len];
    }

    auto start_t = chrono::high_resolution_clock::now();
    MPI::COMM_WORLD.Barrier();
    // cout << "Process " << rank << " out of " << size << " starts to run!" << endl;
    if (rank == 0) {
        start_t = chrono::high_resolution_clock::now();
    }

    // naively reduce (sum one by one)
    if (rank != 0) {
        cout << "Process " << rank << " starts to send!" << endl;
        MPI_Send(local_data, arr_len, MPI::INT, 0, 0, MPI::COMM_WORLD);
        cout << "Process " << rank << " completed sending!" << endl;
    }
    else {
        cout << "Process " << rank << " starts to receive!" << endl;
        memcpy(total_data, local_data, arr_len * sizeof(total_data[0]));
        int* tmp_data = new int[arr_len];
        for (int src_id = 1; src_id < size; src_id++) {
            MPI::COMM_WORLD.Recv(tmp_data, arr_len, MPI::INT, src_id, 0);
            sum(total_data, tmp_data, arr_len);
        }

        const auto end_t = chrono::high_resolution_clock::now();
        const auto duration = chrono::duration_cast<chrono::microseconds>(end_t - start_t).count();

        cout << "Time Cost: " << duration / 1000 << " ms " << duration % 1000 << " us " << endl;
        delete[] tmp_data;
    }

    delete[] local_data;
    delete[] total_data;

    MPI::Finalize();
    return 0;
}

/*
sum: 1-thread
srun -N 1 -n 7 ./naive_reduce 100000000

arr_len: 100000000
Process 0 starts to receive!
Process 2 starts to send!
Process 3 starts to send!
Process 4 starts to send!
Process 6 starts to send!
Process 1 starts to send!
Process 5 starts to send!
Process 1 completed sending!
Process 2 completed sending!
Process 3 completed sending!
Process 4 completed sending!
Process 5 completed sending!
Process 6 completed sending!
Time Cost: 1426 ms 449 us

sum: 28-thread
srun -N 1 -n 7 ./naive_reduce 100000000
arr_len: 100000000
omp default thread num: 28
omp default thread num: 28
omp default thread num: 28
omp default thread num: 28
omp default thread num: 28
omp default thread num: 28
omp default thread num: 28
Process 1 starts to send!
Process 5 starts to send!
Process 0 starts to receive!
Process 3 starts to send!
Process 4 starts to send!
Process 2 starts to send!
Process 6 starts to send!
Process 1 completed sending!
Process 2 completed sending!
Process 3 completed sending!
Process 4 completed sending!
Process 5 completed sending!
Process 6 completed sending!
Time Cost: 1139 ms 197 us
*/