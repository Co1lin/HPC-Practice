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

int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int size = MPI::COMM_WORLD.Get_size();
    const int arr_len = atoi(argv[1]);
    if (rank == 0)
        cout << "arr_len: " << arr_len << endl;
    
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

    // mpi reduce
    cout << "Process " << rank << " starts to reduce!" << endl;    
    MPI::COMM_WORLD.Reduce(local_data, total_data, arr_len, MPI::INT, MPI::SUM, 0);

    if (rank == 0) {
        const auto end_t = chrono::high_resolution_clock::now();
        const auto duration = chrono::duration_cast<chrono::microseconds>(end_t - start_t).count();

        cout << "Time Cost: " << duration / 1000 << " ms " << duration % 1000 << " us " << endl;
    }

    delete[] local_data;
    delete[] total_data;

    MPI::Finalize();
    return 0;
}

/*
srun -N 1 -n 7 ./mpi_reduce 100000000

arr_len: 100000000
Process 3 starts to reduce!
Process 6 starts to reduce!
Process 1 starts to reduce!
Process 0 starts to reduce!
Process 5 starts to reduce!
Process 2 starts to reduce!
Process 4 starts to reduce!
Time Cost: 499 ms 423 us
*/