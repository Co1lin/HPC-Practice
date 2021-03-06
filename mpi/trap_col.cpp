#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include "mpi.h"

using namespace std;

// y = sqrt( 1 - (x-1)^2 )
double f(const double x) {
    return sqrtf64(1 - (x-1) * (x-1));
}

double trap(
    const double a, const double b,
    const int local_n, const double w) {
    double integral = (f(a) + f(b)) / 2;
    for (int i = 1; i < local_n; i++) {
        integral += f(a + i * w);
    }
    return w * integral;
}

int main(int argc, char* argv[]) {
    MPI::Init(argc, argv);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int size = MPI::COMM_WORLD.Get_size();
    const int n_points = atoi(argv[1]);
    if (rank == 0)
        cout << "n_points: " << n_points << endl;
    const double a = 0, b = 2, w = (b - a) / n_points;
    double total_integral = 0;

    auto start_t = chrono::high_resolution_clock::now();
    MPI::COMM_WORLD.Barrier();
    if (rank == 0) {
        start_t = chrono::high_resolution_clock::now();
    }

    // local computation
    const int local_n = n_points / size;
    const double local_a = a + rank * local_n * w;
    const double local_b = local_a + local_n * w;
    const double local_integral = trap(
        local_a, local_b, local_n, w
    );

    // MPI reduce
    MPI::COMM_WORLD.Reduce(&local_integral, &total_integral, 1, MPI::DOUBLE, MPI::SUM, 0);

    if (rank == 0) {
        const auto end_t = chrono::high_resolution_clock::now();
        const auto duration = chrono::duration_cast<chrono::microseconds>(end_t - start_t).count();

        cout << "The double of integral is " << setprecision(16) << total_integral * 2 << endl;
        cout << "Time Cost: " << duration / 1000 << " ms " << duration % 1000 << " us " << endl;
    }

    MPI::Finalize();
    return 0;
}