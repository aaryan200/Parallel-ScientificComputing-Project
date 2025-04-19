#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

/*
    WIP - This code is a work in progress and is not currently fully functional.
*/
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int m = 4;   
    const int n = 2;    
    const int rows_per_proc = m / num_procs;

    if (m % num_procs != 0) {
        if (rank == 0) 
            cerr << "Error: m must be divisible by number of processes\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<vector<double>> A_local(rows_per_proc, vector<double>(n));
    vector<vector<double>> Q_local(rows_per_proc, vector<double>(n));
    vector<double> V_local(rows_per_proc);
    vector<vector<double>> R(n, vector<double>(n, 0.0));

    if (rank == 0) {
        A_local = {{1, 1},  
                   {1, 0}}; 
    } else if (rank == 1) {
        A_local = {{1, 1},  
                   {1, 0}}; 
    }

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < rows_per_proc; ++i)
            V_local[i] = A_local[i][k];

        for (int j = 0; j < k; ++j) {
            double local_dot = 0.0;
            for (int i = 0; i < rows_per_proc; ++i)
                local_dot += Q_local[i][j] * V_local[i];

            double global_dot;
            MPI_Allreduce(&local_dot, &global_dot, 1, 
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            R[j][k] = global_dot;

            for (int i = 0; i < rows_per_proc; ++i)
                V_local[i] -= R[j][k] * Q_local[i][j];
        }

        double local_norm_sq = 0.0;
        for (int i = 0; i < rows_per_proc; ++i)
            local_norm_sq += V_local[i] * V_local[i];
        
        double global_norm;
        MPI_Allreduce(&local_norm_sq, &global_norm, 1,
                     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_norm = sqrt(global_norm);
        R[k][k] = global_norm;

        for (int i = 0; i < rows_per_proc; ++i)
            Q_local[i][k] = V_local[i] / R[k][k];
    }

    vector<double> full_Q;
    vector<int> recv_counts(num_procs, rows_per_proc * n);
    vector<int> displs(num_procs, 0);
    
    if (rank == 0) full_Q.resize(m * n);
    for (int i = 1; i < num_procs; ++i)
        displs[i] = displs[i-1] + recv_counts[i-1];

    vector<double> flat_Q(rows_per_proc * n);
    for (int i = 0; i < rows_per_proc; ++i)
        for (int j = 0; j < n; ++j)
            flat_Q[i*n + j] = Q_local[i][j];

    MPI_Gatherv(flat_Q.data(), rows_per_proc*n, MPI_DOUBLE,
               full_Q.data(), recv_counts.data(), displs.data(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Orthonormal matrix Q:\n";
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j)
                cout << full_Q[i*n + j] << " ";
            cout << "\n";
        }

        cout << "\nUpper triangular matrix R:\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                cout << R[i][j] << " ";
            cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}