#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <chrono>

#define TOLERANCE 1e-7
#define MAX_ITER 100000000

using namespace std;
using namespace std::chrono;
typedef long long ll;

ll getCurTime() {
    return duration_cast<microseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}


vector<vector<double>> create_matrix(int rows, int cols)
{
    return vector<vector<double>>(rows, vector<double>(cols, 0.0));
}

void transpose(int m, int n, vector<vector<double>> &A, vector<vector<double>> &A_T)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A_T[j][i] = A[i][j];
}

void multiply(int m1, int n1, vector<vector<double>> &A,
              int m2, int n2, vector<vector<double>> &B,
              vector<vector<double>> &C)
{
    assert(n1 == m2);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m1; i++)
        for (int j = 0; j < n2; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n1; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

void copy_matrix(int m, int n, vector<vector<double>> &dest, vector<vector<double>> &src)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            dest[i][j] = src[i][j];
}

void set_identity(int size, vector<vector<double>> &I)
{   
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            I[i][j] = (i == j) ? 1.0 : 0.0;
}

void modifiedGS(int N, vector<vector<double>> &A, vector<vector<double>> &Q, vector<vector<double>> &R)
{
    auto V = create_matrix(N, N);
    copy_matrix(N, N, V, A);

    for (int i = 0; i < N; i++)
        fill(R[i].begin(), R[i].end(), 0.0);

    for (int i = 0; i < N; i++)
    {
        double norm = 0.0;
        #pragma omp parallel for reduction(+:norm)
        for (int j = 0; j < N; j++)
            norm += V[j][i] * V[j][i];
        norm = sqrt(norm);
        R[i][i] = norm;

        #pragma omp parallel for
        for (int j = 0; j < N; j++)
            Q[j][i] = V[j][i] / norm;

        for (int j = i + 1; j < N; j++)
        {
            double dot = 0.0;
            #pragma omp parallel for reduction(+:dot)
            for (int k = 0; k < N; k++)
                dot += Q[k][i] * V[k][j];
            R[i][j] = dot;
            #pragma omp parallel for
            for (int k = 0; k < N; k++)
                V[k][j] -= dot * Q[k][i];
        }
    }
}

double l2_norm_diag_diff(int N, vector<vector<double>> &A, vector<vector<double>> &B)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++)
    {
        double diff = A[i][i] - B[i][i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void eigendecompose(int N, vector<vector<double>> &A, vector<vector<double>> &eigvecs, vector<double> &eigvals)
{
    auto Q = create_matrix(N, N);
    auto R = create_matrix(N, N);
    auto Ak = create_matrix(N, N);
    auto Ak_next = create_matrix(N, N);
    auto Qk_total = create_matrix(N, N);

    copy_matrix(N, N, Ak, A);
    set_identity(N, Qk_total);

    int iter = 0;
    while (iter++ < MAX_ITER)
    {
        modifiedGS(N, Ak, Q, R);
        multiply(N, N, R, N, N, Q, Ak_next);

        auto temp = create_matrix(N, N);
        multiply(N, N, Qk_total, N, N, Q, temp);
        copy_matrix(N, N, Qk_total, temp);

        if (l2_norm_diag_diff(N, Ak_next, Ak) < TOLERANCE)
            break;

        copy_matrix(N, N, Ak, Ak_next);
    }

    for (int i = 0; i < N; i++)
        eigvals[i] = Ak_next[i][i];
    copy_matrix(N, N, eigvecs, Qk_total);
}

void compute_U(int M, int N, vector<vector<double>> &D, vector<vector<double>> &V, vector<double> &S, int min_mn, vector<vector<double>> &U)
{
    auto DV = create_matrix(M, N);
    multiply(M, N, D, N, N, V, DV);

    #pragma omp parallel for
    for (int i = 0; i < min_mn; i++)
    {
        if (S[i] > 1e-6)
        {
            #pragma omp parallel for
            for (int j = 0; j < M; j++)
                U[j][i] = DV[j][i] / S[i];
        }
        else
        {
            #pragma omp parallel for
            for (int j = 0; j < M; j++)
                U[j][i] = 0.0;
        }
    }

    for (int k = min_mn; k < M; k++)
    {
        #pragma omp parallel for
        for (int i = 0; i < M; i++)
            U[i][k] = (i == k) ? 1.0 : 0.0;

        for (int j = 0; j < k; j++)
        {
            double dot = 0.0;
            #pragma omp parallel for reduction(+:dot)
            for (int i = 0; i < M; i++)
                dot += U[i][k] * U[i][j];
            #pragma omp parallel for
            for (int i = 0; i < M; i++)
                U[i][k] -= dot * U[i][j];
        }

        double norm = 0.0;
        #pragma omp parallel for reduction(+:norm)
        for (int i = 0; i < M; i++)
            norm += U[i][k] * U[i][k];
        norm = sqrt(norm);
        if (norm > 1e-10)
        {
            #pragma omp parallel for
            for (int i = 0; i < M; i++)
                U[i][k] /= norm;
        }
    }
}

void build_sigma(int M, int N, vector<double> &S, vector<vector<double>> &Sigma)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            Sigma[i][j] = (i < N && i == j) ? S[i] : 0.0;
}

void print_matrix(int m, int n, vector<vector<double>> &A, const string &label)
{
    cout << "\n" << label << ":\n";
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            cout << setw(8) << fixed << setprecision(4) << A[i][j] << " ";
        cout << "\n";
    }
}

void SVD(int M, int N, vector<vector<double>> &D,
         vector<vector<double>> &U, vector<vector<double>> &Sigma, vector<vector<double>> &V_T)
{
    int min_mn = min(M, N);
    vector<double> eigvals(min_mn), S_vec(min_mn);

    auto V = create_matrix(N, N);
    U = create_matrix(M, M);

    if (M >= N)
    {
        auto D_T = create_matrix(N, M);
        transpose(M, N, D, D_T);

        auto D_TD = create_matrix(N, N);
        multiply(N, M, D_T, M, N, D, D_TD);

        auto eigvecs = create_matrix(N, N);
        eigendecompose(N, D_TD, eigvecs, eigvals);

        vector<int> indices(N);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](int a, int b) {
            return eigvals[a] > eigvals[b];
        });

        auto V_sorted = create_matrix(N, N);
        #pragma omp parallel for collapse(2)
        for (int col = 0; col < N; col++)
            for (int row = 0; row < N; row++)
                V_sorted[row][col] = eigvecs[row][indices[col]];

        for (int i = 0; i < min_mn; i++)
            S_vec[i] = sqrt(max(eigvals[indices[i]], 0.0));

        compute_U(M, N, D, V_sorted, S_vec, min_mn, U);
        copy_matrix(N, N, V, V_sorted);
    }
    else
    {
        auto D_T = create_matrix(N, M);
        transpose(M, N, D, D_T);

        auto D_DT = create_matrix(M, M);
        multiply(M, N, D, N, M, D_T, D_DT);

        auto eigvecs = create_matrix(M, M);
        eigendecompose(M, D_DT, eigvecs, eigvals);

        vector<int> indices(M);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](int a, int b) {
            return eigvals[a] > eigvals[b];
        });

        auto U_sorted = create_matrix(M, M);
        #pragma omp parallel for collapse(2)
        for (int col = 0; col < M; col++)
            for (int row = 0; row < M; row++)
                U_sorted[row][col] = eigvecs[row][indices[col]];

        for (int i = 0; i < min_mn; i++)
            S_vec[i] = sqrt(max(eigvals[indices[i]], 0.0));

        auto D_TU = create_matrix(N, min_mn);
        multiply(N, M, D_T, M, min_mn, U_sorted, D_TU);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < min_mn; j++)
                V[i][j] = (S_vec[j] > 1e-6) ? D_TU[i][j] / S_vec[j] : 0.0;

        // Complete V
        for (int j = min_mn; j < N; j++)
        {
            #pragma omp parallel for
            for (int i = 0; i < N; i++)
                V[i][j] = (i == j) ? 1.0 : 0.0;

            for (int k = 0; k < j; k++)
            {
                double dot = 0.0;
                #pragma omp parallel for reduction(+:dot)
                for (int i = 0; i < N; i++)
                    dot += V[i][j] * V[i][k];
                #pragma omp parallel for
                for (int i = 0; i < N; i++)
                    V[i][j] -= dot * V[i][k];
            }

            double norm = 0.0;
            #pragma omp parallel for reduction(+:norm)
            for (int i = 0; i < N; i++)
                norm += V[i][j] * V[i][j];
            norm = sqrt(norm);
            if (norm > 1e-6)
            {
                #pragma omp parallel for
                for (int i = 0; i < N; i++)
                    V[i][j] /= norm;
            }
        }

        copy_matrix(M, M, U, U_sorted);
    }

    Sigma = create_matrix(M, N);
    build_sigma(M, N, S_vec, Sigma);
    V_T = create_matrix(N, N);
    transpose(N, N, V, V_T);
}

bool testOrthogonality(vector<vector<double>> &A)
{
    int n = A.size();
    auto I = create_matrix(n, n);
    set_identity(n, I);
    auto AT = create_matrix(n, n);
    transpose(n, n, A, AT);
    auto result = create_matrix(n, n);
    multiply(n, n, AT, n, n, A, result);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (fabs(result[i][j] - I[i][j]) > 1e-5)
                return false;
    return true;
}

bool testSVD(vector<vector<double>> &A, vector<vector<double>> &U, vector<vector<double>> &Sigma, vector<vector<double>> &V) {
    int m = A.size(), n = A[0].size();
    int min_dim = min(m, n);

    // Check orthogonality of U and V
    if (!testOrthogonality(U)) {
        cout << "U is not orthogonal" << endl;
        return false;
    }

    if (!testOrthogonality(V)) {
        cout << "V is not orthogonal" << endl;
        return false;
    }

    // Create proper S matrix with dimensions m×n
    auto S = create_matrix(m, n);
    #pragma omp parallel for
    for (int i = 0; i < min_dim; ++i)
        S[i][i] = Sigma[i][i];

    // Compute U * S
    auto US = create_matrix(m, n);
    multiply(m, m, U, m, n, S, US);

    // Compute (U * S) * V^T
    auto VT = create_matrix(n, n);
    transpose(n, n, V, VT);
    auto reconstructed = create_matrix(m, n);
    multiply(m, n, US, n, n, VT, reconstructed);

    // Check if A and reconstructed are close
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(A[i][j] - reconstructed[i][j]) > 1e-5) {
                cout << "A and USV^T are not close at (" << i << ", " << j << ")" << endl;
                cout << "A[" << i << "][" << j << "] = " << A[i][j] 
                     << ", USV^T[" << i << "][" << j << "] = " << reconstructed[i][j] 
                     << endl;
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <numRows> <numCols> <numThreads>" << endl;
        return 1;
    }
    
    int M = stoi(argv[1]), N = stoi(argv[2]);
    int size = min(M, N);
    
    int p = stoi(argv[3]);
    omp_set_num_threads(p);

    char filename[256];
    sprintf(filename, "../inputs/input_matrix_%04d_%04d.bin", M, N);

    // Read input matrix
    ifstream file(filename, ios::binary);
    if (!file)
    {
        cerr << "Error opening file: " << filename << endl;
        return 1;
    }

    // Read header
    int file_rows, file_cols;
    file.read(reinterpret_cast<char *>(&file_rows), sizeof(int));
    file.read(reinterpret_cast<char *>(&file_cols), sizeof(int));

    // Verify dimensions match command line arguments
    if (file_rows != M || file_cols != N)
    {
        cerr << "Error: File dimensions " << file_rows << "x" << file_cols
             << " don't match expected " << M << "x" << N << endl;
        return 1;
    }

    auto D = create_matrix(M, N);
    for (int i = 0; i < M; ++i) {
        file.read(reinterpret_cast<char *>(D[i].data()), N * sizeof(double));
    }
    file.close();

    auto orig_D = create_matrix(M, N);

    copy_matrix(M, N, orig_D, D);

    vector<vector<double>> U, Sigma, V_T;

    auto start = getCurTime();

    SVD(M, N, D, U, Sigma, V_T);

    auto end = getCurTime();

    auto V = create_matrix(N, N);
    transpose(N, N, V_T, V);

    if (!testSVD(orig_D, U, Sigma, V)) {
        cout << "FAILED" << endl;
        return 1;
    } else {
        cout << "PASSED" << endl;
    }

    cout << "Time taken for SVD: " << end - start << " microseconds" << endl;

    // print_matrix(M, M, U, "U");
    // print_matrix(M, N, Sigma, "Sigma");
    // print_matrix(N, N, V_T, "V^T");
    // print_matrix(N, N, V, "V");

    sprintf(filename, "../prog_output/U_%04d_%04d.bin", M, N);
    ofstream uFile(filename, ios::binary);
    // First write the dimensions
    uFile.write(reinterpret_cast<char *>(&M), sizeof(int));
    uFile.write(reinterpret_cast<char *>(&M), sizeof(int));

    // Write U matrix
    for (int i = 0; i < M; ++i)
    {
        uFile.write(reinterpret_cast<char *>(U[i].data()), M * sizeof(double));
    }
    uFile.close();

    printf("Saved U matrix to %s\n", filename);

    sprintf(filename, "../prog_output/S_%04d_%04d.bin", M, N);
    ofstream sFile(filename, ios::binary);
    // First write the dimensions
    sFile.write(reinterpret_cast<char *>(&size), sizeof(int));
    // Write singular values
    for (int i = 0; i < size; ++i)
    {
        sFile.write(reinterpret_cast<char *>(&Sigma[i][i]), sizeof(double));
    }
    sFile.close();

    printf("Saved S matrix to %s\n", filename);

    sprintf(filename, "../prog_output/V_%04d_%04d.bin", M, N);
    ofstream vFile(filename, ios::binary);
    // First write the dimensions
    vFile.write(reinterpret_cast<char *>(&N), sizeof(int));
    vFile.write(reinterpret_cast<char *>(&N), sizeof(int));
    // Write V matrix
    for (int i = 0; i < N; ++i)
    {
        vFile.write(reinterpret_cast<char *>(V[i].data()), N * sizeof(double));
    }
    vFile.close();

    printf("Saved V matrix to %s\n", filename);

    return 0;
}