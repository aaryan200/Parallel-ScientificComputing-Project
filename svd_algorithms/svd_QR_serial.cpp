#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <numeric>


#define TOLERANCE 1e-10
#define MAX_ITER 7000

using namespace std;

vector<vector<double>> create_matrix(int rows, int cols)
{
    return vector<vector<double>>(rows, vector<double>(cols, 0.0));
}

void transpose(int m, int n, vector<vector<double>> &A, vector<vector<double>> &A_T)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A_T[j][i] = A[i][j];
}

void multiply(int m1, int n1, vector<vector<double>> &A,
              int m2, int n2, vector<vector<double>> &B,
              vector<vector<double>> &C)
{
    assert(n1 == m2);
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
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            dest[i][j] = src[i][j];
}

void set_identity(int size, vector<vector<double>> &I)
{
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
        for (int j = 0; j < N; j++)
            norm += V[j][i] * V[j][i];
        norm = sqrt(norm);
        R[i][i] = norm;

        for (int j = 0; j < N; j++)
            Q[j][i] = V[j][i] / norm;

        for (int j = i + 1; j < N; j++)
        {
            double dot = 0.0;
            for (int k = 0; k < N; k++)
                dot += Q[k][i] * V[k][j];
            R[i][j] = dot;
            for (int k = 0; k < N; k++)
                V[k][j] -= dot * Q[k][i];
        }
    }
}

double l2_norm_diag_diff(int N, vector<vector<double>> &A, vector<vector<double>> &B)
{
    double sum = 0.0;
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

    for (int i = 0; i < min_mn; i++)
    {
        if (S[i] > 1e-6)
        {
            for (int j = 0; j < M; j++)
                U[j][i] = DV[j][i] / S[i];
        }
        else
        {
            for (int j = 0; j < M; j++)
                U[j][i] = 0.0;
        }
    }

    for (int k = min_mn; k < M; k++)
    {
        for (int i = 0; i < M; i++)
            U[i][k] = (i == k) ? 1.0 : 0.0;

        for (int j = 0; j < k; j++)
        {
            double dot = 0.0;
            for (int i = 0; i < M; i++)
                dot += U[i][k] * U[i][j];
            for (int i = 0; i < M; i++)
                U[i][k] -= dot * U[i][j];
        }

        double norm = 0.0;
        for (int i = 0; i < M; i++)
            norm += U[i][k] * U[i][k];
        norm = sqrt(norm);
        if (norm > 1e-10)
        {
            for (int i = 0; i < M; i++)
                U[i][k] /= norm;
        }
    }
}

void build_sigma(int M, int N, vector<double> &S, vector<vector<double>> &Sigma)
{
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
        for (int col = 0; col < M; col++)
            for (int row = 0; row < M; row++)
                U_sorted[row][col] = eigvecs[row][indices[col]];

        for (int i = 0; i < min_mn; i++)
            S_vec[i] = sqrt(max(eigvals[indices[i]], 0.0));

        auto D_TU = create_matrix(N, min_mn);
        multiply(N, M, D_T, M, min_mn, U_sorted, D_TU);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < min_mn; j++)
                V[i][j] = (S_vec[j] > 1e-6) ? D_TU[i][j] / S_vec[j] : 0.0;

        // Complete V
        for (int j = min_mn; j < N; j++)
        {
            for (int i = 0; i < N; i++)
                V[i][j] = (i == j) ? 1.0 : 0.0;

            for (int k = 0; k < j; k++)
            {
                double dot = 0.0;
                for (int i = 0; i < N; i++)
                    dot += V[i][j] * V[i][k];
                for (int i = 0; i < N; i++)
                    V[i][j] -= dot * V[i][k];
            }

            double norm = 0.0;
            for (int i = 0; i < N; i++)
                norm += V[i][j] * V[i][j];
            norm = sqrt(norm);
            if (norm > 1e-6)
                for (int i = 0; i < N; i++)
                    V[i][j] /= norm;
        }

        copy_matrix(M, M, U, U_sorted);
    }

    Sigma = create_matrix(M, N);
    build_sigma(M, N, S_vec, Sigma);
    V_T = create_matrix(N, N);
    transpose(N, N, V, V_T);
}

int main()
{
    ifstream fin("input_mat.txt");
    if (!fin)
    {
        cerr << "Failed to open input file\n";
        return 1;
    }

    int M, N;
    fin >> M >> N;
    auto D = create_matrix(M, N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            fin >> D[i][j];
    fin.close();

    vector<vector<double>> U, Sigma, V_T;
    SVD(M, N, D, U, Sigma, V_T);

    auto V = create_matrix(N, N);
    transpose(N, N, V_T, V);

    print_matrix(M, M, U, "U");
    print_matrix(M, N, Sigma, "Sigma");
    print_matrix(N, N, V_T, "V^T");
    print_matrix(N, N, V, "V");

    auto US = create_matrix(M, N);
    multiply(M, M, U, M, N, Sigma, US);

    auto reconstructed = create_matrix(M, N);
    multiply(M, N, US, N, N, V_T, reconstructed);
    print_matrix(M, N, reconstructed, "Reconstructed D");

    return 0;
}
