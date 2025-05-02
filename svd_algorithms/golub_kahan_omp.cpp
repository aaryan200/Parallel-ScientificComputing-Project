#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <string>
#include <map>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

typedef long long ll;

double tol = 1e-10;
int maxIter = 100000000;
double EPS = 1e-10;

ll getCurTime() {
    return duration_cast<microseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void mat_mul(vector<vector<double>>& A, vector<vector<double>> B) {
    int m = A.size(), n = A[0].size(), p = B[0].size();
    vector<vector<double>> C(m, vector<double>(p, 0));
    
    #pragma omp parallel for collapse(1)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    A = C;
}

vector<vector<double>> transpose(const vector<vector<double>>& M) {
    int m = M.size(), n = M[0].size();

    vector<vector<double>> T(n, vector<double>(m));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T[j][i] = M[i][j];
        }
    }

    return T;
}

pair<vector<double>, double> householder_vector(vector<double>& x) {
    double sigma = 0;

    #pragma omp parallel for reduction(+:sigma)
    for (int i = 0; i < x.size(); ++i) {
        sigma += x[i] * x[i];
    }

    if (sigma == 0) {
        return {vector<double>(x.size(), 1), 0};
    }

    vector<double> v = x;
    int sign = (x[0] >= 0) ? 1 : -1;
    v[0] += sign * sqrt(sigma);

    double norm_v = 0;
    
    #pragma omp parallel for reduction(+:norm_v)
    for (int i = 0; i < v.size(); ++i) {
        norm_v += v[i] * v[i];
    }
    norm_v = sqrt(norm_v);

    #pragma omp parallel for
    for (int i = 0; i < v.size(); ++i) {
        v[i] /= norm_v;
    }

    double tau = 2.0;
    return {v, tau};
}

void left_householder(vector<vector<double>>& M, vector<vector<double>>& U, int i) {
    int m = M.size(), n = M[0].size();
    vector<double> x(m - i);

    #pragma omp parallel for
    for (int j = i; j < m; j++) {
        x[j - i] = M[j][i];
    }

    auto [v, tau] = householder_vector(x);

    // Apply H = I - tau * v v^T to M from the left: M = H M
    #pragma omp parallel for
    for (int k = i; k < n; k++) {
        double dot = 0;
        for (int j = i; j < m; j++) {
            dot += v[j - i] * M[j][k];
        }
        for (int j = i; j < m; j++) {
            M[j][k] -= tau * v[j - i] * dot;
            if(abs(M[j][k]) < 1e-10) {
                M[j][k] = 0; // Avoid numerical issues
            }
        }
    }

    // Accumulate into U: U = H * U
    #pragma omp parallel for
    for (int k = 0; k < m; k++) {
        double dot = 0;
        for (int j = i; j < m; j++) {
            dot += v[j - i] * U[k][j];
        }
        for (int j = i; j < m; j++) {
            U[k][j] -= tau * v[j - i] * dot;
        }
    }
}

void right_householder(vector<vector<double>>& M, vector<vector<double>>& V, int i) {
    int m = M.size(), n = M[0].size();
    vector<double> x(n - i - 1);

    #pragma omp parallel for
    for (int j = i + 1; j < n; j++) {
        x[j - i - 1] = M[i][j];
    }

    auto [v, tau] = householder_vector(x);

    // Apply H = I - tau * v v^T to M from the right: M = M H
    #pragma omp parallel for
    for (int k = 0; k < m; ++k) {
        double dot = 0;
        for (int j = i + 1; j < n; j++) {
            dot += v[j - i - 1] * M[k][j];
        }
        for (int j = i + 1; j < n; j++) {
            M[k][j] -= tau * v[j - i - 1] * dot;
            if(abs(M[k][j]) < 1e-10) {
                M[k][j] = 0; // Avoid numerical issues
            }
        }
    }

    // Accumulate into V: V = V * H
    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        double dot = 0;
        for (int j = i + 1; j < n; j++) {
            dot += v[j - i - 1] * V[k][j];
        }
        for (int j = i + 1; j < n; j++) {
            V[k][j] -= tau * v[j - i - 1] * dot;
        }
    }
}

vector<vector<vector<double>>> bidiagonalize(vector<vector<double>>& M) {
    int m = M.size(), n = M[0].size();
    vector<vector<double>> U(m, vector<double>(m, 0));
    vector<vector<double>> V(n, vector<double>(n, 0));

    // Initialize U and V to identity matrices
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        U[i][i] = 1;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        V[i][i] = 1;
    }

    for (int i=0; i<min(m,n); i++)
    {
        // Apply left Householder transformation
        left_householder(M, U, i);

        // Apply right Householder transformation
        if (i < n - 1) {
            right_householder(M, V, i);
        }
    }

    return {U, V};
}

// Apply a Givens rotation to update vector v
void apply_rotation_to_Vt(double cs, double sn, vector<double>& v_i, vector<double>& v_i_next, int n) {
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double temp = cs * v_i[j] + sn * v_i_next[j];
        v_i_next[j] = -sn * v_i[j] + cs * v_i_next[j];
        v_i[j] = temp;
    }
}

// Apply a Givens rotation to update vector u
void apply_rotation_to_Ut(double cs, double sn, vector<double>& u_i, vector<double>& u_i_next, int m) {
    #pragma omp parallel for
    for (int j = 0; j < m; j++) {
        double temp = cs * u_i[j] + sn * u_i_next[j];
        u_i_next[j] = -sn * u_i[j] + cs * u_i_next[j];
        u_i[j] = temp;
    }
}

// Function to compute the Givens rotation for eliminating off-diagonal elements
void compute_givens_rotation(double f, double g, double& cs, double& sn, double& r) {
    if (f == 0) {
        cs = 0;
        sn = 1;
        r = g;
    } else if (fabs(f) > fabs(g)) {
        double t = g / f;
        double tt = sqrt(1 + t * t);
        cs = 1 / tt;
        sn = t * cs;
        r = f * tt;
    } else {
        double t = f / g;
        double tt = sqrt(1 + t * t);
        sn = 1 / tt;
        cs = t * sn;
        r = g * tt;
    }
}

// Implicit zero-shift QR step for bidiagonal matrix
void implicitZeroShiftQR(vector<double>& s, vector<double>& e, vector<vector<double>>& Ut, vector<vector<double>>& Vt, int i_start, int i_end, int m, int n) {
    double oldcs = 1, oldsn = 0;
    double f = s[i_start], g = e[i_start], h = 0;
    double cs = 0, sn = 0, r = 0;
    double t, tt;

    for (int i = i_start; i < i_end; i++) {
        // Givens rotation to eliminate e[i]
        compute_givens_rotation(f, g, cs, sn, r);
        apply_rotation_to_Vt(cs, sn, Vt[i], Vt[i + 1], n);

        if (i != i_start) {
            e[i - 1] = oldsn * r;
        }

        f = oldcs * r;
        g = s[i + 1] * sn;
        h = s[i + 1] * cs;

        // Another Givens rotation for eliminating the next element
        compute_givens_rotation(f, g, cs, sn, r);
        apply_rotation_to_Ut(cs, sn, Ut[i], Ut[i + 1], m);

        s[i] = r;
        f = h;
        g = e[i + 1];
        oldcs = cs;
        oldsn = sn;
    }

    e[i_end - 1] = h * sn;
    s[i_end] = h * cs;
}

// Convergence check
bool is_converged(const vector<double>& e) {
    bool converged = true;

    double val = 0;

    #pragma omp parallel for shared(converged)
    for (int i = 0; i < e.size(); i++) {
        if (fabs(e[i]) > EPS) {
            #pragma omp atomic write
            converged = false;
            #pragma omp atomic write
            val = e[i];
        }
    }

    return converged;
}

double maxEl(const vector<double>& v) {
    double maxVal = 0;
    for (double val : v) {
        if (fabs(val) > maxVal) {
            maxVal = fabs(val);
        }
    }
    return maxVal;
}

vector<vector<vector<double>>> SVD_GolubKahan(vector<vector<double>> &M) {
    int m = M.size();
    int n = M[0].size();
    int len = min(m, n);

    vector<double> s(len, 0), e(len, 0);

    #pragma omp parallel for
    for (int i = 0; i < len - 1; i++) {
        s[i] = M[i][i];
        e[i] = M[i][i + 1];
    }
    s[len - 1] = M[len - 1][len - 1];

    vector<vector<double>> Ut(m, vector<double>(m, 0));
    vector<vector<double>> Vt(n, vector<double>(n, 0));
    vector<double> mu(len, 0);

    // Initialize U and V to identity matrices
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        Ut[i][i] = 1;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Vt[i][i] = 1;
    }

    int i_start = 0, i_end = len - 1;
    int ind = 1;

    while (true) {
        int idx = len - 2;
        while (idx >= 0 && abs(e[idx]) <= tol) idx--;
        i_end = idx + 1;

        while (idx >= 0 && abs(e[idx]) > tol) idx--;
        i_start = idx + 1;

        if (i_start == i_end) break;

        bool set2Zero = false;
        mu[i_start] = fabs(s[i_start]);

        for (int j = i_start; j < i_end; j++) {
            mu[j + 1] = fabs(s[j + 1]) * mu[j] / (mu[j] + fabs(e[j]));
            if (fabs(e[j]) <= mu[j] * tol) {
                e[j] = 0;
                set2Zero = true;
            }
        }
        if (set2Zero) continue;

        implicitZeroShiftQR(s, e, Ut, Vt, i_start, i_end, m, n);

        if (ind++ == maxIter) break;
    }

    // Make singular values non-negative
    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        if (s[i] < 0) {
            for (int j = 0; j < m; j++) {
                Ut[i][j] *= -1;
            }
            s[i] *= -1;
        }
    }

    // Sort singular values and corresponding vectors
    vector<int> order(len);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&s](int a, int b) { return s[a] > s[b]; });

    vector<vector<double>> U(m, vector<double>(len));
    vector<vector<double>> V(n, vector<double>(len));

    #pragma omp parallel for
    for (int i = 0; i < len; ++i) {
        int k = order[i];
        // row k of Ut is the true U[:,k]
        for (int row = 0; row < m; ++row) {
            U[row][i] = Ut[k][row];
        }
        // likewise for V
        for (int row = 0; row < n; ++row) {
            V[row][i] = Vt[k][row];
        }
    }
    
    return {U, {s, e}, V};
}

vector<vector<vector<double>>> svd(vector<vector<double>> &M) {
    int m = M.size();
    int n = M[0].size();

    bool wide = (m < n);

    if (wide) {
        M = transpose(M);
        swap(m, n);
    }

    vector<vector<double>> U, V, B;
    vector<vector<vector<double>>> result = bidiagonalize(M);
    U = result[0];
    V = result[1];

    result = SVD_GolubKahan(M);
    mat_mul(U, result[0]);
    vector<double> s = result[1][0];
    mat_mul(V, result[2]);

    if(wide) {
        swap(U, V);
    }

    return {U, {s}, V};
}

void set_identity(int size, vector<vector<double>> &I)
{   
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            I[i][j] = (i == j) ? 1.0 : 0.0;
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

bool testOrthogonality(vector<vector<double>> &A)
{
    int m = A.size(), n = A[0].size();

    vector<vector<double>> I(n, vector<double>(n, 0));
    set_identity(n, I);
    vector<vector<double>> AT = transpose(A);
    vector<vector<double>> result(n, vector<double>(n, 0));
    multiply(n, m, AT, m, n, A, result);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (fabs(result[i][j] - I[i][j]) > 1e-6)
                return false;
    return true;
}

bool testSVD(vector<vector<double>> &A, vector<vector<double>> &U, vector<double> &Sigma, vector<vector<double>> &V) {
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
    vector<vector<double>> S(min_dim, vector<double>(min_dim, 0));

    #pragma omp parallel for
    for (int i = 0; i < min_dim; ++i)
        S[i][i] = Sigma[i];

    // Compute U * S
    vector<vector<double>> US(m, vector<double>(min_dim, 0));
    multiply(m, min_dim, U, min_dim, min_dim, S, US);

    // Compute (U * S) * V^T
    vector<vector<double>> VT = transpose(V);
    vector<vector<double>> reconstructed(m, vector<double>(n, 0));
    multiply(m, min_dim, US, min_dim, n, VT, reconstructed);

    // Check if A and reconstructed are close
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(A[i][j] - reconstructed[i][j]) > 1e-6) {
                cout << "A and USV^T are not close at (" << i << ", " << j << ")" << endl;
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

    
    int m = stoi(argv[1]), n = stoi(argv[2]);
    int size = min(m, n);
    
    int p = stoi(argv[3]);
    omp_set_num_threads(p);


    char filename[256];
    sprintf(filename, "../inputs/input_matrix_%04d_%04d.bin", m, n);

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
    if (file_rows != m || file_cols != n)
    {
        cerr << "Error: File dimensions " << file_rows << "x" << file_cols
             << " don't match expected " << m << "x" << n << endl;
        return 1;
    }

    vector<vector<double>> M(m, vector<double>(n));

    for (int i = 0; i < m; ++i) {
        file.read(reinterpret_cast<char *>(M[i].data()), n * sizeof(double));
    }
    file.close();

    // Copy M to a new matrix
    vector<vector<double>> orig_M(m, vector<double>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            orig_M[i][j] = M[i][j];
        }
    }

    auto start = getCurTime();

    auto result = svd(M);

    auto end = getCurTime();

    vector<vector<double>> U = result[0];
    vector<double> s = result[1][0];
    vector<vector<double>> V = result[2];

    // Check correctness of SVD
    if (testSVD(orig_M, U, s, V))
    {
        cout << "PASSED" << endl;
    }
    else
    {
        cout << "FAILED" << endl;
        return 1;
    }

    cout << "Time taken for SVD: " << end - start << " microseconds" << endl;

    // Write results in U_{m}_{n}.bin, S_{m}_{n}.bin, V_{m}_{n}.bin
    sprintf(filename, "../prog_output/U_%04d_%04d_par.bin", m, n);
    ofstream uFile(filename, ios::binary);
    // First write the dimensions
    uFile.write(reinterpret_cast<char *>(&m), sizeof(int));
    uFile.write(reinterpret_cast<char *>(&size), sizeof(int));
    // Write U matrix
    for (int i = 0; i < m; ++i)
    {
        uFile.write(reinterpret_cast<char *>(U[i].data()), size * sizeof(double));
    }
    uFile.close();

    printf("Saved U matrix to %s\n", filename);

    sprintf(filename, "../prog_output/S_%04d_%04d_par.bin", m, n);
    ofstream sFile(filename, ios::binary);
    // First write the dimensions
    sFile.write(reinterpret_cast<char *>(&size), sizeof(int));
    // Write singular values
    for (int i = 0; i < size; ++i)
    {
        sFile.write(reinterpret_cast<char *>(&s[i]), sizeof(double));
    }
    sFile.close();

    printf("Saved S matrix to %s\n", filename);

    sprintf(filename, "../prog_output/V_%04d_%04d_par.bin", m, n);
    ofstream vFile(filename, ios::binary);
    // First write the dimensions
    vFile.write(reinterpret_cast<char *>(&n), sizeof(int));
    vFile.write(reinterpret_cast<char *>(&size), sizeof(int));
    // Write V matrix
    for (int i = 0; i < n; ++i)
    {
        vFile.write(reinterpret_cast<char *>(V[i].data()), size * sizeof(double));
    }
    vFile.close();
    printf("Saved V matrix to %s\n", filename);

    return 0;
}