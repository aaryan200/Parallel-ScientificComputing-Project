#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <map>
#include <random>
#include <chrono>

using namespace std;

typedef long long ll;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
ll random(ll a, ll b) {
    return uniform_int_distribution<ll>(a, b)(rng);
}

void mat_mul(vector<vector<double>>& A, vector<vector<double>> B) {
    int m = A.size(), n = A[0].size(), p = B[0].size();
    vector<vector<double>> C(m, vector<double>(p, 0));
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
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T[j][i] = M[i][j];
        }
    }
    return T;
}

pair<vector<double>, double> householder_vector(vector<double>& x) {
    double sigma = 0;
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
    for (int i = 0; i < v.size(); ++i) {
        norm_v += v[i] * v[i];
    }
    norm_v = sqrt(norm_v);
    for (int i = 0; i < v.size(); ++i) {
        v[i] /= norm_v;
    }
    double tau = 2.0;

    return {v, tau};
}

void left_householder(vector<vector<double>>& M, vector<vector<double>>& U, int i) {
    int m = M.size(), n = M[0].size();
    vector<double> x(m - i);
    for (int j = i; j < m; j++) {
        x[j - i] = M[j][i];
    }

    auto [v, tau] = householder_vector(x);

    // Apply H = I - tau * v v^T to M from the left: M = H M
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
    for (int j = i + 1; j < n; j++) {
        x[j - i - 1] = M[i][j];
    }
    auto [v, tau] = householder_vector(x);

    // Apply H = I - tau * v v^T to M from the right: M = M H
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
    for (int i = 0; i < m; i++) {
        U[i][i] = 1;
    }
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

// Helper function to apply a Givens rotation to update vector v
void apply_rotation_to_Vt(double cs, double sn, vector<double>& v_i, vector<double>& v_i_next, int n) {
    for (int j = 0; j < n; j++) {
        double temp = cs * v_i[j] + sn * v_i_next[j];
        v_i_next[j] = -sn * v_i[j] + cs * v_i_next[j];
        v_i[j] = temp;
    }
}

// Helper function to apply a Givens rotation to update vector u
void apply_rotation_to_Ut(double cs, double sn, vector<double>& u_i, vector<double>& u_i_next, int m) {
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

double EPS = 1e-10;

int MAX_ITER = 1000;

// Convergence check
bool is_converged(const vector<double>& e) {
    for (double val : e) {
        if (fabs(val) > EPS)
            return false;
    }
    return true;
}

double tol = 1e-10;
int maxIter = 1000;

vector<vector<vector<double>>> SVD_GolubKahan(vector<vector<double>> &M) {
    int m = M.size();
    int n = M[0].size();
    int len = min(m, n);

    vector<double> s(len, 0), e(len, 0);
    for (int i = 0; i < len - 1; i++) {
        s[i] = M[i][i];
        e[i] = M[i][i + 1];
    }
    s[len - 1] = M[len - 1][len - 1];

    vector<vector<double>> Ut(m, vector<double>(m, 0));
    vector<vector<double>> Vt(n, vector<double>(n, 0));
    vector<double> mu(len, 0);

    // Initialize U and V to identity matrices
    for (int i = 0; i < m; i++) {
        Ut[i][i] = 1;
    }
    for (int i = 0; i < n; i++) {
        Vt[i][i] = 1;
    }

    int i_start = 0, i_end = len - 1;
    int ind = 1;

    while (true) {
        int idx = len - 2;
        while (idx >= 0 && e[idx] == 0) idx--;
        i_end = idx + 1;

        while (idx >= 0 && e[idx] != 0) idx--;
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

int main() {
    ll m = 6, n = 10;

    vector<vector<double>> M(m, vector<double>(n));

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            M[i][j] = random(1, 100);
        }
    }

    cout << "m: " << m << ", n: " << n << endl;
    cout << "Matrix M:" << endl;
    for (const auto& row : M) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    auto result = svd(M);
    vector<vector<double>> U = result[0];
    vector<double> s = result[1][0];
    vector<vector<double>> V = result[2];

    cout << "Singular values:" << endl;
    for (double val : s) {
        cout << val << " ";
    } cout << endl;

    cout << "Left singular vectors (U):" << endl;
    for (const auto& row : U) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    cout << "Right singular vectors (V):" << endl;
    for (const auto& row : V) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    vector<vector<double>> S(min(m,n), vector<double>(min(m,n), 0));
    for (int i = 0; i < min(m, n); ++i) {
        S[i][i] = s[i];
    }
    mat_mul(U, S);
    mat_mul(U, transpose(V));
    cout << "Reconstructed matrix:" << endl;
    for (const auto& row : U) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}
