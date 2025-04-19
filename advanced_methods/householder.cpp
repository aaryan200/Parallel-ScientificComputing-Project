#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <map>

using namespace std;

class Matrix
{
public:
    int rows;
    int cols;
    double **data;
    Matrix(int rows, int cols, bool eye = false) : rows(rows), cols(cols)
    {
        data = new double *[rows];
        for (int i = 0; i < rows; ++i)
        {
            data[i] = new double[cols];

            if (eye)
            {
                for (int j = 0; j < cols; ++j)
                {
                    data[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
            else
            {
                for (int j = 0; j < cols; ++j)
                {
                    data[i][j] = 0.0;
                }
            }
        }
    }

    Matrix(int rows, int cols, map<pair<int, int>, double> &map) : rows(rows), cols(cols)
    {
        data = new double *[rows];
        for (int i = 0; i < rows; ++i)
        {
            data[i] = new double[cols];
            for (int j = 0; j < cols; ++j)
            {
                if (map.find(make_pair(i, j)) == map.end())
                {
                    data[i][j] = 0.0;
                }
                else
                {
                    data[i][j] = map[make_pair(i, j)];
                }
            }
        }
    }

    ~Matrix()
    {
        for (int i = 0; i < rows; ++i)
        {
            delete[] data[i];
        }
        delete[] data;
    }

    double getEntry(int i, int j) const
    {
        if ((i >= rows) or (j >= cols)) {
            // Throw an error
            throw out_of_range("Index out of range");
        }
        return data[i][j];
    }

    Matrix &copy() const
    {
        Matrix &res = *new Matrix(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                res.data[i][j] = data[i][j];
            }
        }
        return res;
    }

    int getRowDimension() const
    {
        return rows;
    }

    int getColumnDimension() const
    {
        return cols;
    }

    double **getData()
    {
        return data;
    }

    Matrix &mtimes(const Matrix &other) const
    {
        if (cols != other.getRowDimension())
        {
            throw invalid_argument("Matrix dimensions do not match for multiplication.");
        }
        Matrix &res = *new Matrix(rows, other.cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < other.cols; ++j)
            {
                res.data[i][j] = 0;
                for (int k = 0; k < cols; ++k)
                {
                    res.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return res;
    }

    Matrix &transpose() const
    {
        Matrix &res = *new Matrix(cols, rows);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                res.data[j][i] = data[i][j];
            }
        }
        return res;
    }
};

class SVD
{
public:
    Matrix *res_U;
    Matrix *res_S;
    Matrix *res_V;
    double tol = 1e-6;
    int maxIter = 1000;

    SVD(Matrix &A)
    {
        Matrix **USV = decompose(A);
        res_U = USV[0];
        res_S = USV[1];
        res_V = USV[2];
    }

    ~SVD()
    {
        delete res_U;
        delete res_S;
        delete res_V;
    }

    Matrix **decompose(Matrix &A)
    {
        int m = A.getRowDimension();
        int n = A.getColumnDimension();
        int tmp = max(m, n);
        maxIter = 3 * tmp * tmp;

        // A = U1BV1'
        Matrix **UBV = bidiagonalize(A);
        Matrix &B = *UBV[1];

        // B = U2SV2'
        Matrix **USV = diagonalizeBD(B);

        // A = U1BV1' = U1U2SV2'V1' = (U1U2)S(V1V2)'
        Matrix **res = new Matrix *[3];

        res[0] = &UBV[0]->mtimes(*USV[0]);
        res[1] = USV[1];
        res[2] = &UBV[2]->mtimes(*USV[2]);

        for (int i = 0; i < 3; i++)
        {
            delete UBV[i];
            UBV[i] = NULL;
        }
        delete[] UBV;
        UBV = NULL;
        delete USV[0];
        USV[0] = NULL;
        delete USV[2];
        USV[2] = NULL;
        delete[] USV;
        USV = NULL;

        return res;
    }

    Matrix **bidiagonalize(Matrix &M)
    {

        Matrix &A = M.copy();
        int m = A.getRowDimension();
        int n = A.getColumnDimension();
        Matrix **UBV = new Matrix *[3];
        vector<double> d(n, 0);
        vector<double> e(n, 0);

        double **AData = ((Matrix &)A).getData();
        double c = 0;
        double s = 0;
        double r = 0;
        for (int j = 0; j < n; j++)
        {
            if (j >= m)
            {
                break;
            }
            // Householder transformation on columns of A(j:m, j:n)
            // Compute the norm of A(j:m, j)
            c = 0;
            for (int i = j; i < m; i++)
            {
                c += pow(AData[i][j], 2);
            }
            if (c != 0)
            {
                s = sqrt(c);
                d[j] = AData[j][j] > 0 ? -s : s;
                r = sqrt(s * (s + fabs(AData[j][j])));
                AData[j][j] -= d[j];
                for (int k = j; k < m; k++)
                {
                    AData[k][j] /= r;
                }
                for (int k = j + 1; k < n; k++)
                {
                    s = 0;
                    for (int t = j; t < m; t++)
                    {
                        s += AData[t][j] * AData[t][k];
                    }
                    for (int t = j; t < m; t++)
                    {
                        AData[t][k] -= s * AData[t][j];
                    }
                }
            }

            // Householder transformation on rows of A(j:m, j+1:n)
            if (j >= n - 1) // We do row-wise HouseHolder transformation n - 1 times
                continue;

            c = 0;
            double *ARow_j = AData[j];
            for (int k = j + 1; k < n; k++)
            {
                c += pow(ARow_j[k], 2);
            }
            if (c != 0)
            {
                s = sqrt(c);
                e[j + 1] = ARow_j[j + 1] > 0 ? -s : s;
                r = sqrt(s * (s + fabs(ARow_j[j + 1])));
                ARow_j[j + 1] -= e[j + 1];
                for (int k = j + 1; k < n; k++)
                {
                    ARow_j[k] /= r;
                }
                double *ARow_k = nullptr;
                for (int k = j + 1; k < m; k++)
                {
                    ARow_k = AData[k];
                    s = 0;
                    for (int t = j + 1; t < n; t++)
                    {
                        s += ARow_j[t] * ARow_k[t];
                    }
                    for (int t = j + 1; t < n; t++)
                    {
                        ARow_k[t] -= s * ARow_j[t];
                    }
                }
            }
        }

        UBV = unpack(A, d, e);
        delete &A;
        return UBV;
    }

    Matrix **unpack(Matrix &A, vector<double> &d, vector<double> &e)
    {
        Matrix **UBV = new Matrix *[3];
        int m = A.getRowDimension();
        int n = A.getColumnDimension();

        Matrix *U = new Matrix(m, m);
        double **UData = U->getData();

        double s = 0;
        double *y = nullptr;

        for (int i = 0; i < m; i++)
        {
            // Compute U^T * e_i
            y = UData[i];
            y[i] = 1;
            for (int j = 0; j < n; j++)
            {
                s = 0;
                for (int k = j; k < m; k++)
                {
                    s += A.getEntry(k, j) * y[k];
                }
                for (int k = j; k < m; k++)
                {
                    y[k] -= A.getEntry(k, j) * s;
                }
            }
        }

        map<pair<int, int>, double> map;
        for (int i = 0; i < m; i++)
        {
            if (i < n)
                map.insert(make_pair(make_pair(i, i), d[i]));
            if (i < n - 1)
            {
                map.insert(make_pair(make_pair(i, i + 1), e[i + 1]));
            }
        }

        Matrix *B = new Matrix(m, n, map);

        Matrix *V = new Matrix(n, n);
        double **VData = V->getData();

        s = 0;

        for (int i = 0; i < n; i++)
        {
            // Compute V^T * e_i
            y = VData[i];
            y[i] = 1;
            for (int j = 0; j < m - 1; j++)
            {
                s = 0;
                for (int k = j + 1; k < n; k++)
                {
                    s += A.getEntry(j, k) * y[k];
                }
                for (int k = j + 1; k < n; k++)
                {
                    y[k] -= A.getEntry(j, k) * s;
                }
            }
        }

        UBV[0] = U;
        UBV[1] = B;
        UBV[2] = V;

        return UBV;
    }

    Matrix **diagonalizeBD(Matrix &B)
    {
        int m = B.getRowDimension();
        int n = B.getColumnDimension();
        int len = m >= n ? n : m;
        int idx = 0;

        /*
         * The bidiagonal matrix B is
         * s[0] e[0]
         *      s[1] e[1]
         *           ...
         *               s[len - 2] e[len - 2]
         *                          s[len - 1]
         */
        vector<double> s(len, 0);
        vector<double> e(len, 0);

        /*double[] pr = ((SparseMatrix) B).getPr();
                int nnz = ((SparseMatrix) B).getNNZ();
                int k = 0;
                while (true) {
                    s[idx] = pr[k++];
                    if (k == nnz)
                        break;
                    e[idx] = pr[k++];
                    idx++;
                }*/
        for (int i = 0; i < len - 1; i++)
        {
            s[i] = B.getEntry(i, i);
            e[i] = B.getEntry(i, i + 1);
        }
        s[len - 1] = B.getEntry(len - 1, len - 1);

        /*
         * B = USV' where U is the left singular vectors,
         * and V is the right singular vectors.
         */

        // U': each row of U' is a left singular vector
        double **Ut = nullptr;
        // Matrix *UtMatrix = &eye(m, m);
        Matrix *UtMatrix = new Matrix(m, m, true);

        Ut = UtMatrix->getData();

        // V': each row of V' is a right singular vector
        double **Vt = nullptr;
        Matrix *VtMatrix = new Matrix(n, n, true);
        Vt = VtMatrix->getData();

        vector<double> mu(len, 0);

        double sigma_min = 0;
        double sigma_max = 0;

        /*
         * B = IBI'
         * Therefore, when pre-multiplying B by Givens rotation transform
         * on i-th and j-th rows, we need to change the i-th and j-th rows
         * of U'.
         * B0 = IB0I' = UG'GBVt = (GUt)'BkVt
         * (Ut)'BVt = (Ut)'BG'GVt = (Lt)'Bk(GVt)
         * where G = |cs  sn|
         *           |-sn cs|
         * G' = |cs -sn|
         *      |sn  cs|
         */

        /*
         * Find B_hat, i.e. the bottommost unreduced submatrix of B.
         * Index starts from 0.
         */
        // *********************************************************

        int i_start = 0;
        int i_end = len - 1;
        /*int cnt_zero_shift = 0;
                int cnt_shifted = 0;*/
        int ind = 1;
        while (true)
        {

            idx = len - 2;
            while (idx >= 0)
            {
                if (e[idx] == 0)
                {
                    idx--;
                }
                else
                {
                    break;
                }
            }
            i_end = idx + 1;
            // Now idx = -1 or e[idx] != 0
            // If idx = -1, then e[0] = 0, i_start = i_end = 0, e = 0
            // Else if e[idx] != 0, then e[i] = 0 for i_end = idx + 1 <= i <= len - 1
            while (idx >= 0)
            {
                if (e[idx] != 0)
                {
                    idx--;
                }
                else
                {
                    break;
                }
            }
            i_start = idx + 1;
            // Now idx = -1 or e[idx] = 0
            // If idx = -1 i_start = 0
            // Else if e[idx] = 0, then e[idx + 1] != 0, e[i_end - 1] != 0
            // i.e. e[i] != 0 for i_start <= i <= i_end - 1

            if (i_start == i_end)
            {
                break;
            }

            // Apply the stopping criterion to B_hat
            // If any e[i] is set to zero, return to loop

            bool set2Zero = false;
            mu[i_start] = fabs(s[i_start]);
            for (int j = i_start; j < i_end; j++)
            {
                mu[j + 1] = fabs(s[j + 1]) * mu[j] / (mu[j] + fabs(e[j]));
                if (fabs(e[j]) <= mu[j] * tol)
                {
                    e[j] = 0;
                    set2Zero = true;
                }
            }
            if (set2Zero)
            {
                continue;
            }

            // Estimate the smallest singular value sigma_min and
            // the largest singular value sigma_max of B_hat

            sigma_min = fabs(mu[i_start]);
            for (int j = i_start; j <= i_end; j++)
            {
                if (sigma_min > fabs(mu[j]))
                {
                    sigma_min = fabs(mu[j]);
                }
            }

            sigma_max = fabs(s[i_start]);
            for (int j = i_start; j <= i_end; j++)
            {
                if (sigma_max < fabs(s[j]))
                {
                    sigma_max = fabs(s[j]);
                }
            }
            for (int j = i_start; j < i_end; j++)
            {
                if (sigma_max < fabs(e[j]))
                {
                    sigma_max = fabs(e[j]);
                }
            }

            implicitZeroShiftQR(s, e, Ut, Vt, i_start, i_end, m, n);

            if (ind == maxIter)
            {
                break;
            }

            ind++;
        }
        cout << "Iterations: " << ind << endl;

        // Make sure that all elements of s are nonnegative
        for (int i = 0; i < len; i++)
        {
            if (s[i] < 0)
            {
                for (int j = 0; j < m; j++)
                {
                    Ut[i][j] *= -1;
                }
                s[i] *= -1;
            }
        }

        // Quick sort singular values and singular vectors
        quickSort(s, Ut, Vt, 0, len - 1, "descend");

        Matrix **USV = new Matrix *[3];
        USV[0] = &UtMatrix->transpose();
        delete UtMatrix;

        USV[1] = &buildS(s, m, n);
        USV[2] = &VtMatrix->transpose();
        delete VtMatrix;

        return USV;
    }

    void quickSort(vector<double> &s, double **Ut, double **Vt, int start, int end, string order)
    {
        int i, j;
        double temp;
        i = start;
        j = end;
        temp = s[i];
        double *tempU = Ut[i];
        double *tempV = Vt[i];
        do
        {
            if (order == "ascend")
            {
                while ((s[j] > temp) && (j > i))
                    j--;
            }
            else if (order == "descend")
            {
                while ((s[j] < temp) && (j > i))
                    j--;
            }
            if (j > i)
            {
                s[i] = s[j];
                Ut[i] = Ut[j];
                Vt[i] = Vt[j];
                i++;
            }
            if (order == "ascend")
            {
                while ((s[i] < temp) && (j > i))
                    i++;
            }
            else if (order == "descend")
            {
                while ((s[i] > temp) && (j > i))
                    i++;
            }
            if (j > i)
            {
                s[j] = s[i];
                Ut[j] = Ut[i];
                Vt[j] = Vt[i];
                j--;
            }
        } while (i != j);
        s[i] = temp;
        Ut[i] = tempU;
        Vt[i] = tempV;
        i++;
        j--;
        if (start < j)
            quickSort(s, Ut, Vt, start, j, order);
        if (i < end)
            quickSort(s, Ut, Vt, i, end, order);
    }

    void implicitZeroShiftQR(vector<double> &s, vector<double> &e, double **Ut, double **Vt, int i_start, int i_end, int m, int n)
    {
        double oldcs = 1;
        double oldsn = 0;
        double f = s[i_start];
        double g = e[i_start];
        double h = 0;
        double cs = 0, sn = 0, r = 0;
        double t, tt;

        for (int i = i_start; i < i_end; i++)
        {
            // ROT(f, g, cs, sn, r)
            if (f == 0)
            {
                cs = 0;
                sn = 1;
                r = g;
            }
            else if (fabs(f) > fabs(g))
            {
                t = g / f;
                tt = sqrt(1 + t * t);
                cs = 1 / tt;
                sn = t * cs;
                r = f * tt;
            }
            else
            {
                t = f / g;
                tt = sqrt(1 + t * t);
                sn = 1 / tt;
                cs = t * sn;
                r = g * tt;
            }
            update(cs, sn, Vt[i], Vt[i + 1], n);

            if (i != i_start)
            { // Note that i != i_start rather than i != 0!!!
                e[i - 1] = oldsn * r;
            }

            f = oldcs * r;
            g = s[i + 1] * sn;
            h = s[i + 1] * cs;

            if (f == 0)
            {
                cs = 0;
                sn = 1;
                r = g;
            }
            else if (fabs(f) > fabs(g))
            {
                t = g / f;
                tt = sqrt(1 + t * t);
                /*if (f < 0) {
                            tt = -tt;
                        }*/
                cs = 1 / tt;
                sn = t * cs;
                r = f * tt;
            }
            else
            {
                t = f / g;
                tt = sqrt(1 + t * t);
                /*if (g < 0) {
                            tt = -tt;
                        }*/
                sn = 1 / tt;
                cs = t * sn;
                r = g * tt;
            }
            update(cs, sn, Ut[i], Ut[i + 1], m);

            s[i] = r;
            f = h;
            g = e[i + 1];
            oldcs = cs;
            oldsn = sn;
        }
        e[i_end - 1] = h * sn;
        s[i_end] = h * cs;

    }

    Matrix &buildS(vector<double> &s, int m, int n)
    {
        map<pair<int, int>, double> map;
        for (int i = 0; i < m; i++)
        {
            if (i < n)
                map.insert(make_pair(make_pair(i, i), s[i]));
        }
        return *new Matrix(m, n, map);
    }

    void update(double cs, double sn, double *V1, double *V2, int len)
    {
        double temp;
        for (int i = 0; i < len; i++)
        {
            temp = V1[i];
            V1[i] = cs * temp + sn * V2[i];
            V2[i] = -sn * temp + cs * V2[i];
        }
    }
};

bool testSVD(Matrix &A, Matrix *U, Matrix *S, Matrix *V)
{
    // Check if A = USV'
    Matrix &USV = U->mtimes(*S).mtimes(V->transpose());
    double **AData = A.getData();
    double **USVData = USV.getData();
    int m = A.getRowDimension();
    int n = A.getColumnDimension();
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (fabs(AData[i][j] - USVData[i][j]) > 0.1)
            {
                cout << "A[" << i << "][" << j << "] = " << AData[i][j]
                     << ", USV[" << i << "][" << j << "] = " << USVData[i][j] << endl;
                cout << "Test failed at index (" << i << ", " << j << ")" << endl;
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " m n" << endl;
        return 1;
    }
    int m = stoi(argv[1]), n = stoi(argv[2]);
    int size = min(m, n);

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

    Matrix *A = new Matrix(m, n);
    double **AData = A->getData();

    for (int i = 0; i < m; ++i)
    {
        file.read(reinterpret_cast<char *>(AData[i]), n * sizeof(double));
    }
    file.close();

    // Print the first 3x3 block of A
    printf("Input matrix A (%d x %d):\n", m, n);
    for (int i = 0; i < min(3, m); ++i)
    {
        for (int j = 0; j < min(3, n); ++j)
        {
            printf("%8.4f ", AData[i][j]);
        }
        printf("\n");
    }

    // Copy A to origA for later testing, use copy function
    Matrix &origA = A->copy();

    // Perform SVD
    SVD svd(*A);

    Matrix *U = svd.res_U;
    Matrix *S = svd.res_S;
    Matrix *V = svd.res_V;

    // Print first 3x3 blocks of U, S, V
    printf("U (%d x %d):\n", U->getRowDimension(), U->getColumnDimension());
    for (int i = 0; i < min(3, U->getRowDimension()); ++i)
    {
        for (int j = 0; j < min(3, U->getColumnDimension()); ++j)
        {
            printf("%8.4f ", U->getEntry(i, j));
        }
        printf("\n");
    }
    printf("S (%d x %d):\n", S->getRowDimension(), S->getColumnDimension());
    for (int i = 0; i < min(3, S->getRowDimension()); ++i)
    {
        for (int j = 0; j < min(3, S->getColumnDimension()); ++j)
        {
            printf("%8.4f ", S->getEntry(i, j));
        }
        printf("\n");
    }
    printf("V (%d x %d):\n", V->getRowDimension(), V->getColumnDimension());
    for (int i = 0; i < min(3, V->getRowDimension()); ++i)
    {
        for (int j = 0; j < min(3, V->getColumnDimension()); ++j)
        {
            printf("%8.4f ", V->getEntry(i, j));
        }
        printf("\n");
    }

    // Test if origA = USV'
    if (testSVD(origA, U, S, V))
    {
        cout << "SVD test passed!" << endl;
    }
    else
    {
        cout << "SVD test failed!" << endl;
    }

    // Write results in U_{m}_{n}.bin, S_{m}_{n}.bin, V_{m}_{n}.bin
    sprintf(filename, "../prog_output/U_%04d_%04d.bin", m, n);
    ofstream uFile(filename, ios::binary);
    // First write the dimensions
    uFile.write(reinterpret_cast<char *>(&m), sizeof(int));
    uFile.write(reinterpret_cast<char *>(&m), sizeof(int));
    // Write U matrix
    for (int i = 0; i < m; ++i)
    {
        uFile.write(reinterpret_cast<char *>(U->getData()[i]), m * sizeof(double));
    }
    uFile.close();

    printf("Saved U matrix to %s\n", filename);

    sprintf(filename, "../prog_output/S_%04d_%04d.bin", m, n);
    ofstream sFile(filename, ios::binary);
    // First write the dimensions
    sFile.write(reinterpret_cast<char *>(&size), sizeof(int));
    // Write singular values
    for (int i = 0; i < size; ++i)
    {
        sFile.write(reinterpret_cast<char *>(S->getData()[i, i]), sizeof(double));
    }
    sFile.close();

    printf("Saved S matrix to %s\n", filename);

    sprintf(filename, "../prog_output/V_%04d_%04d.bin", m, n);
    ofstream vFile(filename, ios::binary);
    // First write the dimensions
    vFile.write(reinterpret_cast<char *>(&n), sizeof(int));
    vFile.write(reinterpret_cast<char *>(&n), sizeof(int));
    // Write V matrix
    for (int i = 0; i < n; ++i)
    {
        vFile.write(reinterpret_cast<char *>(V->getData()[i]), n * sizeof(double));
    }
    vFile.close();
    printf("Saved V matrix to %s\n", filename);

    return 0;
}
