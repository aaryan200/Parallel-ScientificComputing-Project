#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <Eigen/Dense>

using namespace std;

// Helper function to write an Eigen matrix (MatrixXd) to a binary file using FILE*.
// File format: [int rows][int cols][data in row-major order (double)]
void writeEigenMatrixToBinaryFile(const Eigen::MatrixXd &mat, const string &filename) {
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename.c_str());
        return;
    }
    int rows = (mat.rows());
    int cols = (mat.cols());
    fwrite(&rows, sizeof(int), 1, f);
    fwrite(&cols, sizeof(int), 1, f);
    // Convert to row-major order for consistent writing.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matRowMajor = mat;
    fwrite(matRowMajor.data(), sizeof(double), rows * cols, f);
    fclose(f);
}

// Helper function to write an Eigen vector (VectorXd) to a binary file using FILE*.
// File format: [int length][data (double)]
void writeEigenVectorToBinaryFile(const Eigen::VectorXd &vec, const string &filename) {
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename.c_str());
        return;
    }
    int len = (vec.size());
    fwrite(&len, sizeof(int), 1, f);
    fwrite(vec.data(), sizeof(double), len, f);
    fclose(f);
}

// Check if a given filename starts with "input_matrix_" and ends with ".bin"
bool isInputMatrixFile(const string &fileName) {
    const string prefix = "input_matrix_";
    const string suffix = ".bin";
    if (fileName.size() < prefix.size() + suffix.size())
        return false;
    if (fileName.substr(0, prefix.size()) != prefix)
        return false;
    if (fileName.substr(fileName.size() - suffix.size(), suffix.size()) != suffix)
        return false;
    return true;
}

int main() {
    // Open current directory using C-style directory handling.
    DIR *dir = opendir(".");
    if (!dir) {
        perror("opendir");
        return EXIT_FAILURE;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        string fileName = entry->d_name;
        if (!isInputMatrixFile(fileName))
            continue;

        // Open the input file for reading using FILE*.
        FILE *f = fopen(fileName.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "Error opening file %s for reading.\n", fileName.c_str());
            continue;
        }
        
        // Read the dimensions (first two ints).
        int rows = 0, cols = 0;
        if (fread(&rows, sizeof(int), 1, f) != 1 || fread(&cols, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Error reading dimensions from file %s.\n", fileName.c_str());
            fclose(f);
            continue;
        }
        
        // Read the matrix data (stored as ints in row-major order).
        vector<int> rawData(rows * cols);
        size_t itemsRead = fread(rawData.data(), sizeof(int), rows * cols, f);
        if (itemsRead != (size_t)(rows * cols)) {
            fprintf(stderr, "Error reading matrix data from file %s.\n", fileName.c_str());
            fclose(f);
            continue;
        }
        fclose(f);

        // Convert the raw integer data into an Eigen::MatrixXd (using double values)
        Eigen::MatrixXd matrix(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                matrix(i, j) = (double)(rawData[i * cols + j]);

        // Compute the SVD using Eigen's JacobiSVD with full U and V.
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::VectorXd singularValues = svd.singularValues();

        // Prepare output file names using 4-digit zero-padded dimensions.
        char uFilename[64], vFilename[64], sFilename[64];
        sprintf(uFilename, "u_%04d_%04d.bin", rows, cols);
        sprintf(vFilename, "v_%04d_%04d.bin", rows, cols);
        sprintf(sFilename, "singular_%04d_%04d.bin", rows, cols);

        // Write SVD components to their respective binary files using C-style I/O.
        writeEigenMatrixToBinaryFile(U, uFilename);
        writeEigenMatrixToBinaryFile(V, vFilename);
        writeEigenVectorToBinaryFile(singularValues, sFilename);

        cout << "Processed file: " << fileName << " (" << rows << " x " << cols << ")\n";
        cout << "  U written to: " << uFilename << "\n";
        cout << "  V written to: " << vFilename << "\n";
        cout << "  Singular values written to: " << sFilename << "\n" << endl;
    }

    closedir(dir);
    return 0;
}
