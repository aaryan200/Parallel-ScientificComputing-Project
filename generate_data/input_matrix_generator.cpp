#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>

using namespace std;

// Generate a random integer in the range [2, 1000]
int getRandomDimension() {
    return rand() % 2000 + 2;
}

// Generate a matrix with given rows and columns filled with random double values
vector<vector<double>> generateMatrix(int rows, int cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (auto &row : matrix)
        for (auto &val : row)
            val = (rand() % 1000 + 1) + (rand() % 1000) / 1000.0;
    return matrix;
}

// Write the matrix to a binary file with naming format
void writeMatrixToBinaryFile(const vector<vector<double>>& matrix, int rows, int cols) {
    char filename[64];
    sprintf(filename, "../inputs/input_matrix_%04d_%04d.bin", rows, cols);

    ofstream outFile(filename, ios::binary);
    if (!outFile) {
        cerr << "Error opening file " << filename << " for writing." << endl;
        return;
    }

    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    for (const auto& row : matrix)
        outFile.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));

    cout << "Matrix written to " << filename << " (" << rows << " x " << cols << ")\n";
}

int main() {
    srand(42); // Fixed seed for reproducibility

    ifstream input("input.in");
    if (!input) {
        cerr << "Failed to open input.in\n";
        return 1;
    }

    int numRectangular, numSquare;
    input >> numRectangular >> numSquare;
    input.close();

    cout << "Generating " << numRectangular << " rectangular matrices...\n";
    for (int i = 0; i < numRectangular; ++i) {
        int rows = getRandomDimension();
        int cols = getRandomDimension();
        auto matrix = generateMatrix(rows, cols);
        writeMatrixToBinaryFile(matrix, rows, cols);
    }

    cout << "\nGenerating " << numSquare << " square matrices...\n";
    for (int i = 0; i < numSquare; ++i) {
        int size = getRandomDimension();
        auto matrix = generateMatrix(size, size);
        writeMatrixToBinaryFile(matrix, size, size);
    }

    return 0;
}
