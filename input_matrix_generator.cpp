#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

// Generate a random integer in the range [2, 1000]
int getRandomDimension() {
    return rand() % 999 + 2;
}

// Function to generate a rectangular matrix with given rows and columns
vector<vector<double>> generateRectangularMatrix(int rows, int cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Generate random double values between 1.0 and 1000.0
            matrix[i][j] = (double)(rand() % 1000 + 1) + (rand() % 1000) / 1000.0;
        }
    }
    return matrix;
}

// Function to generate a square matrix of a given size
vector<vector<double>> generateSquareMatrix(int size) {
    return generateRectangularMatrix(size, size);
}

// Function to write a matrix to a binary file
void writeMatrixToBinaryFile(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    char filename[64];
    sprintf(filename, "input_matrix_%04d_%04d.bin", rows, cols);
    
    ofstream outFile(filename, ios::binary);
    if (!outFile) {
        cerr << "Error opening file " << filename << " for writing." << endl;
        return;
    }
    
    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    
    for (int i = 0; i < rows; ++i) {
        outFile.write(reinterpret_cast<const char*>(matrix[i].data()), cols * sizeof(double));
    }
    
    outFile.close();
    cout << "Matrix written to " << filename << endl;
}

int main() {
    srand(42);

    cout << "\nGenerating 10 rectangular matrices with random dimensions [2,1000]...\n";
    for (int i = 0; i < 10; ++i) {
        int rows = getRandomDimension();
        int cols = getRandomDimension();
        
        cout << "Rectangular Matrix " << i + 1 << " (" 
             << rows << " x " << cols << "):" << endl;
        auto matrix = generateRectangularMatrix(rows, cols);
        writeMatrixToBinaryFile(matrix);
    }

    cout << "\nGenerating 3 square matrices with random dimensions [2,1000]...\n";
    for (int i = 0; i < 3; ++i) {
        int size = getRandomDimension();
        
        cout << "Square Matrix " << i + 1 << " (" 
             << size << " x " << size << "):" << endl;
        auto matrix = generateSquareMatrix(size);
        writeMatrixToBinaryFile(matrix);
    }

    return 0;
}
