#include <iostream>
#include <dirent.h>
#include <cstring>
#include <cstdio>

using namespace std;

bool isBinFile(const string &filename) {
    const string ext = ".bin";
    if (filename.size() >= ext.size()) {
        return filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0;
    }
    return false;
}

int main() {
    DIR *dir = opendir(".");
    if (!dir) {
        perror("Failed to open directory.");
        return EXIT_FAILURE;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        string filename = entry->d_name;

        if (isBinFile(filename)) {
            if (remove(filename.c_str()) == 0) {
                cout << "Deleted: " << filename << endl;
            } else {
                perror(("Failed to delete: " + filename).c_str());
            }
        }
    }

    closedir(dir);
    return 0;
}
