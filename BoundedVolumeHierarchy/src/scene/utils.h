#pragma once
#include <sutil/vec_math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Load an OBJ file into vertex and index buffers
void LoadObjFile(const char* filename, std::vector<float3>& vertices, std::vector<int3>& indices) {
    std::ifstream file(filename);

    if (!file) {
        // Failed to open the file
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v") {
            // Parse vertex position
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(make_float3(x, y, z));
        }
        else if (token == "f") {
            // Parse triangle indices
            std::string vertex1, vertex2, vertex3;
            iss >> vertex1 >> vertex2 >> vertex3;

            // Process each vertex in the face
            int i1 = std::stoi(vertex1.substr(0, vertex1.find("//")));
            int i2 = std::stoi(vertex2.substr(0, vertex2.find("//")));
            int i3 = std::stoi(vertex3.substr(0, vertex3.find("//")));

            indices.push_back(make_int3(i1 - 1, i2 - 1, i3 - 1)); // OBJ indices are 1-based
        }
    }
}
