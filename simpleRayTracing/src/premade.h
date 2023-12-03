#pragma once
#include <vector>
#include <sutil\vec_math.h>
#include "utils\util.cuh"

__forceinline __host__ void stanfordBunny(std::vector<float3>& vertices, std::vector<int3>& indices) {

    int N_fcs_i = indices.size();
    int N_vtx_i = vertices.size();

    LoadObjFile("D:/dev/archive/raytracing/Renderer/src/res/obj/stanford.obj", vertices, indices);
    int N_vtx = vertices.size();
    int N_fcs = indices.size();


    const float height = 10.0f;
    const float scaleFactor = height / (50.0f * 0.02f);
    float minY = std::numeric_limits<float>::max();

    for (int i = N_vtx_i; i < N_vtx; i++) { vertices[i] *= scaleFactor; }
    for (int i = N_vtx_i; i < N_vtx; i++) { minY = fmin(minY, vertices[i].y); }
    for (int i = N_vtx_i; i < N_vtx; i++) { vertices[i].y -= minY; }

    //float minX = std::numeric_limits<float>::max();
    //for (int i = N_vtx_i; i < N_vtx; i++) { minX = fmin(minX, vertices[i].x); }
    //for (int i = N_vtx_i; i < N_vtx; i++) { vertices[i].x -= minX; }

    //float minZ = std::numeric_limits<float>::max();
    //for (int i = N_vtx_i; i < N_vtx; i++) { minZ = fmin(minZ, vertices[i].z); }
    //for (int i = N_vtx_i; i < N_vtx; i++) { vertices[i].z -= minZ; }


    for (int i = N_fcs_i; i < N_fcs; i++) {
        indices[i].x = indices[i].x + N_vtx_i;
        indices[i].y = indices[i].y + N_vtx_i;
        indices[i].z = indices[i].z + N_vtx_i;
    }
}


__forceinline __host__ void addRectangleXY(std::vector<float3>& vertex, std::vector<int3>& face,
    float xMin, float xMax, float yMin, float yMax, float zLevel) {

    int idx = vertex.size();

    // Define the vertices
    vertex.push_back(make_float3(xMin, yMin, zLevel));
    vertex.push_back(make_float3(xMin, yMax, zLevel));
    vertex.push_back(make_float3(xMax, yMax, zLevel));
    vertex.push_back(make_float3(xMax, yMin, zLevel));

    // Define indices
    face.push_back(make_int3(idx + 0, idx + 1, idx + 2));
    face.push_back(make_int3(idx + 0, idx + 2, idx + 3));
}

__forceinline __host__ void addRectangleXZ(std::vector<float3>& vertex, std::vector<int3>& face,
    float xMin, float xMax, float zMin, float zMax, float yLevel) {

    int idx = vertex.size();

    // Define the vertices
    vertex.push_back(make_float3(xMin, yLevel, zMin));
    vertex.push_back(make_float3(xMin, yLevel, zMax));
    vertex.push_back(make_float3(xMax, yLevel, zMax));
    vertex.push_back(make_float3(xMax, yLevel, zMin));

    // Define indices
    face.push_back(make_int3(idx + 0, idx + 1, idx + 2));
    face.push_back(make_int3(idx + 0, idx + 2, idx + 3));
}

__forceinline __host__ void addRectangleYZ(std::vector<float3>& vertex, std::vector<int3>& face,
    float yMin, float yMax, float zMin, float zMax, float xLevel) {

    int idx = vertex.size();

    // Define the vertices
    vertex.push_back(make_float3(xLevel, yMin, zMin));
    vertex.push_back(make_float3(xLevel, yMin, zMax));
    vertex.push_back(make_float3(xLevel, yMax, zMax));
    vertex.push_back(make_float3(xLevel, yMax, zMin));

    // Define indices
    face.push_back(make_int3(idx + 0, idx + 1, idx + 2));
    face.push_back(make_int3(idx + 0, idx + 2, idx + 3));
}

__forceinline __host__ void addBox(std::vector<float3>& vertex, std::vector<int3>& face,
    float3 minPoint, float3 maxPoint) {
    // Add top
    addRectangleXZ(vertex, face, minPoint.x, maxPoint.x, minPoint.z, maxPoint.z, maxPoint.y);
    // Add bottom
    addRectangleXZ(vertex, face, minPoint.x, maxPoint.x, minPoint.z, maxPoint.z, minPoint.y);
    // Add front
    addRectangleXY(vertex, face, minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, maxPoint.z);
    // Add back
    addRectangleXY(vertex, face, minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z);
    // Add left
    addRectangleYZ(vertex, face, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z, minPoint.x);
    // Add right
    addRectangleYZ(vertex, face, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z, maxPoint.x);
}

__forceinline __host__ float randomFloat(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

__forceinline __host__ void initCornellBox(std::vector<float3>& vertex, std::vector<int3>& face)
{

    addRectangleYZ(vertex, face, 0.0f, 555.0f, 0.0f, 555.0f, 555.0f);    // Red wall
    addRectangleYZ(vertex, face, 0.0f, 555.0f, 0.0f, 555.0f, 0.0f);    // Green wall
    addRectangleXZ(vertex, face, 0.0f, 555.0f, 178.0f, 378.0f, 554.0f);// Light
    addRectangleXZ(vertex, face, 0.0f, 555.0f, 0.0f, 555.0f, 0.0f);    // Floor
    addRectangleXZ(vertex, face, 0.0f, 555.0f, 0.0f, 555.0f, 555.0f);  // Ceiling
    addRectangleXY(vertex, face, 0.0f, 555.0f, 0.0f, 555.0f, 555.0f); // Back wall

    // Parameters for boxes within Cornell Box
    float minBoxSize = 20.0f;
    float maxBoxSize = 100.0f;
    float minCoord = 10.0f;
    float maxCoord = 540.0f; // Slightly less than the box size to avoid overlap with walls

    for (int i = 0; i < 0; ++i) {
        float x0 = randomFloat(minCoord, maxCoord - maxBoxSize);
        float y0 = 0.0f; // Boxes sit on the floor
        float z0 = randomFloat(minCoord, maxCoord - maxBoxSize);

        float x1 = x0 + randomFloat(minBoxSize, maxBoxSize);
        float y1 = y0 + randomFloat(minBoxSize, maxBoxSize);
        float z1 = z0 + randomFloat(minBoxSize, maxBoxSize);

        addBox(vertex, face, make_float3(x0, y0, z0), make_float3(x1, y1, z1));
    }

}
