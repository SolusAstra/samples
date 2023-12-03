#pragma once
#include <vector>
#include <sutil\vec_math.h>

#include "utils.h"

template <typename T>
void vertexInit(std::vector<T>& vertices, const int& width, const int& height) {

    for (int i = 0; i < vertices.size(); i++) {
        // Generate non-repeating random integers between 0 and (xMax, yMax)
        //inc[i] = (float)i;

        vertices[i].x = (float) (rand() % width);
        vertices[i].y = (float) (rand() % height);
        if constexpr (std::is_same<T, float3>::value) {
            vertices[i].z = (float) (rand() % height);
        }
    }

}

void randomizePoints(std::vector<float3>& vertices, int N, int width, int height) {

    vertices = std::vector<float3>(N);
    vertexInit(vertices, width, height);
}

void stanfordBunny(std::vector<float3>& vertices, std::vector<int3>& indices) {

    LoadObjFile("D:/dev/archive/raytracing/Renderer/src/res/obj/bunny.obj", vertices, indices);
    int N_vtx = vertices.size();
    int N_fcs = indices.size();

    float bsize = 1.0f;
    float3 min = make_float3(std::numeric_limits<float>::max());
    float3 max = make_float3(std::numeric_limits<float>::min());


    // Modify bunny coordinates so that it's centered
    for (int i = 0; i < N_vtx; i++) {
        //vertices[i] = normalize(vertices[i]);

        vertices[i].x = vertices[i].x;
        min.x = fmin(min.x, vertices[i].x);
        min.y = fmin(min.y, vertices[i].y);
        min.z = fmin(min.z, vertices[i].z);
        max.x = fmax(max.x, vertices[i].x);
        max.y = fmax(max.y, vertices[i].y);
        max.z = fmax(max.z, vertices[i].z);
    }

    // Shift such that all coordinates are positive
    for (int i = 0; i < N_vtx; i++) {
        vertices[i].y -= min.y;
    }

    max -= min;
    float3 scale = bsize / max;
    for (int i = 0; i < N_vtx; i++) {
        vertices[i] = scale * vertices[i];
    }

}

void exemplar(std::vector<float3>& vertex, std::vector<int3>& face) {

    // Object 1
    vertex.push_back(make_float3(64.0f,  571.0f, 0.0f));
    vertex.push_back(make_float3(17.0f,  731.0f, 0.0f));
    vertex.push_back(make_float3(103.0f, 772.0f, 0.0f));
    vertex.push_back(make_float3(172.0f, 512.0f, 0.0f));
    vertex.push_back(make_float3(172.0f, 648.0f, 0.0f));
    vertex.push_back(make_float3(280.0f, 546.0f, 0.0f));
    vertex.push_back(make_float3(311.0f, 382.0f, 0.0f));
    vertex.push_back(make_float3(338.0f, 548.0f, 0.0f));
    vertex.push_back(make_float3(407.0f, 363.0f, 0.0f));
    vertex.push_back(make_float3(149.0f, 211.0f, 0.0f));
    vertex.push_back(make_float3(81.0f,  296.0f, 0.0f));
    vertex.push_back(make_float3(217.0f, 283.0f, 0.0f));
    vertex.push_back(make_float3(546.0f, 36.0f,  0.0f));
    vertex.push_back(make_float3(435.0f, 128.0f, 0.0f));
    vertex.push_back(make_float3(524.0f, 261.0f, 0.0f));
    vertex.push_back(make_float3(637.0f, 205.0f, 0.0f));
    vertex.push_back(make_float3(632.0f, 310.0f, 0.0f));
    vertex.push_back(make_float3(731.0f, 326.0f, 0.0f));
    vertex.push_back(make_float3(513.0f, 576.0f, 0.0f));
    vertex.push_back(make_float3(463.0f, 678.0f, 0.0f));
    vertex.push_back(make_float3(593.0f, 676.0f, 0.0f));
    vertex.push_back(make_float3(618.0f, 537.0f, 0.0f));
    vertex.push_back(make_float3(772.0f, 720.0f, 0.0f));
    vertex.push_back(make_float3(742.0f, 496.0f, 0.0f));

    face.push_back(make_int3(1, 2, 3));
    face.push_back(make_int3(4, 5, 6));
    face.push_back(make_int3(7, 8, 9));
    face.push_back(make_int3(10, 11, 12));
    face.push_back(make_int3(13, 14, 15));
    face.push_back(make_int3(16, 17, 18));
    face.push_back(make_int3(19, 20, 21));
    face.push_back(make_int3(22, 23, 24));

    for (int k = 0; k < face.size(); k++) {

        face[k].x = face[k].x - 1;
        face[k].y = face[k].y - 1;
        face[k].z = face[k].z - 1;
    }

}