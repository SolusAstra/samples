#pragma once
#include <SFML/Graphics.hpp>
#include "env/Particle.h"
#include "BVH.h"
#include "scene/Camera.h"
#include "env/Triangle.h"


void generateWireFrames(AccelStruct::BVH* bvh, std::vector<float3>& wireframeVertices) {
    // WORLD COORDINATES

    int nVerticesPerBox = 24;
    unsigned int indices[24] = {
    0, 1, 1, 2, 2, 3, 3, 0, // Front face
    4, 5, 5, 6, 6, 7, 7, 4, // Back face
    0, 4, 1, 5, 2, 6, 3, 7  // Connecting lines
    };

    for (int k = 0; k < bvh->size; k++) {

        // Get node ptr
        Node* node = &bvh->node[k];
        AABB* box = &bvh->bbox[k];
        if (node->isLeaf) { continue; }

        // Get box
        float3 min = box->min;
        float3 max = box->max;

        std::vector<float3> baseVertices(8);
        baseVertices[0] = make_float3(min.x, min.y, min.z); // Front-bottom-left
        baseVertices[1] = make_float3(max.x, min.y, min.z); // Front-bottom-right
        baseVertices[2] = make_float3(max.x, max.y, min.z); // Front-top-right
        baseVertices[3] = make_float3(min.x, max.y, min.z); // Front-top-left
        baseVertices[4] = make_float3(min.x, min.y, max.z); // Back-bottom-left
        baseVertices[5] = make_float3(max.x, min.y, max.z); // Back-bottom-right
        baseVertices[6] = make_float3(max.x, max.y, max.z); // Back-top-right
        baseVertices[7] = make_float3(min.x, max.y, max.z); // Back-top-left


        for (int j = 0; j < nVerticesPerBox; j++) {
            wireframeVertices[k * nVerticesPerBox + j] = baseVertices[indices[j]];
        }

    }

}

void generateWireFrames(AccelStruct::BVH* bvh, std::vector<float3>& wireframeVertices, std::vector<int>& path) {
    // WORLD COORDINATES


    int nVerticesPerBox = 24;
    unsigned int indices[24] = {
    0, 1, 1, 2, 2, 3, 3, 0, // Front face
    4, 5, 5, 6, 6, 7, 7, 4, // Back face
    0, 4, 1, 5, 2, 6, 3, 7  // Connecting lines
    };

    for (int k = 0; k < path.size(); k++) {

        // Get node ptr
        Node* node = &bvh->node[path[k]];
        AABB* box = &bvh->bbox[path[k]];
        if (node->isLeaf) { continue; }

        // Get box
        float3 min = box->min;
        float3 max = box->max;

        std::vector<float3> baseVertices(8);
        baseVertices[0] = make_float3(min.x, min.y, min.z); // Front-bottom-left
        baseVertices[1] = make_float3(max.x, min.y, min.z); // Front-bottom-right
        baseVertices[2] = make_float3(max.x, max.y, min.z); // Front-top-right
        baseVertices[3] = make_float3(min.x, max.y, min.z); // Front-top-left
        baseVertices[4] = make_float3(min.x, min.y, max.z); // Back-bottom-left
        baseVertices[5] = make_float3(max.x, min.y, max.z); // Back-bottom-right
        baseVertices[6] = make_float3(max.x, max.y, max.z); // Back-top-right
        baseVertices[7] = make_float3(min.x, max.y, max.z); // Back-top-left


        for (int j = 0; j < nVerticesPerBox; j++) {
            wireframeVertices[k * nVerticesPerBox + j] = baseVertices[indices[j]];
        }

    }




}

sf::VertexArray buildWireFrame(AccelStruct::BVH& bvh) {

    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;
    int nBoxes = bvh.bbox.size();
    int nVerticesPerBox = 8;
    int nIndicesPerBox = 12;
    int nLines = nBoxes * nIndicesPerBox;


    // Wireframe properties
    sf::Color wireColor = sf::Color::Cyan;
    wireColor.a = 60.0f;

    std::vector<float3> wireframeVertices(2 * nLines);
    sf::VertexArray sfWireframe(sf::PrimitiveType::Lines, 2 * nLines);

    generateWireFrames(&bvh, wireframeVertices);

    // TRANSFORM COORDINATES
    int nVertices = wireframeVertices.size();
    for (int k = 0; k < nVertices; k++) {

        //float2 pixCoord = cam.worldToPixel(wireframeVertices[k], (float) WIDTH, (float) HEIGHT);

        sfWireframe[k].position = sf::Vector2f(wireframeVertices[k].x, wireframeVertices[k].y);
        sfWireframe[k].color = wireColor;
    }

    return sfWireframe;
}

sf::VertexArray buildWireFrame(AccelStruct::BVH& bvh, Trace::Camera cam) {

    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;
    int nBoxes = bvh.bbox.size();
    int nVerticesPerBox = 8;
    int nIndicesPerBox = 12;
    int nLines = nBoxes * nIndicesPerBox;


    // Wireframe properties
    sf::Color wireColor = sf::Color::Cyan;
    wireColor.a = 2.0f;

    std::vector<float3> wireframeVertices(2 * nLines);
    sf::VertexArray sfWireframe(sf::PrimitiveType::Lines, 2 * nLines);

    generateWireFrames(&bvh, wireframeVertices);

    // TRANSFORM COORDINATES
    int nVertices = wireframeVertices.size();
    for (int k = 0; k < nVertices; k++) {

        float2 pixCoord = cam.worldToPixel(wireframeVertices[k], (float) WIDTH, (float) HEIGHT);

        sfWireframe[k].position = sf::Vector2f(pixCoord.x, pixCoord.y);
        sfWireframe[k].color = wireColor;
    }   

    return sfWireframe;
}

sf::VertexArray buildWireFrame(AccelStruct::BVH& bvh, std::vector<int>& path, Trace::Camera cam) {

    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;
    int nBoxes = path.size();
    int nVerticesPerBox = 8;
    int nIndicesPerBox = 12;
    int nLines = nBoxes * nIndicesPerBox;


    // Wireframe properties
    sf::Color wireColor = sf::Color::Cyan;
    wireColor.a = 50.0f;

    std::vector<float3> wireframeVertices(2 * nLines);
    sf::VertexArray sfWireframe(sf::PrimitiveType::Lines, 2 * nLines);

    generateWireFrames(&bvh, wireframeVertices, path);

    // TRANSFORM COORDINATES
    int nVertices = wireframeVertices.size();
    for (int k = 0; k < nVertices; k++) {

        float2 pixCoord = cam.worldToPixel(wireframeVertices[k], (float) WIDTH, (float) HEIGHT);
        sfWireframe[k].position = sf::Vector2f(pixCoord.x, pixCoord.y);
        sfWireframe[k].color = wireColor;
    }

    return sfWireframe;
}

sf::VertexArray buildWireFrame(AccelStruct::BVH& bvh, std::vector<int>& path) {

    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;
    int nBoxes = path.size();
    int nVerticesPerBox = 8;
    int nIndicesPerBox = 12;
    int nLines = nBoxes * nIndicesPerBox;


    // Wireframe properties
    sf::Color wireColor = sf::Color::Cyan;
    wireColor.a = 50.0f;

    std::vector<float3> wireframeVertices(2 * nLines);
    sf::VertexArray sfWireframe(sf::PrimitiveType::Lines, 2 * nLines);

    generateWireFrames(&bvh, wireframeVertices, path);

    // TRANSFORM COORDINATES
    int nVertices = wireframeVertices.size();
    for (int k = 0; k < nVertices; k++) {
        sfWireframe[k].position = sf::Vector2f(wireframeVertices[k].x, wireframeVertices[k].y);
        sfWireframe[k].color = wireColor;
    }

    return sfWireframe;
}
