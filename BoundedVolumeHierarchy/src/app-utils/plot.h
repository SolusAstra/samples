#pragma once
#include <SFML/Graphics.hpp>
#include "env/Particle.h"
#include "BVH.h"

#include "scene/Camera.h"

#include "env/Triangle.h"

template <sf::PrimitiveType T>
sf::VertexArray generateSFMLPrimitives(const std::vector<float3>& vertex, const std::vector<int3>& face, Trace::Camera camera) {

    return nullptr;
}

sf::Color shadePrimitives(int idx, const std::vector<float3>& vertex, const std::vector<int3>& face, Trace::Camera camera, float medianDistance) {

    // Light properties
    float3 lightDir = normalize(make_float3(-1.0f, 1.0f, 0.0f)); // Example light direction
    float3 ambientColor = make_float3(0.2f, 0.2f, 0.2f); // Ambient light
    float3 baseColor = make_float3(100.0f / 255.0f); // Base color of the surface

    // Get triangle properties
    float3 target = computeBarycenter(vertex, face, idx);
    float3 normal = computeNormal(vertex, face, idx);

    // Ray from camera to target
    float3 origin = camera.getPosition();
    float3 direction = normalize(target - origin);

    // Compute lighting
    float cosTheta = dot(normal, -lightDir); // Dot product with negative light direction
    float lightIntensity = fmaxf(cosTheta, 0.0f); // Diffuse intensity
    float3 diffuseColor = baseColor * lightIntensity; // Diffuse reflection
    float3 color = ambientColor + diffuseColor; // Total color with ambient and diffuse

    // Clamp color components
    color.x = fminf(color.x, 1.0f);
    color.y = fminf(color.y, 1.0f);
    color.z = fminf(color.z, 1.0f);

    // Convert to sf::Color
    sf::Color litColor = sf::Color(color.x * 255.0f, color.y * 255.0f, color.z * 255.0f);

    // Check distance to apply color
    float distanceTR = length(target - origin);
    return (distanceTR <= medianDistance) ? litColor : sf::Color(0, 0, 0, 0);
}

sf::Color shadePrimitive(int idx, const std::vector<float3>& vertex, const std::vector<int3>& face, Trace::Camera camera, float medianDistance) {
    // Light properties
    float3 lightDir = normalize(make_float3(-1.0f, 1.0f, 0.0f)); // Example light direction
    float3 ambientColor = make_float3(0.2f, 0.2f, 0.2f); // Ambient light
    float3 baseColor = make_float3(100.0f / 255.0f); // Base color of the surface

    // Get triangle properties
    float3 target = computeBarycenter(vertex, face, idx);
    float3 normal = computeNormal(vertex, face, idx);

    // Ray from camera to target
    float3 origin = camera.getPosition();
    float3 direction = normalize(target - origin);

    // Compute lighting
    float cosTheta = dot(normal, -lightDir); // Dot product with negative light direction
    float lightIntensity = fmaxf(cosTheta, 0.0f); // Diffuse intensity
    float3 diffuseColor = baseColor * lightIntensity; // Diffuse reflection
    float3 color = ambientColor + diffuseColor; // Total color with ambient and diffuse

    // Clamp color components
    color.x = fminf(color.x, 1.0f);
    color.y = fminf(color.y, 1.0f);
    color.z = fminf(color.z, 1.0f);

    // Convert to sf::Color
    sf::Color litColor = sf::Color(color.x * 255.0f, color.y * 255.0f, color.z * 255.0f);

    // Check distance to apply color
    float distanceTR = length(target - origin);
    return (distanceTR <= medianDistance) ? litColor : sf::Color(0, 0, 0, 0);
}


template <>
sf::VertexArray generateSFMLPrimitives<sf::PrimitiveType::Triangles>(const std::vector<float3>& vertex, const std::vector<int3>& face, Trace::Camera camera) {

    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;


    int nFaces = face.size();
    sf::VertexArray sfTriangles(sf::PrimitiveType::Triangles, nFaces * 3);

    // Example light direction
    float3 lightDir = normalize(make_float3(1.0f, -1.0f, 0.0f));
    float3 baseColor = make_float3(100.0f);




    std::vector<std::pair<float, int>> distanceIndexPairs;
    for (int i = 0; i < nFaces; i++) {
        float3 barycenter = computeBarycenter(vertex, face, i);
        float dist = length(barycenter - camera.getPosition());
        distanceIndexPairs.push_back(std::make_pair(dist, i));
    }
    // Sort based on distance
    std::sort(distanceIndexPairs.begin(), distanceIndexPairs.end());
    float medianDistance = distanceIndexPairs[nFaces / 2].first;

    for (int k = 0; k < nFaces; k++) {

        sf::Color color = shadePrimitive(k, vertex, face, camera, medianDistance);

        float3 A = vertex[face[k].x];
        float3 B = vertex[face[k].y];
        float3 C = vertex[face[k].z];
        float2 vrt_px_0 = camera.worldToPixel(A, (float) WIDTH, (float) HEIGHT);
        float2 vrt_px_1 = camera.worldToPixel(B, (float) WIDTH, (float) HEIGHT);
        float2 vrt_px_2 = camera.worldToPixel(C, (float) WIDTH, (float) HEIGHT);

        // Set vertices for the triangle
        sfTriangles[k * 3 + 0] = sf::Vertex(sf::Vector2f(vrt_px_0.x, vrt_px_0.y), color);
        sfTriangles[k * 3 + 1] = sf::Vertex(sf::Vector2f(vrt_px_1.x, vrt_px_1.y), color);
        sfTriangles[k * 3 + 2] = sf::Vertex(sf::Vector2f(vrt_px_2.x, vrt_px_2.y), color);
    }

    return sfTriangles;
}

template <>
sf::VertexArray generateSFMLPrimitives<sf::PrimitiveType::Points>(const std::vector<float3>& vertex, const std::vector<int3>& face, Trace::Camera camera) {

    int nP = vertex.size();

    sf::VertexArray sfParticles(sf::PrimitiveType::Points);
    for (int i = 0; i < nP; i++) {
        sfParticles.append(sf::Vertex(sf::Vector2f(vertex[i].x, vertex[i].y), sf::Color::White));
    }

    return sfParticles;
}


struct NodeState {

    int currentNode = 1;
    std::vector<int> previousNodes;

    void push() {
        previousNodes.push_back(currentNode);
    }

    void pop() {
        previousNodes.pop_back();
    }

    int size() {
        return previousNodes.size();
    }

    bool empty() {
        return previousNodes.empty();
    }

};
//
//
//void updateTransparencyOfBox(int boxIdx, sf::VertexArray& sfWireframe, const float transparency) {
//    for (int i = 0; i < 8; i++) {
//        sfWireframe[boxIdx * 8 + i].color.a = transparency * 255;
//    }
//}
//
//
////void dynamicTransparencyUpdateOfDepth(BVH* bvh, std::vector<float>& transparency, const int viewNode) {
////    // Recursively set transparency of all nodes below the desired node
////    std::fill(transparency.begin(), transparency.end(), 0.0f);
////    recursiveNodeSearchSetTransparency(bvh, viewNode, transparency);
////}
//
//void updateVerticesOfBox(int boxIdx, sf::VertexArray& sfWireframe, const std::vector<float2>& wireframeVertices) {
//    std::vector<int> idx = { 0, 1, 1, 2, 2, 3, 3, 0 };
//    for (int i = 0; i < 8; i++) {
//        sfWireframe[boxIdx * 8 + i].position = sf::Vector2f(
//            wireframeVertices[boxIdx * 4 + idx[i]].x,
//            wireframeVertices[boxIdx * 4 + idx[i]].y);
//    }
//}
//
//
//
////void recursiveNodeSearchSetTransparency(BVH* bvh, const int NodeIdx, std::vector<float>& transparency) {
////
////    // Get node ptr
////    Node* node = &bvh->node[NodeIdx];
////    if (node->isLeaf) { return; }
////
////    // Set transparency of current node to 1
////    transparency[NodeIdx] = 1.0f;
////
////    // Call recursive function on children
////    int leftChildIdx = node->branchIdx[0];
////    int rightChildIdx = node->branchIdx[1];
////    recursiveNodeSearchSetTransparency(bvh, leftChildIdx, transparency);
////    recursiveNodeSearchSetTransparency(bvh, rightChildIdx, transparency);
////}
//
//
//void generateWireFrame(BVH* bvh, std::vector<float2>& wireframeVertices, std::vector<float>& transparency, Trace::Camera cam) {
//
//    size_t nthInernalNode = -1;
//    for (int i = 0; i < bvh->size; i++) {
//
//        // Get node ptr
//        Node* node = &bvh->node[i];
//        AABB* box = &bvh->bbox[i];
//        if (node->isLeaf) { continue; }
//
//        // Get box
//
//        //AABB<float3>* box = node->box;
//        float3 min = box->min;
//        float3 max = box->max;
//
//        float3 min_c = cam.worldToCameraFrame(min);
//        min_c = (min_c + 0.5f) * 800.0f;
//
//        float3 max_c = cam.worldToCameraFrame(max);
//        max_c = (max_c + 0.5f) * 800.0f;
//
//        // Set vertices, indices are nthInternalNode * 4 + 0, 1, 2, 3
//        nthInernalNode++;
//        wireframeVertices[(nthInernalNode) * 4 + 0] = make_float2(min_c.x, min_c.y);
//        wireframeVertices[(nthInernalNode) * 4 + 1] = make_float2(max_c.x, min_c.y);
//        wireframeVertices[(nthInernalNode) * 4 + 2] = make_float2(max_c.x, max_c.y);
//        wireframeVertices[(nthInernalNode) * 4 + 3] = make_float2(min_c.x, max_c.y);
//
//        // Set transparency to be proportional to the depth of the node
//        int nodeDepth = node->depth;
//        transparency[nthInernalNode] = 0.5f * ((float) nodeDepth / (float) bvh->depth);
//    }
//}
//
//
//sf::VertexArray createWireFrame(BVH& bvh,
//    std::vector<float>& transparency, int nP, Trace::Camera cam) {
//
//    // Wireframe properties
//    sf::Color wireColor = sf::Color::Cyan;
//    wireColor.a = 0.2;
//    //size_t nWireframeVertices = 8 * (bvh.size - nP);
//
//    // Actually, this duplicates the vertices. The real is 4 * (bvh.size - nP)
//    size_t nWireframeVertices = 4 * (bvh.size - nP);
//
//    // Generate wireframe
//    std::vector<float2> wireframeVertices(nWireframeVertices);
//    generateWireFrame(&bvh, wireframeVertices, transparency, cam);
//
//    // Allocate memory for the wireframe
//    int nBoxes = bvh.size - nP;
//    int nWireframe = 8 * nBoxes;
//    sf::VertexArray sfWireframe(sf::PrimitiveType::Lines, nWireframe);
//    for (int i = 0; i < nWireframe; i++) {
//        sfWireframe[i].color = wireColor;
//    }
//
//    for (size_t i = 0; i < nBoxes; i++) {
//
//        // Update box position
//        updateVerticesOfBox(i, sfWireframe, wireframeVertices);
//
//        // Update box transparency
//        updateTransparencyOfBox(i, sfWireframe, transparency[i]);
//    }
//
//    return sfWireframe;
//}
//
////template <typename T>
////void generateWireFrameN(BVH<T>* bvh, std::vector<float2>& wireframeVertices, std::vector<int>& path, Trace::Camera cam) {
////
////    size_t nthInernalNode = -1;
////    for (int i = 0; i < path.size(); i++) {
////
////        // Get node ptr
////        Node<T>* node = &bvh->node[path[i]];
////        AABB<T>* box = &bvh->bbox[path[i]];
////        if (node->isLeaf) { continue; }
////
////        // Get box
////
////        //AABB<float3>* box = node->box;
////        float3 min = box->min;
////        float3 max = box->max;
////
////        float3 min_c = cam.worldToCameraFrame(min);
////        min_c = (min_c + 0.5f) * 800.0f;
////
////        float3 max_c = cam.worldToCameraFrame(max);
////        max_c = (max_c + 0.5f) * 800.0f;
////
////        // Set vertices, indices are nthInternalNode * 4 + 0, 1, 2, 3
////        nthInernalNode++;
////        wireframeVertices[(nthInernalNode) * 4 + 0] = make_float2(min_c.x, min_c.y);
////        wireframeVertices[(nthInernalNode) * 4 + 1] = make_float2(max_c.x, min_c.y);
////        wireframeVertices[(nthInernalNode) * 4 + 2] = make_float2(max_c.x, max_c.y);
////        wireframeVertices[(nthInernalNode) * 4 + 3] = make_float2(min_c.x, max_c.y);
////
////    }
////}
//
//
////sf::VertexArray createWireFrameN(BVH& bvh,
////    std::vector<float>& transparency, std::vector<int>& path, Trace::Camera cam) {
////
////
////    // Wireframe properties
////    sf::Color wireColor = sf::Color::Cyan;
////    wireColor.a = 0.2;
////
////    // Allocate memory for the wireframe
////    int nBoxes = path.size();
////    int nWireframe = 8 * nBoxes;
////    sf::VertexArray sfWireframe(sf::PrimitiveType::Lines, nWireframe);
////    for (int i = 0; i < nWireframe; i++) {
////        sfWireframe[i].color = wireColor;
////    }
////
////
////    // Actually, this duplicates the vertices. The real is 4 * (bvh.size - nP)
////    size_t nWireframeVertices = 4 * path.size();
////
////    // Generate wireframe
////    std::vector<float2> wireframeVertices(nWireframeVertices);
////    generateWireFrameN(&bvh, wireframeVertices, path, cam);
////
////
////    
////    //size_t nWireframeVertices = 8 * (bvh.size - nP);
////
////    for (size_t i = 0; i < nBoxes; i++) {
////
////        // Update box position
////        updateVerticesOfBox(i, sfWireframe, wireframeVertices);
////
////        // Update box transparency
////        updateTransparencyOfBox(i, sfWireframe, transparency[i]);
////    }
////
////    return sfWireframe;
////}
//
//
//
//
////// Processing user interaction
////void navigateNodeStructure(sf::Event& event, sf::RenderWindow& window, NodeState& nodeState, BVH& bvh, 
////    std::vector<float>& transparency, sf::VertexArray& sfWireframe) {
////    // Close window: exit
////    if (event.type == sf::Event::Closed) {
////        window.close();
////    }
////
////    // Check for key press events
////    if (event.type == sf::Event::KeyPressed) {
////
////        bool leftPressed = event.key.code == sf::Keyboard::Left;
////        bool rightPressed = event.key.code == sf::Keyboard::Right;
////        bool downPressed = event.key.code == sf::Keyboard::Down;
////
////        if (leftPressed || rightPressed) {
////
////            if (nodeState.size() == 0) {
////                nodeState.push();
////            }
////            if (nodeState.currentNode == nodeState.previousNodes[nodeState.size() - 1]) {
////                if (nodeState.size() > 1) {
////                    nodeState.pop();
////                }
////                //previousNodes.pop_back();
////            }
////            else {
////                nodeState.push(); // Store the current node before updating
////            }
////
////        }
////
////        if (downPressed) {
////
////            if (!nodeState.empty()) { // If we have previously visited nodes
////
////                if (nodeState.size() > 1) {
////                    nodeState.currentNode = nodeState.previousNodes[nodeState.size() - 2];
////                    nodeState.pop();
////
////                }
////                else if (nodeState.size() == 1) {
////                    nodeState.currentNode = nodeState.previousNodes[nodeState.size() - 1];
////                    nodeState.pop();
////                }
////                else {
////                    nodeState.currentNode = 1;
////                }
////            }
////            else { // If we're already at the root node
////                nodeState.currentNode = 1; // Reset the nodes
////                nodeState.push(); // Store the root node
////            }
////
////        }
////
////        if (leftPressed) {
////            bool isLeaf = bvh.node[nodeState.currentNode].isLeaf;
////
////            if (!isLeaf) {
////                nodeState.currentNode = bvh.node[nodeState.currentNode].branchIdx[0];
////                std::cout << "Moved to left node: " << nodeState.currentNode << std::endl;
////            }
////        }
////
////        if (rightPressed) {
////            bool isLeaf = bvh.node[nodeState.currentNode].isLeaf;
////
////            if (!isLeaf) {
////                nodeState.currentNode = bvh.node[nodeState.currentNode].branchIdx[1];
////                std::cout << "Moved to right node: " << nodeState.currentNode << std::endl;
////            }
////        }
////
////        // Update transparency vector
////        dynamicTransparencyUpdateOfDepth(&bvh, transparency, nodeState.currentNode);
////
////        // Now we have the correct transparency vector
////        // Update the transparency of the box
////        int nBoxes = sfWireframe.getVertexCount() / 8;
////        for (size_t i = 0; i < nBoxes; i++) {
////            updateTransparencyOfBox(i, sfWireframe, transparency[i]);
////        }
////
////        // Good for now
////    }
////}