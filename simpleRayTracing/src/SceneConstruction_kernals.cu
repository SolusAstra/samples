#include "SceneConstruction_kernals.h"

#include "materials\Lambertian.cuh"
#include "materials\Reflective.cuh"
#include "materials\Light.cuh"
#include "materials\Dielectric.cuh"

#include "env\PrimitiveArray.cuh"

#include "env\Sphere.cuh"
#include "env\Triangle.cuh"

#include "utils/util.cuh"
#include "tiny_obj_loader.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

__device__ float3 randF3(curandState* rand) { return make_float3(curand_uniform(rand), curand_uniform(rand), curand_uniform(rand)); }
__device__ float3 randF3(curandState* rand, float min, float max) { return (max - min) * randF3(rand) + min; }
__device__ float randFloat(curandState* rand, float min, float max) { return (max - min) * curand_uniform(rand) + min; }

__global__ void setUpPrimitiveArray_k(Trace::PrimitiveArray** d_environment, Trace::Primitive** objects) {


}


__global__ void initScene(Trace::PrimitiveArray** env, Trace::Primitive** objects)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        Trace::Material* dark = new Trace::Reflective(make_float3(0.1, 0.1, 0.1), 0.05f);
        Trace::Material* black = new Trace::Lambertian(make_float3(0.5, 0.5, 0.5));
        Trace::Material* ugly = new Trace::Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
        Trace::Material* pink = new Trace::Lambertian(make_float3(0.7f, 0.3f, 0.3f));
        Trace::Material* grey = new Trace::Reflective(make_float3(0.8f, 0.8f, 0.8f), 0.0f);

        Trace::Material* ground = new Trace::Lambertian(make_float3(0.5f, 0.5f, 0.5f));

        // Add spheres to the environment
        objects[0] = new Trace::Triangle(
            make_float3(-50.0f, 0.0f, -50.0f),
            make_float3(50.0f, 0.0f, -50.0f),
            make_float3(-50.0f, 0.0f, 50.0f), dark);

        objects[1] = new Trace::Triangle(
            make_float3(50.0f, 0.0f, -50.0f),
            make_float3(-50.0f, 0.0f, 50.0f),
            make_float3(50.0f, 0.0f, 50.0f), ugly);


        objects[2] = new Trace::Sphere(make_float3(2.0f, 1.0f, 2.0f), 1.0f, dark);
        objects[3] = new Trace::Sphere(make_float3(-2.0f, 1.0f, 1.0f), 1.0f, pink);
        objects[4] = new Trace::Sphere(make_float3(0.25f, 0.25f, 0.25f), 0.25f, grey);

        *env = new Trace::PrimitiveArray(objects, 5);
    }
}

__global__ void initFinalScene(Trace::PrimitiveArray** env, Trace::Primitive** objects)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        const int seed = 1234;
        const int sequence = 2;
        const int offset = 10;
        curandState randState;
        curand_init(seed, sequence, offset, &randState);

        Trace::Material* ground = new Trace::Lambertian(make_float3(0.5f, 0.5f, 0.5f));

        // Add spheres to the environment
        objects[0] = new Trace::Triangle(
            make_float3(-50.0f, 0.0f, -50.0f),
            make_float3(50.0f, 0.0f, -50.0f),
            make_float3(-50.0f, 0.0f, 50.0f), ground);

        objects[1] = new Trace::Triangle(
            make_float3(50.0f, 0.0f, -50.0f),
            make_float3(-50.0f, 0.0f, 50.0f),
            make_float3(50.0f, 0.0f, 50.0f), ground);

        float3* centers = new float3[3];
        centers[0] = make_float3(0.0f, 1.0f, 0.0f);
        centers[1] = make_float3(-4.0f, 1.0f, 0.0f);
        centers[2] = make_float3(4.0f, 1.0f, 0.0f);

        Trace::Material* mat1 = new Trace::Dielectric(2.9f);
        objects[2] = new Trace::Sphere(centers[0], 1.0f, mat1);

        Trace::Material* mat2 = new Trace::Lambertian(make_float3(0.4f, 0.2f, 0.1f));
        objects[3] = new Trace::Sphere(centers[1], 1.0f, mat2);

        Trace::Material* mat3 = new Trace::Reflective(make_float3(0.7f, 0.6f, 0.5f), 0.0f);
        objects[4] = new Trace::Sphere(centers[2], 1.0f, mat3);


        int objCount = 5;
        const int spread = 11;

        float offSet = 0.8f;
        for (int a = -spread; a < spread; a++) {
            for (int b = -spread; b < spread; b++) {

                // Compute center of the sphere
                float3 center = make_float3(a + 0.9f * curand_uniform(&randState), 0.2, b + 0.9f * curand_uniform(&randState));

                float d1 = length(center - centers[0]);
                float d2 = length(center - centers[1]);
                float d3 = length(center - centers[2]);

                if (d1 < 0.8f || d2 < 0.8f || d3 < 0.8f) {

                    center.x += 22.0f;
                    center.z += 22.0f;
                }

                float chooseMat = curand_uniform(&randState);
                if (chooseMat < 0.8f) {
                    // Diffuse object
                    float3 albedo = randF3(&randState) * randF3(&randState);
                    Trace::Material* mat = new Trace::Lambertian(albedo);
                    objects[objCount] = new Trace::Sphere(center, 0.2f, mat);
                    objCount++;

                }
                else {
                    float3 albedo = randF3(&randState, 0.5f, 1.0f);
                    float rough = randFloat(&randState, 0.0f, 0.5f);
                    Trace::Material* mat = new Trace::Reflective(albedo, rough);
                    objects[objCount] = new Trace::Sphere(center, 0.2f, mat);
                    objCount++;
                }
            }
        }


        *env = new Trace::PrimitiveArray(objects, 484);

    }
}

__global__ void initBunny(Trace::PrimitiveArray** env, Trace::Primitive** objects, float3* v, int3* f, int nFaces)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        Trace::Material* dark = new Trace::Reflective(make_float3(0.1, 0.1, 0.1), 0.05f);
        Trace::Material* black = new Trace::Lambertian(make_float3(0.5, 0.5, 0.5));
        Trace::Material* ugly = new Trace::Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
        Trace::Material* pink = new Trace::Lambertian(make_float3(0.7f, 0.3f, 0.3f));
        Trace::Material* grey = new Trace::Reflective(make_float3(0.8f, 0.8f, 0.8f), 0.0f);

        Trace::Material* ground = new Trace::Lambertian(make_float3(0.5f, 0.5f, 0.5f));

        // Add spheres to the environment
        objects[0] = new Trace::Triangle(
            make_float3(-50.0f, 0.0f, -50.0f),
            make_float3(50.0f, 0.0f, -50.0f),
            make_float3(-50.0f, 0.0f, 50.0f), dark);

        objects[1] = new Trace::Triangle(
            make_float3(50.0f, 0.0f, -50.0f),
            make_float3(-50.0f, 0.0f, 50.0f),
            make_float3(50.0f, 0.0f, 50.0f), ugly);

        objects[2] = new Trace::Sphere(make_float3(1.5f, 1.0f, -1.0f), 1.0f, dark);


        for (int i = 0; i < nFaces; i++) {
            Trace::Triangle* triangle = new Trace::Triangle(v[f[i].x], v[f[i].y], v[f[i].z], pink);
            objects[i + 3] = triangle;
        }

        int nObjects = nFaces + 3;


        *env = new Trace::PrimitiveArray(objects, nObjects);

    }
}

extern "C" void setUpPrimitiveArray(SceneType sceneType, Trace::Pipeline & pipeline) {

    Trace::PrimitiveArray** d_environment;
    Trace::Primitive** d_sceneObjects;

    cudaMalloc((void**) &d_environment, sizeof(Trace::PrimitiveArray*));

    Trace::Camera* d_camera;

    int numObjects;
    switch (sceneType) {
        case (SceneType::FLAT_WORLD):
        {
            float3 origin = make_float3(0.0f, 2.0f, 7.0f);
            float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);
            float vFOV = 40.0f;
            float aspectRatio = (float) pipeline.imageWidth / (float) pipeline.imageHeight;
            d_camera = new Trace::Camera(origin, lookAt, vFOV, aspectRatio);



            cudaMalloc((void**) &d_sceneObjects, 5 * sizeof(Trace::Primitive*));

            // Build flat world
            initScene << <1, 1 >> > (d_environment, d_sceneObjects);
            cudaFree(d_sceneObjects);

            break;
        }

        case (SceneType::FINAL_SCENE):
        {

            float3 origin = make_float3(13.0f, 2.0f, 3.0f);
            float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);
            float vFOV = 40.0f;
            float aspectRatio = (float) pipeline.imageWidth / (float) pipeline.imageHeight;
            d_camera = new Trace::Camera(origin, lookAt, vFOV, aspectRatio);

            cudaMalloc((void**) &d_sceneObjects, 484 * sizeof(Trace::Primitive*));

            initFinalScene << <1, 1 >> > (d_environment, d_sceneObjects);

            break;
        }

        case (SceneType::BACKROOM):
            numObjects = 20;
            break;

        case (SceneType::CORNELL_BOX):

            numObjects = 14;
            break;

        case (SceneType::CORNELL_BOX2):
            numObjects = 26;
            break;

        case (SceneType::BUNNY):
        {
            // Camera
            float3 origin = make_float3(2.0f, 2.0f, 1.0f);
            float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);
            float vFOV = 40.0f;
            float aspectRatio = (float) pipeline.imageWidth / (float) pipeline.imageHeight;
            d_camera = new Trace::Camera(origin, lookAt, vFOV, aspectRatio);





            // Load stanford bunny
            numObjects = 4968 + 3;
            float3 v[2503];
            int3 f[4968];
            const float height = 4.0f;
            const float scaleFactor = height / (50.0f * 0.02f);
            float minY = std::numeric_limits<float>::max();

            // Load stanford bunny
            LoadObjFile("D:/dev/archive/raytracing/Renderer/src/res/obj/stanford.obj", v, f);
            for (int i = 0; i < 2503; ++i) { v[i] *= scaleFactor; }
            for (int i = 0; i < 2503; i++) { minY = fmin(minY, v[i].y); }
            for (int i = 0; i < 2503; i++) { v[i].y -= minY; }

            // Allocate memory for vertices and indices on device
            float3* d_vertices;
            int3* d_indices;
            cudaMalloc((void**) &d_vertices, 2503 * sizeof(float3));
            cudaMalloc((void**) &d_indices, 4968 * sizeof(int3));

            // Copy vertices and indices from host to device
            cudaMemcpy(d_vertices, v, 2503 * sizeof(float3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, f, 4968 * sizeof(int3), cudaMemcpyHostToDevice);

            cudaMalloc((void**) &d_sceneObjects, numObjects * sizeof(Trace::Primitive*));

            initBunny << <1, 1 >> > (d_environment, d_sceneObjects, d_vertices, d_indices, 4968);

            break;
        }


    }

    pipeline.d_environment = d_environment;
    pipeline.camera = *d_camera;

}

//Camera buildScene(
//    SceneType sceneType,
//    PrimitiveArray::AccelerationStructure accelerationStructure, 
//    PrimitiveArray** environment, 
//    PrimitiveArray** objects, 
//    int width, int height, int numObjects) {
//
//    // Create scene
//    float3 position;
//    float3 focus;
//    float vFOV;
//
//
//    switch (sceneType) {
//    case (SceneType::FLAT_WORLD):
//        // Build flat world
//        initScene << <1, 1 >> > (environment, objects, numObjects, accelerationStructure);
//
//        // Set camera
//        position = make_float3(0.0f, 2.0f, 7.0f);
//        focus = make_float3(0.0f, 0.0f, 0.0f);
//        vFOV = 40.0f;
//        break;
//
//    case (SceneType::FINAL_SCENE):
//        // Build flat world
//        initFinalScene << <1, 1 >> > (environment, objects, numObjects, accelerationStructure);
//
//        // Set camera
//        position = make_float3(13.0f, 2.0f, 3.0f);
//        focus = make_float3(0.0f, 0.0f, 0.0f);
//        vFOV = 40.0f;
//        break;
//
//    case (SceneType::BACKROOM):
//        // Build backroom
//        initBackRoom << <1, 1 >> > (environment, objects, numObjects, accelerationStructure);
//
//        // Set camera
//        position = make_float3(278, 278, -800);
//        focus = make_float3(278, 278, 0);
//        vFOV = 40.0f;
//        break;
//
//    case (SceneType::CORNELL_BOX):
//        initCornellBox<<<1,1>>>(environment, objects, numObjects, accelerationStructure);
//
//        // Set camera
//        position = make_float3(278, 278, -800);
//        focus = make_float3(278, 278, 0);
//        vFOV = 40.0f;
//        
//        break;
//
//    case (SceneType::CORNELL_BOX2):
//        initCornellBox2<<<1,1>>>(environment, objects, numObjects, accelerationStructure);
//
//        // Set camera
//        position = make_float3(278, 278, -800);
//        focus = make_float3(278, 278, 0);
//        vFOV = 40.0f;
//        break;
//        
//    case (SceneType::BUNNY):
//        // Load stanford bunny
//        float3 v[2503];
//        int3 f[4968];
//        const float height = 4.0f;
//        const float scaleFactor = height / (50.0f * 0.02f);
//        float minY = std::numeric_limits<float>::max();
//        
//        // Load stanford bunny
//        LoadObjFile("C:/dev/raytracing/Renderer/src/res/obj/stanford.obj", v, f);
//        for (int i = 0; i < 2503; ++i) { v[i] *= scaleFactor; }
//        for (int i = 0; i < 2503; i++) { minY = fmin(minY, v[i].y); }
//        for (int i = 0; i < 2503; i++) { v[i].y -= minY; }
//
//        // Allocate memory for vertices and indices on device
//        float3* d_vertices;
//        int3* d_indices;
//        cudaMalloc((void**)&d_vertices, 2503 * sizeof(float3));
//        cudaMalloc((void**)&d_indices, 4968 * sizeof(int3));
//        // Copy vertices and indices from host to device
//        cudaMemcpy(d_vertices, v, 2503 * sizeof(float3), cudaMemcpyHostToDevice);
//        cudaMemcpy(d_indices, f, 4968 * sizeof(int3), cudaMemcpyHostToDevice);
//        initBunny<<<1,1>>>(environment, objects, accelerationStructure, d_vertices, d_indices, 4968);
//
//
//		// Set camera
//		position = make_float3(0.0f, 1.0f, 3.0f);
//		focus = make_float3(0.0f, -0.2f, -1.0f);
//		vFOV = 40.0f;
//		break;
//
//
//    }
//
//    return Camera(position, focus, vFOV, (float)width / (float)height);
//}


//__device__ void Icosahedron(float3* vertices, int* indices) {
//
//    const float X = 0.525731112119133606f;
//    const float Z = 0.850650808352039932f;
//    const float N = 0.0f;
//
//    float3 v[] = {
//        { -X,N,Z },
//        { X,N,Z },
//        { -X,N,-Z },
//        { X,N,-Z },
//        { N,Z,X },
//        { N,Z,-X },
//        { N,-Z,X },
//        { N,-Z,-X },
//        { Z,X,N },
//        { -Z,X,N },
//        { Z,-X,N },
//        { -Z,-X,N } };
//
//    int i[] = {
//        0,4,1,
//        0,9,4,
//        9,5,4,
//        4,5,8,
//        4,8,1,
//        8,10,1,
//        8,3,10,
//        5,3,8,
//        5,2,3,
//        2,7,3,
//        7,10,3,
//        7,6,10,
//        7,11,6,
//        11,0,6,
//        0,1,6,
//        6,1,10,
//        9,0,11,
//        9,11,2,
//        9,2,5,
//        7,2,11 };
//
//    for (int j = 0; j < 12; j++) { vertices[j] = v[j]; }
//    for (int j = 0; j < 60; j++) { indices[j] = i[j]; }
//}
//
//__device__ void shiftScaleIcosahedron(float3* vertices, float3 center, float radius, int nPoints) {
//
//    // Scale and center the vertices
//    for (int i = 0; i < nPoints; i++) {
//        vertices[i] *= radius;
//        vertices[i] += center;
//    }
//}
//
//__global__ void initTriangleScene(PrimitiveArray** env, PrimitiveArray** objects)
//{
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//        Material* dark = new Reflective(make_float3(0.1, 0.1, 0.1), 0.05f);
//        Material* black = new Lambertian(make_float3(0.5, 0.5, 0.5));
//        Material* ugly = new Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
//        Material* pink = new Lambertian(make_float3(0.7f, 0.3f, 0.3f));
//        Material* grey = new Reflective(make_float3(0.8f, 0.8f, 0.8f), 0.0f);
//
//        // Add spheres to the environment
//        objects[0] = new Triangle(
//            make_float3(-50.0f, 0.0f, -50.0f),
//            make_float3(50.0f, 0.0f, -50.0f),
//            make_float3(-50.0f, 0.0f, 50.0f), black);
//
//        objects[1] = new Triangle(
//            make_float3(50.0f, 0.0f, -50.0f),
//            make_float3(-50.0f, 0.0f, 50.0f),
//            make_float3(50.0f, 0.0f, 50.0f), ugly);
//
//        objects[2] = new Sphere(make_float3(2.0f, 1.0f, 2.0f), 1.0f, dark);
//        objects[3] = new Sphere(make_float3(-2.0f, 1.0f, 1.0f), 1.0f, pink);
//        objects[4] = new Sphere(make_float3(0.25f, 0.25f, 0.25f), 0.25f, pink);
//
//        const int numOtherObjects = 5;
//
//        // Load icosahedron
//        const int baseVertices = 12;
//        const int baseIndices = 60;
//        float3 baseVerts[baseVertices];
//        int baseIdx[baseIndices];
//        Icosahedron(baseVerts, baseIdx);
//
//        //// Subdivide the icosahedron
//        //int numSubdivisions = 1; // Choose the number of subdivisions
//        //const int numVertices = 72;
//        //const int numIndices = 240;
//        //float3 vertices[numVertices];
//        //int indices[numIndices];
//
//
//        int numSubdivisions = 1; // Choose the number of subdivisions
//
//        int numVertices = 12;
//        int numIndices = 60;
//        if (numSubdivisions == 1) {
//            numVertices = 72;
//            numIndices = 240;
//        }
//        else if (numSubdivisions == 2) {
//            numVertices = 162;
//            numIndices = 960;
//        }
//
//
//
//        const int maxNumVertices = 72;
//        const int maxNumIndices = 240;
//        float3 vertices[maxNumVertices];
//        int indices[maxNumIndices];
//
//
//
//        //for (int i = 0; i < numSubdivisions; ++i) {
//        //    numVertices *= 4;
//        //}
//
//        //// Subdivide the icosahedron
//        //int numSubdivisions = 2; // Choose the number of subdivisions
//        //const int numVertices = 162;
//        //const int numIndices = 960;
//
//
//
//
//        for (int i = 0; i < baseVertices; i++) {
//            vertices[i] = baseVerts[i];
//        }
//        for (int i = 0; i < baseIndices; i++) {
//            indices[i] = baseIdx[i];
//        }
//
//        int numV = baseVertices;
//        int numI = baseIndices;
//
//        // Subdivide the icosahedron
//        for (int i = 0; i < numSubdivisions; ++i) {
//            int prevNumVertices = numV;
//            int prevNumIndices = numI;
//            numI = 0; // Reset the number of indices for each subdivision
//
//            // Loop through the triangles and subdivide each one
//            for (int j = 0; j < prevNumIndices; j += 3) {
//                int v0 = baseIdx[j];
//                int v1 = baseIdx[j + 1];
//                int v2 = baseIdx[j + 2];
//
//                // Calculate the midpoints and normalize them to get the new vertices
//                float3 m0 = normalize((baseVerts[v0] + baseVerts[v1]) * 0.5f);
//                float3 m1 = normalize((baseVerts[v1] + baseVerts[v2]) * 0.5f);
//                float3 m2 = normalize((baseVerts[v2] + baseVerts[v0]) * 0.5f);
//
//                // Add the new vertices to the output array
//                int vm0 = numV++;
//                int vm1 = numV++;
//                int vm2 = numV++;
//                vertices[vm0] = m0;
//                vertices[vm1] = m1;
//                vertices[vm2] = m2;
//
//                // Create the four new triangles
//                indices[numI++] = v0;
//                indices[numI++] = vm0;
//                indices[numI++] = vm2;
//
//                indices[numI++] = v1;
//                indices[numI++] = vm1;
//                indices[numI++] = vm0;
//
//                indices[numI++] = v2;
//                indices[numI++] = vm2;
//                indices[numI++] = vm1;
//
//                indices[numI++] = vm0;
//                indices[numI++] = vm1;
//                indices[numI++] = vm2;
//            }
//        }
//
//        // subdivideIcosahedron(baseVerts, baseIdx, baseVertices, baseIndices, vertices, indices, numSubdivisions);
//
//         //// Load icosahedron
//         //const int numVertices = 42;
//         //const int numIdx = 216;
//         //float3 vertices[numVertices];
//         //int indices[numIdx];
//         //mediumIcosahedron(vertices, indices);
//
//         //// Load icosahedron
//         //const int numVertices = 163;
//         //const int numIdx = 240;
//         //float3 vertices[numVertices];
//         //int indices[numIdx];
//         //bigIcosahedron(vertices, indices);
//
//
//        float3 center = make_float3(0.0f, 1.0f, -1.0f);
//        float radius = 1.0f;
//        shiftScaleIcosahedron(vertices, center, radius, numVertices);
//
//
//        // Create triangles for icosahedron
//        int numTriangles = numIndices / 3;
//        for (int i = 0; i < numTriangles; i++) {
//            int index = i * 3;
//            float3 v1 = vertices[indices[index]];
//            float3 v2 = vertices[indices[index + 1]];
//            float3 v3 = vertices[indices[index + 2]];
//
//            // Compute the normal of the triangle
//            float3 e1 = v2 - v1;
//            float3 e2 = v3 - v1;
//            float3 normal = normalize(cross(e1, e2));
//
//            if (i % 2 == 0) {
//                objects[numOtherObjects + i] = new Triangle(v1, v2, v3, normal, grey);
//            }
//            else {
//                objects[numOtherObjects + i] = new Triangle(v1, v2, v3, normal, grey);
//            }
//
//
//        }
//
//        *env = new SceneObject(objects, numOtherObjects + numTriangles);
//    }
//}
//
//
//__global__ void initCornellBox(PrimitiveArray** env, PrimitiveArray** objects, const int nObjects, PrimitiveArray::AccelerationStructure accelerationStructure)
//{
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//        Material* dark = new Reflective(make_float3(0.1, 0.1, 0.1), 0.00f);
//        Material* black = new Lambertian(make_float3(0.5, 0.5, 0.5));
//        Material* ugly = new Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
//        Material* pink = new Lambertian(make_float3(0.7f, 0.3f, 0.3f));
//        Material* grey = new Reflective(make_float3(0.8f, 0.8f, 0.8f), 0.0f);
//
//        Material* green = new Lambertian(make_float3(0.12f, 0.45f, 0.15f));
//        Material* red = new Lambertian(make_float3(0.65f, 0.05f, 0.05f));
//        Material* white = new Lambertian(make_float3(1.0f, 1.0f, 1.0f));
//        Material* rodrigo = new Lambertian(make_float3(124.0f, 125.0f, 193.0f) / 255.0f);
//
//        // Soft light
//        Material* light = new Light(make_float3(0.8f, 0.8f, 0.8f));
//
//        // Build cornell box
//        Triangle::rectangleYZ(&objects[0], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, red);    // Red wall
//        Triangle::rectangleYZ(&objects[2], 0.0f, 555.0f, 0.0f, 555.0f, 0.0f, green);    // Green wall
//        Triangle::rectangleXZ(&objects[4], 0.0f, 555.0f, 178.0f, 378.0f, 554.0f, light);// Light
//        Triangle::rectangleXZ(&objects[6], 0.0f, 555.0f, 0.0f, 555.0f, 0.0f, white);    // Floor
//        Triangle::rectangleXZ(&objects[8], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white);  // Ceiling
//        Triangle::rectangleXY(&objects[10], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white); // Back wall
//
//        // Add spheres
//        float3 center = make_float3(278.0f, 0.0f, 278.0f);
//        float radius = 100.0f;
//        objects[12] = new Sphere(make_float3(75.0f, 2 * 100.0f, 100.0f) + center, 2 * radius, dark);
//        objects[13] = new Sphere(make_float3(-175.0f, 100.0f, -100.0f) + center, radius, rodrigo);
//
//        switch (accelerationStructure) {
//
//            case (PrimitiveArray::AccelerationStructure::LINEAR):
//                // Build a brute force ray-intersection search
//                *env = new SceneObject(objects, nObjects);
//                break;
//
//            case (PrimitiveArray::AccelerationStructure::BVH):
//                // Build a bounded volume heirachy acceleration structure
//
//                const int seed = 1234;
//                const int sequence = 0;
//                const int offset = 0;
//                curandState randState;
//                curand_init(seed, sequence, offset, &randState);
//                *env = new bvhNode(objects, 0, nObjects, 0.0f, 2.0f, &randState);
//                break;
//
//            default:
//                // Build a brute force ray-intersection search
//                *env = new SceneObject(objects, nObjects);
//                break;
//        }
//    }
//}
//
//
//__global__ void initCornellBox2(PrimitiveArray** env, PrimitiveArray** objects, const int nObjects, PrimitiveArray::AccelerationStructure accelerationStructure)
//{
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//        Material* dark = new Reflective(make_float3(0.48, 0.83, 0.53), 0.00f);
//        Material* grey = new Reflective(make_float3(0.8f, 0.8f, 0.8f), 0.0f);
//        Material* green = new Lambertian(make_float3(0.12f, 0.45f, 0.15f));
//        Material* red = new Lambertian(make_float3(0.65f, 0.05f, 0.05f));
//        Material* white = new Lambertian(make_float3(1.0f, 1.0f, 1.0f));
//        Material* rodrigo = new Reflective(make_float3(124.0f, 125.0f, 193.0f) / 255.0f, 0.4f);
//        Material* light = new Light(make_float3(0.8f, 0.8f, 0.8f));
//
//        // Build cornell box
//        Triangle::rectangleYZ(&objects[0], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, red);    // Red wall
//        Triangle::rectangleYZ(&objects[2], 0.0f, 555.0f, 0.0f, 555.0f, 0.0f, green);    // Green wall
//        Triangle::rectangleXZ(&objects[4], 0.0f, 555.0f, 178.0f, 378.0f, 554.0f, light);// Light
//        Triangle::rectangleXZ(&objects[6], 0.0f, 555.0f, 0.0f, 555.0f, 0.0f, white);    // Floor
//        Triangle::rectangleXZ(&objects[8], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white);  // Ceiling
//        Triangle::rectangleXY(&objects[10], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white); // Back wall
//
//        // Add cube
//        Triangle::box(&objects[12],
//            make_float3(75.0f, 0.0f, 50.0f) + make_float3(265, 0, 295),
//            make_float2(200, 430), grey);
//
//        objects[24] = new Sphere(
//            make_float3(-25.0f, 75.0f, -200.0f) + make_float3(278.0f, 0.0f, 278.0f),
//            75.0f,
//            rodrigo);
//        objects[25] = new Sphere(
//            make_float3(-175.0f, 100.0f, -100.0f) + make_float3(278.0f, 0.0f, 278.0f),
//            100.0f, dark);
//
//        switch (accelerationStructure) {
//
//            case (PrimitiveArray::AccelerationStructure::LINEAR):
//                // Build a brute force ray-intersection search
//                *env = new SceneObject(objects, nObjects);
//                break;
//
//            case (PrimitiveArray::AccelerationStructure::BVH):
//                // Build a bounded volume heirachy acceleration structure
//
//                const int seed = 1234;
//                const int sequence = 0;
//                const int offset = 0;
//                curandState randState;
//                curand_init(seed, sequence, offset, &randState);
//                *env = new bvhNode(objects, 0, nObjects, 0.0f, 2.0f, &randState);
//                break;
//
//            default:
//                // Build a brute force ray-intersection search
//                *env = new SceneObject(objects, nObjects);
//                break;
//        }
//
//
//        delete dark;
//        delete grey;
//        delete green;
//        delete red;
//        delete white;
//        delete rodrigo;
//        delete light;
//    }
//}
//
//__global__ void initBackRoom(PrimitiveArray** env, PrimitiveArray** objects, const int nObjects, PrimitiveArray::AccelerationStructure accelerationStructure)
//{
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//        Material* ugly = new Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
//        Material* white = new Lambertian(make_float3(1.0f, 1.0f, 1.0f));
//        Material* rodrigo = new Reflective(make_float3(124.0f, 125.0f, 193.0f) / 255.0f, 0.4f);
//        Material* light = new Light(make_float3(0.85f, 0.85f, 0.85f));
//
//        // Light down hallway
//        Triangle::rectangleYZ(&objects[0], 0.0f, 555.0f, -5000.0f, 10000.0f, 0.0f, rodrigo);
//        Triangle::rectangleYZ(&objects[2], 0.0f, 555.0f, -5000.0f, 500.0f, 555.0f, rodrigo);
//        Triangle::rectangleYZ(&objects[4], 0.0f, 555.0f, 800.0f, 10000.0f, 555.0f, rodrigo);
//        Triangle::rectangleXY(&objects[6], 555.0f, 5000.0f, 0.0f, 555.0f, 800.0f, white);
//        Triangle::rectangleXY(&objects[8], 555.0f, 5000.0f, 0.0f, 555.0f, 500.0f, white);
//        Triangle::rectangleXZ(&objects[10], 555.0f, 5000.0f, 500.0f, 800.0f, 0.0f, white);
//        Triangle::rectangleXZ(&objects[12], 555.0f, 5000.0f, 500.0f, 800.0f, 555.0f, white);
//        Triangle::rectangleXZ(&objects[14], 0.0f, 555.0f, -5000.0f, 10000.0f, 0.0f, white);
//        Triangle::rectangleXZ(&objects[16], 0.0f, 555.0f, -5000.0f, 10000.0f, 555.0f, white);
//        Triangle::rectangleYZ(&objects[18], 000.0f, 555.0f, 500.0f, 800.0f, 800.0f, light);
//        //float3 center = make_float3(278.0f, 0.0f, 278.0f);
//        //float radius = 100.0f;
//        //objects[20] = new Sphere(make_float3(-75.0f, 100.0f, 1000.0f) + center, radius, ugly);
//
//        switch (accelerationStructure) {
//
//            case (PrimitiveArray::AccelerationStructure::LINEAR):
//                // Build a brute force ray-intersection search
//                *env = new SceneObject(objects, nObjects);
//                break;
//
//            case (PrimitiveArray::AccelerationStructure::BVH):
//                // Build a bounded volume heirachy acceleration structure
//
//                const int seed = 1234;
//                const int sequence = 0;
//                const int offset = 0;
//                curandState randState;
//                curand_init(seed, sequence, offset, &randState);
//                *env = new bvhNode(objects, 0, nObjects, 0.0f, 2.0f, &randState);
//                break;
//
//            default:
//                // Build a brute force ray-intersection search
//                *env = new SceneObject(objects, nObjects);
//                break;
//        }
//    }
//}
//
//__global__ void initBunny(PrimitiveArray** env, PrimitiveArray** objects, PrimitiveArray::AccelerationStructure accelerationStructure, float3* v, int3* f, int nFaces)
//{
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//        Material* dark = new Reflective(make_float3(0.1, 0.1, 0.1), 0.05f);
//        Material* black = new Lambertian(make_float3(0.5, 0.5, 0.5));
//        Material* ugly = new Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
//        Material* pink = new Lambertian(make_float3(0.7f, 0.3f, 0.3f));
//        Material* grey = new Reflective(make_float3(0.8f, 0.8f, 0.8f), 0.0f);
//
//        // Add spheres to the environment
//        objects[0] = new Triangle(
//            make_float3(-50.0f, 0.0f, -50.0f),
//            make_float3(50.0f, 0.0f, -50.0f),
//            make_float3(-50.0f, 0.0f, 50.0f), black);
//
//        objects[1] = new Triangle(
//            make_float3(50.0f, 0.0f, -50.0f),
//            make_float3(-50.0f, 0.0f, 50.0f),
//            make_float3(50.0f, 0.0f, 50.0f), ugly);
//
//        objects[2] = new Sphere(make_float3(1.5f, 1.0f, -1.0f), 1.0f, dark);
//
//        for (int i = 0; i < nFaces; i++) {
//            Triangle* triangle = new Triangle(v[f[i].x], v[f[i].y], v[f[i].z], pink);
//            objects[i + 3] = triangle;
//        }
//
//        int nObjects = nFaces + 3;
//
//        switch (accelerationStructure) {
//
//            case (PrimitiveArray::AccelerationStructure::LINEAR):
//                // Build a brute force ray-intersection search
//                *env = new SceneObject(objects, nObjects);
//                break;
//
//            case (PrimitiveArray::AccelerationStructure::BVH):
//                // Build a bounded volume heirachy acceleration structure
//
//                const int seed = 1234;
//                const int sequence = 0;
//                const int offset = 0;
//                curandState randState;
//                curand_init(seed, sequence, offset, &randState);
//                *env = new bvhNode(objects, 0, nObjects - 1, 0.0f, 2.0f, &randState);
//                break;
//
//            default:
//                // Build a brute force ray-intersection search
//                *env = new SceneObject(objects, nObjects);
//                break;
//        }
//    }
//}
//
//__global__ void initWall(PrimitiveArray** env, PrimitiveArray** objects)
//{
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//        Material* ugly = new Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
//        Material* white = new Lambertian(make_float3(1.0f, 1.0f, 1.0f));
//        Material* rodrigo = new Reflective(make_float3(124.0f, 125.0f, 193.0f) / 255.0f, 0.4f);
//        Material* light = new Light(make_float3(0.85f, 0.85f, 0.85f));
//        Material* green = new Lambertian(make_float3(0.12f, 0.45f, 0.15f));
//
//        // Build wall
//        Triangle::rectangleXY(&objects[0], -5.0f, -4.0f, -5.0f, -4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[2], -4.0f, -3.0f, -5.0f, -4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[4], -3.0f, -2.0f, -5.0f, -4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[6], -2.0f, -1.0f, -5.0f, -4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[8], -1.0f, 0.0f, -5.0f, -4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[10], 0.0f, 1.0f, -5.0f, -4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[12], 1.0f, 2.0f, -5.0f, -4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[14], 2.0f, 3.0f, -5.0f, -4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[16], 3.0f, 4.0f, -5.0f, -4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[18], 4.0f, 5.0f, -5.0f, -4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[20], -5.0f, -4.0f, -5.0f, -3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[22], -4.0f, -3.0f, -5.0f, -3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[24], -3.0f, -2.0f, -5.0f, -3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[26], -2.0f, -1.0f, -5.0f, -3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[28], -1.0f, 0.0f, -5.0f, -3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[30], 0.0f, 1.0f, -5.0f, -3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[32], 1.0f, 2.0f, -5.0f, -3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[34], 2.0f, 3.0f, -5.0f, -3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[36], 3.0f, 4.0f, -5.0f, -3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[38], 4.0f, 5.0f, -5.0f, -3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[40], -5.0f, -4.0f, -3.0f, -2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[42], -4.0f, -3.0f, -3.0f, -2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[44], -3.0f, -2.0f, -3.0f, -2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[46], -2.0f, -1.0f, -3.0f, -2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[48], -1.0f, 0.0f, -3.0f, -2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[50], 0.0f, 1.0f, -3.0f, -2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[52], 1.0f, 2.0f, -3.0f, -2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[54], 2.0f, 3.0f, -3.0f, -2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[56], 3.0f, 4.0f, -3.0f, -2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[58], 4.0f, 5.0f, -3.0f, -2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[60], -5.0f, -4.0f, -2.0f, -1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[62], -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[64], -3.0f, -2.0f, -2.0f, -1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[66], -2.0f, -1.0f, -2.0f, -1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[68], -1.0f, 0.0f, -2.0f, -1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[70], 0.0f, 1.0f, -2.0f, -1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[72], 1.0f, 2.0f, -2.0f, -1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[74], 2.0f, 3.0f, -2.0f, -1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[76], 3.0f, 4.0f, -2.0f, -1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[78], 4.0f, 5.0f, -2.0f, -1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[80], -5.0f, -4.0f, -1.0f, 0.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[82], -4.0f, -3.0f, -1.0f, 0.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[84], -3.0f, -2.0f, -1.0f, 0.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[86], -2.0f, -1.0f, -1.0f, 0.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[88], -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[90], 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[92], 1.0f, 2.0f, -1.0f, 0.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[94], 2.0f, 3.0f, -1.0f, 0.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[96], 3.0f, 4.0f, -1.0f, 0.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[98], 4.0f, 5.0f, -1.0f, 0.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[100], -5.0f, -4.0f, 0.0f, 1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[102], -4.0f, -3.0f, 0.0f, 1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[104], -3.0f, -2.0f, 0.0f, 1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[106], -2.0f, -1.0f, 0.0f, 1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[108], -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[110], 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[112], 1.0f, 2.0f, 0.0f, 1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[114], 2.0f, 3.0f, 0.0f, 1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[116], 3.0f, 4.0f, 0.0f, 1.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[118], 4.0f, 5.0f, 0.0f, 1.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[120], -5.0f, -4.0f, 1.0f, 2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[122], -4.0f, -3.0f, 1.0f, 2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[124], -3.0f, -2.0f, 1.0f, 2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[126], -2.0f, -1.0f, 1.0f, 2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[128], -1.0f, 0.0f, 1.0f, 2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[130], 0.0f, 1.0f, 1.0f, 2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[132], 1.0f, 2.0f, 1.0f, 2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[134], 2.0f, 3.0f, 1.0f, 2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[136], 3.0f, 4.0f, 1.0f, 2.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[138], 4.0f, 5.0f, 1.0f, 2.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[140], -5.0f, -4.0f, 2.0f, 3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[142], -4.0f, -3.0f, 2.0f, 3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[144], -3.0f, -2.0f, 2.0f, 3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[146], -2.0f, -1.0f, 2.0f, 3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[148], -1.0f, 0.0f, 2.0f, 3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[150], 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[152], 1.0f, 2.0f, 2.0f, 3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[154], 2.0f, 3.0f, 2.0f, 3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[156], 3.0f, 4.0f, 2.0f, 3.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[158], 4.0f, 5.0f, 2.0f, 3.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[160], -5.0f, -4.0f, 3.0f, 4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[162], -4.0f, -3.0f, 3.0f, 4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[164], -3.0f, -2.0f, 3.0f, 4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[166], -2.0f, -1.0f, 3.0f, 4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[168], -1.0f, 0.0f, 3.0f, 4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[170], 0.0f, 1.0f, 3.0f, 4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[172], 1.0f, 2.0f, 3.0f, 4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[174], 2.0f, 3.0f, 3.0f, 4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[176], 3.0f, 4.0f, 3.0f, 4.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[178], 4.0f, 5.0f, 3.0f, 4.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[180], -5.0f, -4.0f, 4.0f, 5.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[182], -4.0f, -3.0f, 4.0f, 5.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[184], -3.0f, -2.0f, 4.0f, 5.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[186], -2.0f, -1.0f, 4.0f, 5.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[188], -1.0f, 0.0f, 4.0f, 5.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[190], 0.0f, 1.0f, 4.0f, 5.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[192], 1.0f, 2.0f, 4.0f, 5.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[194], 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, green);
//        Triangle::rectangleXY(&objects[196], 3.0f, 4.0f, 4.0f, 5.0f, 0.0f, rodrigo);
//        Triangle::rectangleXY(&objects[198], 4.0f, 5.0f, 4.0f, 5.0f, 0.0f, green);
//
//        // Create the BVH
//        const int seed = 1234;
//        const int sequence = 0;
//        const int offset = 0;
//        curandState randState;
//        curand_init(seed, sequence, offset, &randState);
//
//        int totalObjects = 200;
//        //*env = new SceneObject(objects, totalObjects);
//        *env = new bvhNode(objects, 0, totalObjects, 0.0f, 1.0f, &randState);
//    }
//}