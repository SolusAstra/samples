#include "SceneConstruction_kernals.h"

#include "materials\Material.cuh"
#include "env\Environment.h"
#include "env\Triangle.h"
#include "premade.h"
//#include "env\Sphere.cuh"

#include "utils/util.cuh"
#include "tiny_obj_loader.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

__device__ float3 randF3(curandState* rand) { return make_float3(curand_uniform(rand), curand_uniform(rand), curand_uniform(rand)); }
__device__ float3 randF3(curandState* rand, float min, float max) { return (max - min) * randF3(rand) + min; }
__device__ float randFloat(curandState* rand, float min, float max) { return (max - min) * curand_uniform(rand) + min; }

__global__ void buildEnvironment_k(Trace::dPrimitive* primitive, Trace::Material* material) {


}

//__global__ void initCornellBox(Trace::PrimitiveArray** env, Trace::Primitive** objects)
//{
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//        Trace::Material* dark = new Trace::Reflective(make_float3(0.1, 0.1, 0.1), 0.05f);
//        Trace::Material* black = new Trace::Lambertian(make_float3(0.5, 0.5, 0.5));
//        Trace::Material* ugly = new Trace::Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
//        Trace::Material* pink = new Trace::Lambertian(make_float3(0.7f, 0.3f, 0.3f));
//        Trace::Material* grey = new Trace::Reflective(make_float3(0.8f, 0.8f, 0.8f), 0.0f);
//        Trace::Material* green = new Trace::Lambertian(make_float3(0.12f, 0.45f, 0.15f));
//        Trace::Material* red = new Trace::Lambertian(make_float3(0.65f, 0.05f, 0.05f));
//        Trace::Material* white = new Trace::Lambertian(make_float3(1.0f, 1.0f, 1.0f));
//        Trace::Material* rodrigo = new Trace::Lambertian(make_float3(124.0f, 125.0f, 193.0f) / 255.0f);
//
//        // Soft light
//        Trace::Material* light = new Trace::Light(make_float3(0.8f, 0.8f, 0.8f));
//
//        // Build cornell box
//        Trace::Rectangle::YZ(&objects[0], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, red);    // Red wall
//        Trace::Rectangle::YZ(&objects[2], 0.0f, 555.0f, 0.0f, 555.0f, 0.0f, green);    // Green wall
//        Trace::Rectangle::XZ(&objects[4], 0.0f, 555.0f, 178.0f, 378.0f, 554.0f, light);// Light
//        Trace::Rectangle::XZ(&objects[6], 0.0f, 555.0f, 0.0f, 555.0f, 0.0f, white);    // Floor
//        Trace::Rectangle::XZ(&objects[8], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white);  // Ceiling
//        Trace::Rectangle::XY(&objects[10], 0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white); // Back wall
//
//        *env = new Trace::PrimitiveArray(objects, 12);
//    } 
//
//}

//__global__ void initBunny(Trace::PrimitiveArray** env, Trace::Primitive** objects, float3* v, int3* f, int nFaces)
//{
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//        Trace::Material* dark = new Trace::Reflective(make_float3(0.1, 0.1, 0.1), 0.05f);
//        Trace::Material* black = new Trace::Lambertian(make_float3(0.5, 0.5, 0.5));
//        Trace::Material* ugly = new Trace::Reflective(make_float3(0.3f, 0.8f, 0.4f), 0.0f);
//        Trace::Material* pink = new Trace::Lambertian(make_float3(0.7f, 0.3f, 0.3f));
//        Trace::Material* grey = new Trace::Reflective(make_float3(0.8f, 0.8f, 0.8f), 0.0f);
//
//        Trace::Material* ground = new Trace::Lambertian(make_float3(0.5f, 0.5f, 0.5f));
//
//        int nObjects = nFaces + 2;
//        float3 vertices = new float3[nObjects];
//
//
//
//
//
//
//        // Add spheres to the environment
//        objects[0] = Trace::Triangle(
//            make_float3(-50.0f, 0.0f, -50.0f),
//            make_float3(50.0f, 0.0f, -50.0f),
//            make_float3(-50.0f, 0.0f, 50.0f), dark);
//
//        objects[1] = Trace::Triangle(
//            make_float3(50.0f, 0.0f, -50.0f),
//            make_float3(-50.0f, 0.0f, 50.0f),
//            make_float3(50.0f, 0.0f, 50.0f), ugly);
//        //Trace::Rectangle::XZ(&objects[0], -50.0f, 50.0f, -50.0f, 50.0f, 0.0f, ground);
//
//        //objects[2] = new Trace::Sphere(make_float3(1.5f, 1.0f, -1.0f), 1.0f, dark);
//
//
//        for (int i = 0; i < nFaces; i++) {
//            Trace::Primitive* triangle = Trace::Primitive::Triangle(v[f[i].x], v[f[i].y], v[f[i].z], pink);
//            objects[i + 2] = triangle;
//        }
//
//        
//
//
//        *env = new Trace::PrimitiveArray(objects, nObjects);
//
//    }
//}

float computeAspectRatio(Trace::Pipeline& pipeline) {
    return (float) pipeline.imageWidth / (float) pipeline.imageHeight;
}

// Kernel to use dPrimitive
__global__ void usePrimitives(Trace::dPrimitive* d_prim) {
    if (threadIdx.x < d_prim->N) {
        float3 vertex = d_prim->vertex[threadIdx.x];
        // Use vertex data for some computations
        printf("Vertex %d: (%f, %f, %f)\n", threadIdx.x, vertex.x, vertex.y, vertex.z);
    }
}

__global__ void initEnvironment(Trace::Environment** env, Trace::dPrimitive* primitive, Trace::Material* materials) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *env = new Trace::Environment(primitive, materials);
    }
}

extern "C" void buildEnvironment(SceneType sceneType, Trace::Pipeline& pipeline) {

    // Scene data
    float3 origin;
    float3 target;
    float vFOV;

    int numVertices;
    int numPolygons;
    std::vector<float3> vertices(0);
    std::vector<int3> indices(0);
    std::vector<Trace::Material> mats(0);
    
    Trace::Camera cam;
    Trace::Triangle* h_triangle = nullptr;
    Trace::Material* d_materials = nullptr;
    Trace::dPrimitive* d_triangle = nullptr;

    Trace::Environment** d_envPtr;
    cudaMalloc((void**) &d_envPtr, sizeof(Trace::Environment*));

    switch (sceneType) {
        case (SceneType::CORNELL):
        {
            // Camera
            origin = make_float3(278, 278, -800);
            target = make_float3(278, 278, 0);
            vFOV = 40.0f;

            initCornellBox(vertices, indices);

            Trace::Material green = Trace::Material(MATERIAL_TYPE::LAMBERTIAN, make_float3(0.12f, 0.45f, 0.15f));
            Trace::Material red = Trace::Material(MATERIAL_TYPE::LAMBERTIAN, make_float3(0.65f, 0.05f, 0.05f));
            Trace::Material white = Trace::Material(MATERIAL_TYPE::LAMBERTIAN, make_float3(1.0f, 1.0f, 1.0f));
            Trace::Material light = Trace::Material(MATERIAL_TYPE::EMITTER, make_float3(0.8f, 0.8f, 0.8f));

            mats.push_back(red);
            mats.push_back(green);
            mats.push_back(light);
            mats.push_back(white);

            std::vector<int> matID(indices.size());
            matID[0] = 0;
            matID[1] = 0;
            matID[2] = 1;
            matID[3] = 1;
            matID[4] = 2;
            matID[5] = 2;
            matID[6] = 3;
            matID[7] = 3;
            matID[8] = 3;
            matID[9] = 3;
            matID[10] = 3;
            matID[11] = 3;

            //for (int k = 12; k < indices.size(); k++) {
            //    matID[k] = 0;
            //}

            h_triangle = new Trace::Triangle(vertices, indices, matID);
            numVertices = h_triangle->vertex.size();
            numPolygons = h_triangle->face.size();

            break;
        }
        case (SceneType::BUNNY):
        {
            // Camera
            origin = make_float3(2.0f, 3.0f, -1.0f);
            target = make_float3(0.0f, 0.0f, 0.0f);
            vFOV = 20.0f;

            // Initialize scene geometry

            vertices.push_back(make_float3(-50.0f, 0.0f, -50.0f));
            vertices.push_back(make_float3(50.0f, 0.0f, -50.0f));
            vertices.push_back(make_float3(-50.0f, 0.0f, 50.0f));
            vertices.push_back(make_float3(50.0f, 0.0f, 50.0f));
            indices.push_back(make_int3(0, 1, 2));
            indices.push_back(make_int3(1, 2, 3));

            stanfordBunny(vertices, indices);

            Trace::Material dark = Trace::Material(MATERIAL_TYPE::REFLECTIVE, make_float3(0.1, 0.1, 0.1), 0.05f);
            Trace::Material ugly = Trace::Material(MATERIAL_TYPE::REFLECTIVE, make_float3(0.3f, 0.8f, 0.4f), 0.0f);
            Trace::Material rodrigo = Trace::Material(MATERIAL_TYPE::LAMBERTIAN, make_float3(124.0f, 125.0f, 193.0f) / 255.0f);

            Trace::Material pink = Trace::Material(MATERIAL_TYPE::LAMBERTIAN, make_float3(0.7f, 0.3f, 0.3f));

            mats.push_back(pink);
            mats.push_back(dark);
            mats.push_back(ugly);
            

            std::vector<int> matID(indices.size());
            matID[0] = 1;
            matID[1] = 2;
            for (int k = 2; k < indices.size(); k++) {
                matID[k] = 0;
            }

            h_triangle = new Trace::Triangle(vertices, indices, matID);
            numVertices = h_triangle->vertex.size();
            numPolygons = h_triangle->face.size();
            break;
        }


    }

    std::cout << "# Vertices: " << numVertices << std::endl;
    std::cout << "# Polygons: " << numPolygons << std::endl;

    // Device side primitives
    d_triangle = h_triangle->createDeviceVersion();

    // Device side materials
    cudaMalloc((void**) &d_materials, mats.size() * sizeof(Trace::Material));
    cudaMemcpy(d_materials, mats.data(), mats.size() * sizeof(Trace::Material), cudaMemcpyHostToDevice);

    // Set environment properties
    initEnvironment << <1, 1 >> > (d_envPtr, d_triangle, d_materials);
    cudaDeviceSynchronize();

    // Device side environment
    cudaMemcpy(&pipeline.d_environment, d_envPtr, sizeof(Trace::Environment*), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Camera
    cam = Trace::Camera(origin, target, vFOV, computeAspectRatio(pipeline));
    pipeline.camera = cam;
    

    Trace::dPrimitive::deleteDeviceData(d_triangle);
    delete h_triangle;            // Clean up the host object

    return;

}

//// Use the dPrimitive in a kernel
//usePrimitives << <1, 3 >> > (d_triangle);
//cudaDeviceSynchronize();