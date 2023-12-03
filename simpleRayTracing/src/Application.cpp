#include "Application.h"
//#include "raytracing_kernals.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "SceneConstruction_kernals.h"

#include "materials\Material.cuh"
#include "env\Environment.h"
#include "env\Triangle.h"
#include "premade.h"
#include "utils/util.cuh"

float computeAspectRatio(Trace::Pipeline& pipeline) {
    return (float) pipeline.imageWidth / (float) pipeline.imageHeight;
}

__global__ void setBuffer(float3* pixelBuffer, float3* cpuBuffer, int width, int height) {

    // Get thread information
    const int px = threadIdx.x + blockIdx.x * blockDim.x;
    const int py = threadIdx.y + blockIdx.y * blockDim.y;
    const int pIdx = py * width + px;


    if (px < width && py < height) {
        pixelBuffer[pIdx].x = cpuBuffer[pIdx].x;
        pixelBuffer[pIdx].y = cpuBuffer[pIdx].y;
        pixelBuffer[pIdx].z = cpuBuffer[pIdx].z;
    }
}

void buildCPUEnvironment(SceneType sceneType, Trace::Pipeline& pipeline) {

    // Scene data
    float3 origin = make_float3(1.0f, 2.0f, 1.0f);
    float3 target = make_float3(0.0f, 0.0f, 0.0f);
    float vFOV = 40.0f;

    int numVertices = 0;
    int numPolygons = 0;
    std::vector<float3> vertices(0);
    std::vector<int3> indices(0);
    std::vector<Trace::Material> mats(0);

    Trace::Camera cam;
    Trace::Triangle* h_triangle = nullptr;
    Trace::Material* d_materials = nullptr;
    Trace::dPrimitive* d_triangle = nullptr;

    Trace::Environment* d_envPtr = nullptr;
    //cudaMalloc((void**) &d_envPtr, sizeof(Trace::Environment*));

    switch (sceneType) {
        case (SceneType::BUNNY):
        {
            // Camera
            origin = make_float3(1.0f, 2.0f, 1.0f);
            target = make_float3(0.0f, 0.0f, 0.0f);
            vFOV = 40.0f;

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


    Trace::Primitive* primitive = new Trace::Triangle(h_triangle->vertex, h_triangle->face, h_triangle->matID);

    Trace::Material* material = mats.data();


    d_envPtr = new Trace::Environment(primitive, material);

    // Camera
    cam = Trace::Camera(origin, target, vFOV, computeAspectRatio(pipeline));
    pipeline.camera = cam;

    pipeline.d_environment = d_envPtr;

    //Trace::dPrimitive::deleteDeviceData(d_triangle);
    //delete h_triangle;            // Clean up the host object

    return;

}

__forceinline __device__ float3 rayEnvironmentIntersection(Trace::Ray& ray, Trace::Environment* env) {

    int nBounces = 5;
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);
    float3 background = make_float3(0.0f, 0.0f, 0.0f);

    for (int k = 0; k < nBounces; k++) {

        // Ray environment intersection test
        Trace::Record hit;
        if (!(env->hit(ray, hit))) {
            return attenuation * background;
        }

        // Material effect
        Trace::Material mat = env->materials[hit.matID];
        if (mat.emitted(ray, hit)) {
            return attenuation * ray.albedo;
        }
        mat.scatter(ray, hit);

        // Attenuate light
        attenuation *= ray.albedo;
    }

    return make_float3(0.0f, 0.0f, 0.0f);
}

void execute_k(
    float3* pixelBuffer,
    int width,
    int height,
    int nSamples,
    Trace::Environment* env,
    Trace::Camera camera) {

    // Iterate over all pixels
    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            int pIdx = py * width + px;

            // Sample pixel color
            float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
            for (int j = 0; j < nSamples; ++j) {
                // Pixel position with random jitter for anti-aliasing
                float u = ((float) px + randomFloat(0.0f, 1.0f) - 0.5f) / (float) width;
                float v = ((float) py + randomFloat(0.0f, 1.0f) - 0.5f) / (float) height;

                // Project ray
                Trace::Ray ray(camera.position, camera.getPixelPosition(u, v));

                pixelColor += rayEnvironmentIntersection(ray, env);

            }
            pixelColor /= (float) nSamples;

            pixelBuffer[pIdx].x = sqrtf(pixelColor.x);
            pixelBuffer[pIdx].y = sqrtf(pixelColor.y);
            pixelBuffer[pIdx].z = sqrtf(pixelColor.z);
        }
    }
}

void execute(const Trace::Pipeline & pipeline, float3* pixelBuffer) {

    execute_k(
        pixelBuffer,
        pipeline.imageWidth,
        pipeline.imageHeight,
        pipeline.nSamples,
        pipeline.d_environment,
        pipeline.camera);
}

int Application::run() {

    // Inputs
    int imageWidth = config.width;
    int imageHeight = config.height;
    int nSamples = 100;
    int seedConstant = 1;

    int txy = 16;
    dim3 B = dim3(imageWidth / txy + 1, imageHeight / txy + 1);
    dim3 T = dim3(txy, txy, 1);

    SceneType sceneType;
    sceneType = SceneType::BUNNY;
    
    // Create instance of a ray tracing pipeline
    Trace::Pipeline rayTracingPipeline(imageWidth, imageHeight, nSamples);


    buildCPUEnvironment(sceneType, rayTracingPipeline);

    float3* cpuBuffer = new float3[imageWidth * imageHeight];





    // Instantiate layer for rendering ray traced scene
    Layer3 baseLayer(config.width, config.height);
    baseLayer.map();

    // Main loop
    while (window.isOpen()) {

        // Process user interaction with window
        glfwPollEvents();

        // Execute ray tracing pipeline
        kernelTimer.start();

        execute(rayTracingPipeline, cpuBuffer);

        setBuffer<<<B,T>>>(baseLayer.getPixelBuffer(), cpuBuffer, imageWidth, imageHeight);

        //Trace::execute(rayTracingPipeline, baseLayer.getPixelBuffer());
        //cudaDeviceSynchronize();
        kernelTimer.stop();

        // Render image to window
        update(baseLayer, 1);
    }






    //// Device side random number generation
    //Trace::initRandomState(rayTracingPipeline, 1);
    //cudaDeviceSynchronize();

    // Initialize Environment
    //buildEnvironment(sceneType, rayTracingPipeline);

    //// Instantiate layer for rendering ray traced scene
    //Layer3 baseLayer(config.width, config.height);
    //baseLayer.map();

    //// Main loop
    //while (window.isOpen()) {

    //    // Process user interaction with window
    //    glfwPollEvents();

    //    // Execute ray tracing pipeline
    //    kernelTimer.start();
    //    Trace::execute(rayTracingPipeline, baseLayer.getPixelBuffer());
    //    cudaDeviceSynchronize();
    //    kernelTimer.stop();

    //    // Render image to window
    //    update(baseLayer, 1);
    //}

    // Free the pixel buffer
    baseLayer.unmap();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}

inline void Application::update(Layer& layer, int nSamples) {

    timer.tick();
    timer.accumulateFrameTime();
    if (timer.getMultiframeTime() < 1.0f / 60.0f) {
        return;
    }

    // Render to screen
    layer.update();
    computeFPS(nSamples);

    // Swap buffers and poll events
    window.swapBuffers();
    window.processInput();
    timer.resetMultiframeTime();
}

int main(void) {
    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;

    ApplicationConfig config;
    config.aspectRatio = 16.0f / 9.0f;
    config.width = 1440;
    config.height = (float) config.width / config.aspectRatio;


    Application app(config);
    app.run();
    return 0;
}