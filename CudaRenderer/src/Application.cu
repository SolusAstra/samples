#include "Application.h"

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "utils/Camera.h"
#include "utils/CameraController.cuh"
//#include "utils\CameraController.h"
#include "utils/Ray.h"

#include "time/Time.h"
//#include "core/time/Clock.h"
#include "state/frame/Frame.h"
#include "state/frame/ITRF.h"

#include "materials/Material.cuh"

#include "env/PrimitiveArray.cuh"
#include "env/Sphere.cuh"

#include "Image.h"

#include "core/celestial/Background.cuh"
#include "core/celestial/Earth.cuh"

__global__ void init_rand(curandState* rand, int width, int height, int SEED_CONSTANT);

__device__ float3 obliquityToEcliptic(float3 v) {
    double obliquity = 23.5 * M_PI / 180.0; // Axial tilt in radians

    // Create the axial tilt rotation matrix
    float3x3 tilt_rotation(
        make_float3(1, 0, 0),
        make_float3(0, cos(obliquity), -sin(obliquity)),
        make_float3(0, sin(obliquity), cos(obliquity)));
    //float3x3 tilt_rotation = float3x3(
    //    1, 0, 0,
    //    0, cos(obliquity), sin(obliquity),
    //    0, -sin(obliquity), cos(obliquity)
    //);

    // Rotate the vector
    return tilt_rotation * v;
}

__device__ float4 perPixelTex(cudaTextureObject_t background, Trace::Ray& ray, Trace::Environment* env, curandState* rand) {

    int nBounces = 1;
    float4 attenuation = make_float4(1.0f);

    for (int i = 0; i < nBounces; i++) {

        // Ray environment intersection test
        Trace::Record rec;
        if (!(env->hit(ray, rec, rand))) {

            float3 unitDir = obliquityToEcliptic(ray.dir);


            //return make_float4(0.0f, 1.0f, 1.0f, 0.0f);
            return make_float4(fetchSphereTextureColor(background, unitDir), 1.0f);
        }

        // Attenuate ray color
        attenuation *= make_float4(ray.albedo, 1.0f);
    }

    return attenuation;
}

// Render each pixel
__global__ void renderTex(Background background, float4* dst, int width, int height, Trace::Camera cam,
    Trace::Environment* env, curandState* rand, int nSamples) {

    // Pixel position in measurement space
    const int px = threadIdx.x + blockIdx.x * blockDim.x;
    const int py = threadIdx.y + blockIdx.y * blockDim.y;
    const int pIdx = py * width + px;

    //int nSamples = 300;

    if ((px < width) && (py < height)) {

        // Initialize random number generator
        curandState randN = rand[pIdx];

        // Compute pixel colors for block of pixels
        float4 pixelColor = make_float4(0.0f);
        for (int j = 0; j < nSamples; j++) {

            // Accumulate color from perPixel function
            Trace::Ray project_ray(cam.position, cam.getPixelPosition(
                (float(px) + (curand_uniform(&randN) - 0.5f)) / float(width),
                (float(py) + (curand_uniform(&randN) - 0.5f)) / float(height)));
            pixelColor += perPixelTex(background.tex, project_ray, env, &randN);
        }
        pixelColor /= (float)nSamples;

        // Set pixel color data in image buffer
        dst[pIdx].x = pixelColor.x;
        dst[pIdx].y = pixelColor.y;
        dst[pIdx].z = pixelColor.z;
        dst[pIdx].w = pixelColor.w;
    }
}


__global__ void initData(Trace::Environment* d_environment, Earth earth,
    Trace::SphereSoA* d_spheres, float3* d_centers, float* d_radii, Trace::Material** d_mats) {

    // Set sphere properties pointers
    earth.initProperties(d_spheres, d_centers, d_radii);
    earth.initEarthMaterial(d_mats);

    // Set in environment
    d_environment->sphereSoA = d_spheres;
    d_environment->globalMaterialPool = d_mats;
}

//__device__ void updateTransform(Trace::Material** d_mats, Frame<ITRF>& itrf, double simtime) {
//    //Transform t = itrf.getTransform(simtime);
//    d_mats[0]->updateTransform(itrf.transform);
//}

__global__ void simTick(Trace::Environment* d_environment, Transform t) {
    d_environment->globalMaterialPool[0]->updateTransform(t);
}

int Application::run() {

    int nSamples = 100;


    // cuda launch parameters
    int txy = 16;
    dim3 B = dim3(config.width / txy + 1, config.height / txy + 1);
    dim3 T = dim3(txy, txy, 1);

    // Initialize random number generator
    curandState* d_rand;
    cudaMalloc((void**)&d_rand, config.width * config.height * sizeof(curandState));
    init_rand << <B, T >> > (d_rand, config.width, config.height, 1);

    // Galactic Background
    Background background("res/starmap_2020_4k_gal.exr");

    // Build environment
    Trace::Environment* d_environment = Trace::Environment::allocate();
    Earth earth;
    initData << <1, 1 >> > (d_environment, earth, earth.d_sphere, earth.d_position, earth.d_radius, earth.d_material);

    // Create camera
    float cam_dist = 42164000.0f;
    float3 origin = cam_dist * normalize(make_float3(-7.0f, -6.0f, 0.1f));
    float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);
    float vFOV = 60.0f;
    float aspectRatio = (float)config.width / (float)config.height;
    Trace::Camera _camera(origin, lookAt, vFOV, aspectRatio);
    CameraController cameraController = CameraController(_camera, window.getWindow());

    // Create clock
    DateTime startTime{ 2000, 1, 1, 0, 0, 0 };
    //Time startTime{ 2000, 1, 1, 0, 0, 0 };
    double stepSize = 10 * 60; // 10 minutes
    double duration = 24; // 24 hours
    //Clock clock = Clock(startTime, stepSize, duration);

    Frame<ITRF> itrf{startTime};

    //ITRF irtf{};


    // Create layers
    Layer4 baseLayer(config.width, config.height);
    baseLayer.map();

    cudaDeviceSynchronize();

    // 1 sec = 1 hour
    double playbackSpeed = 300.0;
    double simulationTime = 0.0; // Starting at zero

    bool firstFrame = true;

    // Start the main loop
    double lastFrameTime = glfwGetTime(); // Track the time at the start of the loop
    while (window.isOpen()) {

        double currentFrameTime = glfwGetTime(); // Get the current time
        float deltaTime = currentFrameTime - lastFrameTime; // Compute the delta time
        lastFrameTime = currentFrameTime; // Update the last frame time

        // Update the simulation time
        //simulationTime += deltaTime * playbackSpeed;
        Transform transformUpdate = itrf.getTransform(deltaTime * playbackSpeed);

        std::cout<< transformUpdate.rotation.u.x << transformUpdate.rotation.u.y << transformUpdate.rotation.u.z << std::endl;

        simTick << <1, 1 >> > (d_environment, transformUpdate);
        cudaDeviceSynchronize();

        // Process events
        glfwPollEvents();

        // Update the camera
        cameraController.update(_camera.position, lookAt, vFOV, aspectRatio, deltaTime);
        cudaDeviceSynchronize();

        // Call the kernel
        kernelTimer.start();
        renderTex << <B, T >> > (background, baseLayer.getPixelBuffer(), config.width, config.height, _camera, d_environment, d_rand, nSamples);
        cudaDeviceSynchronize();
        kernelTimer.stop();

        update(baseLayer, 1);
    }

    // Free the screen on the device
    baseLayer.unmap();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}

int main(void) {
    float aspectRatio = 16.0f / 9.0f;

    // 3840
    // 1440

    ApplicationConfig config;
    config.aspectRatio = 16.0f / 9.0f;
    config.width = 1440;
    config.height = (float)config.width / config.aspectRatio;


    Application app(config);
    app.run();
    return 0;
}

__global__ void init_rand(curandState* rand, int width, int height, int SEED_CONSTANT) {

    // Pixel position in measurement space
    const int px = threadIdx.x + blockIdx.x * blockDim.x;
    const int py = threadIdx.y + blockIdx.y * blockDim.y;
    const int pIdx = py * width + px;

    if ((px < width) && (py < height)) {

        // Initialize random number generator
        curand_init(pIdx + px + SEED_CONSTANT, px + py * width, blockIdx.x * blockDim.x, &rand[pIdx]);
    }
}


//int Application::run() {
//
//    // cuda launch parameters
//    int txy = 16;
//    dim3 B = dim3(config.width / txy + 1, config.height / txy + 1);
//    dim3 T = dim3(txy, txy, 1);
//
//    int nSamples = 100;
//
//    // Initialize random number generator
//    curandState* d_rand;
//    cudaMalloc((void**)&d_rand, config.width * config.height * sizeof(curandState));
//    init_rand << <B, T >> > (d_rand, config.width, config.height, 1);
//
//
//
//    // Galactic Background
//    Img<float4> sky_img = Img<float4>::loadEXRImage("res/starmap_2020_4k_gal.exr");
//    sky_img.generateCudaTexture();
//    //Layer galLayer(config.width, config.height, 3);
//
//    //galLayer.map();
//    //layerStack.addLayer(std::move(galLayer));
//
//
//
//    // Build environment
//    Trace::Environment* d_environment = Trace::Environment::allocate();
//    Earth earth;
//    initData << <1, 1 >> > (d_environment, earth, earth.d_sphere, earth.d_position, earth.d_radius, earth.d_material);
//
//
//
//    // Create camera
//    float cam_dist = 42164000.0f;
//    float3 origin = cam_dist * normalize(make_float3(-7.0f, -6.0f, 0.1f));
//    float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);
//    float vFOV = 60.0f;
//    float aspectRatio = (float)config.width / (float)config.height;
//    Trace::Camera _camera(origin, lookAt, vFOV, aspectRatio);
//    CameraController cameraController = CameraController(_camera, window.getWindow());
//
//    // Create clock
//    Time startTime{ 2000, 1, 1, 0, 0, 0 };
//    double stepSize = 10 * 60; // 10 minutes
//    double duration = 24; // 24 hours
//    Clock clock = Clock(startTime, stepSize, duration);
//
//    ReferenceFrame irtf = ReferenceFrame::ITRF(clock);
//
//
//    // Create layers
//    Layer baseLayer(config.width, config.height, 3);
//
//
//    baseLayer.map();
//
//    //layerStack.addLayer(std::move(baseLayer));
//
//    cudaDeviceSynchronize();
//
//    // 1 sec = 1 hour
//    double playbackSpeed = 300.0;
//    double simulationTime = 0.0; // Starting at zero
//
//    bool firstFrame = true;
//
//    // Start the main loop
//    double lastFrameTime = glfwGetTime(); // Track the time at the start of the loop
//    while (window.isOpen()) {
//
//        double currentFrameTime = glfwGetTime(); // Get the current time
//        float deltaTime = currentFrameTime - lastFrameTime; // Compute the delta time
//        lastFrameTime = currentFrameTime; // Update the last frame time
//
//        // Update the simulation time
//        simulationTime += deltaTime * playbackSpeed;
//
//        Transform t_update = irtf.getTransform(simulationTime);
//
//        //std::cout << "Simulation Time: " << simulationTime << std::endl;
//        //std::cout << "Transform: " << std::endl; 
//        //std::cout << t_update.rotation.u.x << ", " << t_update.rotation.v.x << ", " << t_update.rotation.w.x << std::endl;
//        //std::cout << t_update.rotation.u.y << ", " << t_update.rotation.v.y << ", " << t_update.rotation.w.y << std::endl;
//        //std::cout << t_update.rotation.u.z << ", " << t_update.rotation.v.z << ", " << t_update.rotation.w.z << std::endl;
//
//        simTick << <1, 1 >> > (d_environment, t_update);
//        cudaDeviceSynchronize();
//
//        // Process events
//        glfwPollEvents();
//
//        // Update the camera
//        cameraController.update(_camera.position, lookAt, vFOV, aspectRatio, deltaTime);
//        cudaDeviceSynchronize();
//
//        // Call the kernel
//        kernelTimer.start();
//        renderTex << <B, T >> > (sky_img.tex, baseLayer.pixelBuffer.getMappedData(), config.width, config.height, _camera, d_environment, d_rand, nSamples);
//        cudaDeviceSynchronize();
//        kernelTimer.stop();
//
//        // Update the screen
//        //update();
//        update(baseLayer, 1);
//
//        //if (cameraController.hasPanned || cameraController.hasZoomed || firstFrame) {
//        //    firstFrame = false;
//
//        //    // Call the kernel
//        //    kernelTimer.start();
//        //    renderTex<<<B,T>>>(sky_img.tex, baseLayer.pixelBuffer.getMappedData(), config.width, config.height, _camera, d_environment, d_rand, nSamples);
//        //    cudaDeviceSynchronize();
//        //    kernelTimer.stop();
//
//        //    cameraController.hasPanned = false; // Reset the flag
//        //    cameraController.hasZoomed = false; // Reset the flag
//
//        //    //// Update the screen
//        //    update(baseLayer, 1);
//
//        //}
//
//    }
//
//    cudaFree(d_environment);
//
//    // Free the screen on the device
//    //galLayer.unmap();
//    //layerStack.unmap();
//    baseLayer.unmap();
//
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
//    }
//
//    return 0;
//}