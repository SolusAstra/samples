#include "Application.h"
#include "raytracing_kernals.h"
#include "SceneConstruction_kernals.h"



int Application::run() {

    // Inputs
    int imageWidth = config.width;
    int imageHeight = config.height;
    int nSamples = 1;
    int seedConstant = 1;


    // Create instance of a ray tracing pipeline
    Trace::Pipeline rayTracingPipeline(config.width, config.height, 1);
    rayTracingPipeline.nSamples = nSamples;



    // Device side random number generation
    Trace::initRandomState(rayTracingPipeline, 1);
    cudaDeviceSynchronize();

    // Scene construction
    SceneType sceneType;
    //sceneType = SceneType::FLAT_WORLD;
    //sceneType               = SceneType::FINAL_SCENE;
    sceneType = SceneType::CORNELL_BOX;
    //sceneType               = SceneType::CORNELL_BOX2;
    sceneType = SceneType::BUNNY;
    //sceneType               = SceneType::BACKROOM;
    setUpPrimitiveArray(sceneType, rayTracingPipeline);
    cudaDeviceSynchronize();

    //CameraController cameraController = CameraController(rayTracingPipeline.camera, window.getWindow());

    // Instantiate layer for rendering ray traced scene
    Layer3 baseLayer(config.width, config.height);
    baseLayer.map();


    // 1 sec = 1 hour
    double playbackSpeed = 300.0;
    double simulationTime = 0.0; // Starting at zero

    bool firstFrame = true;

    // Start the main loop
    double lastFrameTime = glfwGetTime(); // Track the time at the start of the loop

    // Main loop
    while (window.isOpen()) {



        double currentFrameTime = glfwGetTime(); // Get the current time
        float deltaTime = currentFrameTime - lastFrameTime; // Compute the delta time
        lastFrameTime = currentFrameTime; // Update the last frame time

        // Process user interaction with window
        glfwPollEvents();

        //cameraController.update(deltaTime);
        cudaDeviceSynchronize();

        // Execute ray tracing pipeline
        kernelTimer.start();
        Trace::execute(rayTracingPipeline, baseLayer.getPixelBuffer());
        cudaDeviceSynchronize();
        kernelTimer.stop();



        // Render image to window
        update(baseLayer, 1);
    }

    // Free the pixel buffer
    baseLayer.unmap();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
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