#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Graphics header files
#include "Layer.h"
#include "Texture.h"
#include "Window.h"
#include "Renderer.h"
#include "PixelBuffer.h"
#include "Shader.h"
#include "Timer.h"

struct ApplicationConfig {
    int width;
    int height;
    float aspectRatio;
};

class Application {

public:

    ApplicationConfig config;
    Timer timer;
    Timer kernelTimer;

    OpenGL::Window window;
    LayerStack layerStack;
    
public:

    Application(ApplicationConfig app_config) : config(app_config),
        window(config.width, config.height)
    {
        timer.tick();
    }
    ~Application() = default;


    int run();

    void update(Layer& layer, int nSamples) {

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

    void update() {

        timer.tick();
        timer.accumulateFrameTime();
        if (timer.getMultiframeTime() < 1.0f / 60.0f) {
            return;
        }

        // Render to screen
        layerStack.update();

        //layer.update();
        computeFPS(1);

        // Swap buffers and poll events
        window.swapBuffers();
        window.processInput();
        timer.resetMultiframeTime();
    }


    void computeFPS(int nSamples) {
        float fps = timer.computeFPS();
        float frameTime = timer.getMultiframeTime();

        if (fps > 0.0f) {
            char fpsStr[256];
            sprintf(fpsStr, "<RayTracer> %3.1f fps - nSamples %d - Render Time = %3.5f sec", fps, nSamples, frameTime);
            window.setTitle(fpsStr);
        }
    }
};