#pragma once

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>


#include "env\Primitive.h"
#include "Pipeline.h"

enum SceneType {
    FLAT_WORLD = 0,
    CORNELL_BOX = 1,
    CORNELL_BOX2 = 2,
    COMPLEX_SCENE = 3,
    BACKROOM = 4,
    BUNNY = 5,
    SPHERES = 6,
    FINAL_SCENE = 7
};

__global__ void initScene(Trace::Environment** env, Trace::Primitive** objects);
__global__ void setUpPrimitiveArray_k(Trace::Environment** d_environment, Trace::Primitive** objects);

extern "C" void setUpPrimitiveArray(SceneType sceneType, Trace::Pipeline & pipeline);