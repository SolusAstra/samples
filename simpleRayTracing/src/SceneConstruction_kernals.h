#pragma once
#include "Pipeline.h"
#include "env\Primitive.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

enum SceneType {
    RANDOM = 0,
    BUNNY = 1,
    EXEMPLAR = 2,
    CORNELL = 3
};

__global__ void buildEnvironment_k(Trace::dPrimitive* primitive, Trace::Material* material);
extern "C" void buildEnvironment(SceneType sceneType, Trace::Pipeline & pipeline);
