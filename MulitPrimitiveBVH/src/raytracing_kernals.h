#pragma once

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "Pipeline.h"
#include "utils/Camera.h"
#include "env/Primitive.h"


namespace Trace {


    __global__ void execute_k(
        float3* pixelBuffer, 
        int width, 
        int height, 
        int nSamples,
        int SEED_CONSTANT,
        Trace::Camera camera,
        Trace::PrimitiveArray** environment,
        curandState* rand);

    extern "C" void execute(const Trace::Pipeline& pipeline, float3* pixelBuffer);


    __global__ void initRandomState_k(curandState* rand, int width, int height, int SEED_CONSTANT);
    extern "C" void initRandomState(Trace::Pipeline& pipeline, int SEED_CONSTANT);

};