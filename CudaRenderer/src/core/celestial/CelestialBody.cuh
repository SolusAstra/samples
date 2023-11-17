#pragma once
#include <sutil/vec_math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>


#include "materials/Material.cuh"
#include "env/Sphere.cuh"

__inline__ __device__ float3 fetchSphereTextureColor(cudaTextureObject_t tex, float3 direction) {

    float phi, theta;
    unitVectorToSpherical(normalize(direction), phi, theta);
    float4 mapped_color = tex2D<float4>(tex, 1 - phi, 1 - theta);

    return make_float3(mapped_color.x, mapped_color.y, mapped_color.z);
}

class CelestialBody {

public:
    __host__ __device__ CelestialBody() {}

};
