#include "raytracing_kernals.h"
#include "materials\Material.cuh"


__forceinline __device__ float3 rayEnvironmentIntersection(Trace::Ray& ray, Trace::PrimitiveArray** environment, curandState* rand) {

    int nBounces = 5;
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);

    for (int k = 0; k < nBounces; k++) {

        // Ray environment intersection test
        Trace::Record hit;
        if (!((*environment)->hit(ray, hit))) {

            float3 background = make_float3(0.5f, 0.7f, 1.0f);
            return attenuation * background;
        }

        // Attenuate light
        if (hit.material->emitted(ray, hit)) { return attenuation * ray.albedo; }
        hit.material->scatter(ray, hit, rand);
        attenuation *= ray.albedo;
    }

    return make_float3(0.0f, 0.0f, 0.0f);
}



__global__ void Trace::execute_k(
    float3* pixelBuffer,
    int width,
    int height,
    int nSamples,
    int SEED_CONSTANT,
    Trace::Camera camera,
    Trace::PrimitiveArray** environment,
    curandState* rand) {

    // Get thread information
    const int px = threadIdx.x + blockIdx.x * blockDim.x;
    const int py = threadIdx.y + blockIdx.y * blockDim.y;
    const int pIdx = py * width + px;


    if (px < width && py < height) {

        // Initialize random number generator
        //curand_init(pIdx + px + SEED_CONSTANT, px + py * width, blockIdx.x * blockDim.x, &rand[pIdx]);
        curandState randN = rand[pIdx];

        // Pixel color
        float3 pixelColor = make_float3(0.0f);

        // Sample pixel color
        for (int j = 0; j < nSamples; j++) {

            Trace::Ray ray(camera.getPosition(), camera.getPixelPosition(
                (float(px) + (curand_uniform(&randN) - 0.5f)) / float(width),
                (float(py) + (curand_uniform(&randN) - 0.5f)) / float(height)));

            pixelColor += rayEnvironmentIntersection(ray, environment, &randN);

            // Sync threads to limit the number of active warps
            __syncwarp();
            //__syncthreads();
        }
        pixelColor /= (float) nSamples;

        pixelBuffer[pIdx].x = sqrtf(pixelColor.x);
        pixelBuffer[pIdx].y = sqrtf(pixelColor.y);
        pixelBuffer[pIdx].z = sqrtf(pixelColor.z);
    }
}

extern "C" void Trace::execute(const Trace::Pipeline & pipeline, float3 * pixelBuffer) {

    int txy = 16;
    dim3 B = dim3(pipeline.imageWidth / txy + 1, pipeline.imageHeight / txy + 1);
    dim3 T = dim3(txy, txy, 1);



    execute_k << <B, T >> > (
        pixelBuffer,
        pipeline.imageWidth,
        pipeline.imageHeight,
        pipeline.nSamples,
        1,
        pipeline.camera,
        pipeline.d_environment,
        pipeline.d_rand);

}

__global__ void Trace::initRandomState_k(curandState* rand, int width, int height, int SEED_CONSTANT) {

    const int px = threadIdx.x + blockIdx.x * blockDim.x;
    const int py = threadIdx.y + blockIdx.y * blockDim.y;
    const int pIdx = py * width + px;

    if ((px < width) && (py < height)) {

        // Initialize random number generator
        curand_init(pIdx + px + SEED_CONSTANT, px + py * width, blockIdx.x * blockDim.x, &rand[pIdx]);
    }
}

extern "C" void Trace::initRandomState(Trace::Pipeline & pipeline, int SEED_CONSTANT) {

    int txy = 16;
    dim3 B = dim3(pipeline.imageWidth / txy + 1, pipeline.imageHeight / txy + 1);
    dim3 T = dim3(txy, txy, 1);


    cudaMalloc((void**) &pipeline.d_rand, pipeline.imageWidth * pipeline.imageHeight * sizeof(curandState));
    initRandomState_k << <B, T >> > (pipeline.d_rand, pipeline.imageWidth, pipeline.imageHeight, SEED_CONSTANT);

}
