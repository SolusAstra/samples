#include "pch.h"
#include "raytracing_kernals.h"
#include "materials\Material.cuh"


__forceinline __device__ float3 fetchSphereTextureColor(cudaTextureObject_t tex, float3 direction) {

    float phi, theta;
    unitVectorToSpherical(normalize(direction), phi, theta);
    float4 mapped_color = tex2D<float4>(tex, 1 - phi, 1 - theta);

    return make_float3(mapped_color.x, mapped_color.y, mapped_color.z);
}

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

            Trace::Ray ray(camera.position, camera.getPixelPosition(
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

extern "C" void Trace::execute(const Trace::Pipeline& pipeline, float3* pixelBuffer) {

    int txy = 16;
    dim3 B = dim3(pipeline.imageWidth / txy + 1, pipeline.imageHeight / txy + 1);
    dim3 T = dim3(txy, txy, 1);



    execute_k<<<B,T>>>(
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

extern "C" void Trace::initRandomState(Trace::Pipeline& pipeline, int SEED_CONSTANT) {

    int txy = 16;
    dim3 B = dim3(pipeline.imageWidth / txy + 1, pipeline.imageHeight / txy + 1);
    dim3 T = dim3(txy, txy, 1);


    cudaMalloc((void**) &pipeline.d_rand, pipeline.imageWidth * pipeline.imageHeight * sizeof(curandState));
    initRandomState_k<<<B,T>>>(pipeline.d_rand, pipeline.imageWidth, pipeline.imageHeight, SEED_CONSTANT);

}

//// Render each pixel
//__global__ void adaptiveRender(float3* dst, int width, int height,
//    Trace::Camera camera, Trace::PrimitiveArray** environment,
//    curandState* rand, int SEED_CONSTANT, int* complexityMap, float adaptiveFactor, float3* coarseColors) {
//
//    // Samples
//    const int nSamplesMin = 10;
//    const int nSamplesMax = 500;
//
//    // Pixel position in measurement space
//    const int px = threadIdx.x + blockIdx.x * blockDim.x;
//    const int py = threadIdx.y + blockIdx.y * blockDim.y;
//    const int pIdx = py * width + px;
//
//
//    if ((px < width) && (py < height)) {
//
//        // Initialize random number generator
//        curand_init(
//            pIdx + px + SEED_CONSTANT,
//            px + py * width, blockIdx.x * blockDim.x,
//            &rand[pIdx]);
//        curandState randN = rand[pIdx];
//
//        // Compute pixel colors for block of pixels
//        float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
//        float3 meanColor = make_float3(0.0f, 0.0f, 0.0f);
//        float3 squaredMeanColor = make_float3(0.0f, 0.0f, 0.0f);
//
//
//
//        // Coarse sampling and complexity map computation
//        int nCoarseSamples = nSamplesMin;
//        for (int j = 0; j < nCoarseSamples; j++) {
//            // Create new ray from camera position through pixel
//            Trace::Ray ray = Trace::Ray(camera.position, camera.getPixelPosition(
//                (float(px) + (curand_uniform(&randN) - 0.5f)) / float(width),
//                (float(py) + (curand_uniform(&randN) - 0.5f)) / float(height)));
//
//            // Accumulate color from perPixel function
//            float3 sampleColor = perPixel(ray, environment, &randN);
//
//            pixelColor += sampleColor;
//            meanColor += sampleColor / float(nCoarseSamples);
//            squaredMeanColor += (sampleColor * sampleColor) / float(nCoarseSamples);
//
//            // Sync threads to limit the number of active warps
//            __syncwarp();
//        }
//        pixelColor /= float(nCoarseSamples);
//
//        // Store the coarse sampled color in the coarseColors buffer
//        coarseColors[pIdx] = pixelColor;
//
//        // Ensure all threads in the block have finished coarse sampling before calculating neighboring pixel colors
//        __syncthreads();
//
//        // Average color of neighboring pixels
//        float3 neighborColorSum = make_float3(0.0f, 0.0f, 0.0f);
//        float3 neighborSquaredColorSum = make_float3(0.0f, 0.0f, 0.0f);
//        int numNeighbors = 0;
//        for (int nx = max(0, px - 1); nx <= min(width - 1, px + 1); nx++) {
//            for (int ny = max(0, py - 1); ny <= min(height - 1, py + 1); ny++) {
//                if (nx != px || ny != py) {
//                    int nIdx = ny * width + nx;
//
//                    // Accumulate neighbor colors and squared colors
//                    float3 neighborColor = coarseColors[nIdx];
//                    neighborColorSum += neighborColor;
//                    neighborSquaredColorSum += neighborColor * neighborColor;
//
//                    numNeighbors++;
//                }
//            }
//        }
//        // Compute average neighbor color and squared color
//        float3 avgNeighborColor = neighborColorSum / float(numNeighbors);
//        float3 avgNeighborSquaredColor = neighborSquaredColorSum / float(numNeighbors);
//
//        // Compute variance between current pixel and neighboring pixels
//        float3 neighborVariance = avgNeighborSquaredColor - (avgNeighborColor * avgNeighborColor);
//        int complexity = int((neighborVariance.x + neighborVariance.y + neighborVariance.z) * adaptiveFactor);
//
//        // Adaptive sampling based on computed complexity
//        int nAdaptiveSamples = nSamplesMax;
//        int totalSamples = nCoarseSamples + complexity * nAdaptiveSamples;
//        for (int j = nCoarseSamples; j < totalSamples; j++) {
//            // Create new ray from camera position through pixel
//            Trace::Ray ray = Trace::Ray(camera.position, camera.getPixelPosition(
//                (float(px) + (curand_uniform(&randN) - 0.5f)) / float(width),
//                (float(py) + (curand_uniform(&randN) - 0.5f)) / float(height)));
//
//            // Accumulate color from perPixel function
//            pixelColor += perPixel(ray, environment, &randN);
//            __syncwarp();
//        }
//        pixelColor /= float(totalSamples);
//
//        // Calculate blending factor based on the number of samples
//        float blendingFactor = float(totalSamples - nSamplesMin) / float(nSamplesMax - nSamplesMin);
//
//        // Only modify the red channel if the total samples are greater than the minimum samples
//        if (totalSamples > (nSamplesMin + 300)) {
//            float redChannel = pixelColor.x * (1.0f - blendingFactor) + 1.0f * blendingFactor;
//
//            // Clamp redChannel to the range [0, 1] to ensure it doesn't exceed the valid color range
//            redChannel = redChannel > 1.0f ? 1.0f : redChannel;
//
//            // Update the red channel of the pixel color
//            pixelColor.x = redChannel;
//        }
//
//        // Set pixel color data in image buffer
//        dst[pIdx].x = sqrtf(pixelColor.x);
//        dst[pIdx].y = sqrtf(pixelColor.y);
//        dst[pIdx].z = sqrtf(pixelColor.z);
//    }
//}

