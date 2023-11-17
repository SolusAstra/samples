#pragma once
#include "CelestialBody.cuh"

#include <sutil/vec_math.h>
#include <cuda_runtime.h>

#include "materials/Material.cuh"
#include "env/Sphere.cuh"
#include "state/frame/Frame.h"



class EarthMaterial : public Trace::Material {

public:
    float TOD = 11.0f;
    Transform earthTransform;
    cudaTextureObject_t dayTex;
    cudaTextureObject_t nightTex;

    __device__ EarthMaterial() {}

    __device__ virtual void updateTransform(const Transform& transform) override {
        earthTransform = transform;
    }

    __device__ virtual bool scatter(Trace::Ray& ray, Trace::Record& hit, curandState* rand = nullptr) override {


        ray.org = hit.point;
        ray.dir = hit.normal;

        float3 normalToSurface = hit.normal;

        // Rotate the normal to the surface into the inertial frame
        double3x3 rotMatrix = earthTransform.rotation;

        float3 rotNormal = make_float3(
            rotMatrix.u.x * normalToSurface.x + rotMatrix.u.y * normalToSurface.y + rotMatrix.u.z * normalToSurface.z,
            rotMatrix.v.x * normalToSurface.x + rotMatrix.v.y * normalToSurface.y + rotMatrix.v.z * normalToSurface.z,
            rotMatrix.w.x * normalToSurface.x + rotMatrix.w.y * normalToSurface.y + rotMatrix.w.z * normalToSurface.z);

        float sun_angle, sun_intensity;
        shade(hit.point, hit.normal, sun_angle, sun_intensity);

        // Fetch texture colors based on location of hit point
        float3 day_color = fetchSphereTextureColor(dayTex, rotNormal);
        day_color *= sun_intensity;


        float3 night_color = fetchSphereTextureColor(nightTex, rotNormal);
        night_color *= (1.0f - sun_intensity);
        gammaCorrect(night_color);

        // Linearly interpolate between the adjusted night and day colors based on sun_intensity
        ray.albedo = lerp(night_color, day_color, sun_intensity);
        //ray.albedo = day_color;

        // Lambertian scattering
        Trace::LambertianScattering(rotNormal, hit.normal, rand);
        return true;
    }

private:

    __device__ void computeSunPosition(float3& sunPosition, float TOD) {

        // Default sun position
        float3 sun_position = 151000000000 * make_float3(1.0f, 0.0f, 0.0f);

        // Angle in radians of the earth's equator relative to the ecliptic
        float ERA_rel_ecliptic = 23.4f * M_PI / 180.0f;

        // TODO: swap TOD should be handled by a reference frame transform based on the earth's rotation and the time of day
        // For now we just rotate the sun position about the inertial y-axis 
        float theta = 2.0f * M_PI * (TOD / 24.0f); // Convert to radians

        // Rotate the sun position about the y-axis (world's up axis) by the time of day
        float3 rotated_position;
        rotated_position.x = sun_position.x * cosf(theta) - sun_position.z * sinf(theta);
        rotated_position.y = sun_position.y;
        rotated_position.z = sun_position.x * sinf(theta) + sun_position.z * cosf(theta);

        // Tilt the rotated position by the earth's equatorial tilt
        sunPosition.x = rotated_position.x;
        sunPosition.y = rotated_position.y * cosf(ERA_rel_ecliptic) - rotated_position.z * sinf(ERA_rel_ecliptic);
        sunPosition.z = rotated_position.y * sinf(ERA_rel_ecliptic) + rotated_position.z * cosf(ERA_rel_ecliptic);
    }

    __device__ void shade(const float3& pos, const float3& normal, float& sunAngle, float& sunIntensity) {

        float3 sunPosition;
        computeSunPosition(sunPosition, TOD);

        float3 sun_dir = normalize(sunPosition - pos);

        sunAngle = acosf(dot(sun_dir, normal));
        sunIntensity = fmaxf(0.0f, dot(normal, sun_dir));
        //sunIntensity = cosf(sunAngle);
    }

    // Gamma correct night color
    __device__ void gammaCorrect(float3& color) {
        float gamma = 5.0f;
        color.x = powf(color.x, gamma);
        color.y = powf(color.y, gamma);
        color.z = powf(color.z, gamma);
    }

};


class Earth : public CelestialBody {


public:
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 velocity = make_float3(0.0f, 0.0f, 0.0f);
    float radius = 0.0f;

    Trace::Material** d_material; // Array of pointers to materials.
    Trace::SphereSoA* d_sphere;
    float3* d_position;
    float* d_radius;

    cudaTextureObject_t dayTex;
    cudaTextureObject_t nightTex;

    //Earth() {}

    __host__ Earth() {
        // Allocate memory for sphere, position, radius, and material
        cudaMalloc(&d_sphere, sizeof(Trace::SphereSoA));
        cudaMalloc(&d_position, sizeof(float3));
        cudaMalloc(&d_radius, sizeof(float));
        cudaMalloc(&d_material, sizeof(Trace::Material*)); // Allocate space for two material pointers

        // Initialize other values if needed
        position = make_float3(0.0f, 0.0f, 0.0f);
        velocity = make_float3(0.0f, 0.0f, 0.0f);
        radius = 6371000.0f;

        // Load textures
        loadData();
    }

    __host__ static Earth* allocate() {
        Earth* d_earth;
        

        cudaMalloc(&d_earth, sizeof(Earth));

        // Host side initialization
        Earth* h_earth = new Earth();
        h_earth->loadData();

        // Copy to device
        cudaMemcpy(d_earth, h_earth, sizeof(Earth), cudaMemcpyHostToDevice);



        return d_earth;
    }

    __host__ ~Earth() {
        cudaFree(d_sphere);
        cudaFree(d_position);
        cudaFree(d_radius);
        cudaFree(d_material);
    }

    __device__ void initProperties(Trace::SphereSoA* sphPtr, float3* posPtr, float* radPtr) {
        sphPtr->center = posPtr;
        sphPtr->center[0] = this->position;
        sphPtr->radius = radPtr;
        sphPtr->radius[0] = this->radius;
        sphPtr->size += 1;
    }

    __device__ void updateTransform(Trace::Material** d_mats, Frame<ITRF>& itrf, double simtime) {
        Transform t = itrf.getTransform(simtime);
        d_mats[0]->updateTransform(t);
    }

    __device__ void initEarthMaterial(Trace::Material** d_mats) {

        EarthMaterial* earthMaterial = new EarthMaterial;
        earthMaterial->dayTex = dayTex;
        earthMaterial->nightTex = nightTex;
        d_mats[0] = earthMaterial;
    }

    void loadData() {
        position = make_float3(0.0f, 0.0f, 0.0f);
        velocity = make_float3(0.0f, 0.0f, 0.0f);
        radius = 6371000.0f;

        // Day time texture
        Img day_img = loadImage("res/earth.jpg");
        day_img.generateCudaTexture();
        dayTex = day_img.getTexture();

        // Night time texture
        Img night_img = loadImage("res/earth_night.jpg");
        night_img.generateCudaTexture();
        nightTex = night_img.getTexture();

    }

};