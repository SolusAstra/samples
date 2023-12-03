#pragma once
#include <sutil/vec_math.h>

namespace Trace {

    class Camera {

    public:



    private:
        float3 _position = make_float3(0.0f, 0.0f, 0.0f);
        float3 _target = make_float3(0.0f);
        float vFOV = 0.0f;
        float AR = 0.0f;

        float3 u = make_float3(1.0f, 0.0f, 0.0f);
        float3 v = make_float3(0.0f, 1.0f, 0.0f);
        float3 w = make_float3(0.0f, 0.0f, 1.0f);

        float3 u_rot = make_float3(1.0f, 0.0f, 0.0f);
        float3 v_rot = make_float3(0.0f, 1.0f, 0.0f);
        float3 w_rot = make_float3(0.0f, 0.0f, 1.0f);

    public:

        __host__ Camera() {}
        __host__ Camera(const float3& position, const float3& target, const float vFOV, const float AR) :
            _position(position), _target(target), vFOV(vFOV), AR(AR) {

            updateRotationMatrix();
        }

        __host__ __device__ float3 getPosition() { return _position; }
        __host__ __device__ float3 getTarget() { return _target; }

        __host__ void updateRotationMatrix() {
            sensorFrame(getPosition(), getTarget());
            
        }

        __host__ void setPosition(float3& position) {
            this->_position = position;
            updateRotationMatrix();
        }
        __host__ void setTarget(float3& target) {
            this->_target = target;
            updateRotationMatrix();
        }

        __host__ float computeDistanceToTarget() {

            return length(_target - _position);
        }

        __host__ __device__ float2 worldToPixel(const float3& worldPoint, const float screenWidth, const float screenHeight)
        {
            // Transform world point to camera space
            float3 viewSpacePoint = worldPoint - _position;
            viewSpacePoint = make_float3(dot(viewSpacePoint, u_rot), dot(viewSpacePoint, v_rot), dot(viewSpacePoint, w_rot));

            // Compute the scale factors based on FOV and aspect ratio
            float scaleHeight = tanf(vFOV * 0.5f * M_PIf / 180.0f); // Vertical scale
            float scaleWidth = AR * scaleHeight; // Horizontal scale

            // Project onto camera's image plane, considering FOV and aspect ratio
            float3 imagePlanePoint = make_float3(-viewSpacePoint.x / (viewSpacePoint.z * scaleWidth),
                -viewSpacePoint.y / (viewSpacePoint.z * scaleHeight),
                1.0f);

            // Convert to pixel coordinates
            float x = (1.0f - (imagePlanePoint.x + 1.0f) * 0.5f) * screenWidth;
            float y = (1.0f - (imagePlanePoint.y + 1.0f) * 0.5f) * screenHeight;

            return make_float2(x, y);
        }

        __host__ __device__ float3 pixelToDirection(float xPixel, float yPixel, float screenWidth, float screenHeight) {
            // Convert pixel coordinates to normalized device coordinates (NDC)
            float xNDC = (xPixel / screenWidth) * 2.0f - 1.0f;
            float yNDC = (yPixel / screenHeight) * 2.0f - 1.0f;

            // Invert the projection transform
            float scaleHeight = tanf(vFOV * 0.5f * M_PIf / 180.0f); // Vertical scale
            float scaleWidth = AR * scaleHeight; // Horizontal scale

            float3 imagePlanePoint = make_float3(xNDC * scaleWidth, yNDC * scaleHeight, -1.0f);

            // Transform back to world space
            float3 direction = normalize(make_float3(-dot(imagePlanePoint, u_rot), -dot(imagePlanePoint, v_rot), dot(imagePlanePoint, w_rot)));

            return direction;
        }

        __host__ __device__ float3 worldToCameraFrame(const float3& worldVector)
        {
            // Rotate the world space vector to align with the camera's orientation (pure rotation)
            return make_float3(dot(worldVector, u_rot), dot(worldVector, v_rot), dot(worldVector, w_rot));
        }

        

        __forceinline __device__ float3 getPixelPosition(float i, float j)
        {
            return u * (i - 0.5f) + v * (j - 0.5f) - w;
        }

    private:

        __host__ void sensorFrame(const float3& position, const float3& target)
        {
            const float3 n = make_float3(0.0f, 1.0f, 0.0f);

            // Calculate the camera frame (pure rotation)
            w_rot = normalize(position - target);
            u_rot = normalize(cross(n, w_rot));
            v_rot = normalize(cross(w_rot, u_rot));

            // Update u, v, and w for viewport calculations
            u = u_rot;
            v = v_rot;
            w = w_rot;
            updateViewPort();
        }

        __host__ void updateViewPort() {
            float halfHeight = 2.0f * tanf(vFOV * M_PIf / 360.0f);
            float halfWidth = AR * halfHeight;
            u = u_rot * halfWidth;
            v = v_rot * halfHeight;
        }

    };

};

//__host__ float3 transformToCameraSpace(const float3& worldPoint) {
//    // Translate the world point relative to the camera position
//    float3 translatedPoint = _position - worldPoint;
//
//    // Rotate the translated point to align with the camera's axes
//    float3 a = make_float3(u.x, v.x, w.x);
//    float3 b = make_float3(u.y, v.y, w.y);
//    float3 c = make_float3(u.z, v.z, w.z);
//
//
//
//    return make_float3(dot(translatedPoint, a), dot(translatedPoint, b), dot(translatedPoint, c));
//}
//
//__host__ float3 worldToCameraFrame(const float3& worldPoint) {
//    // Transform the point from world space to camera space
//
//    float halfHeight = 2.0f * tanf(vFOV * M_PIf / 360.0f);
//    float halfWidth = AR * halfHeight;
//
//    float3 rotatedPoint = transformToCameraSpace(worldPoint);
//
//    return make_float3(rotatedPoint.x, rotatedPoint.y, rotatedPoint.z);
//}