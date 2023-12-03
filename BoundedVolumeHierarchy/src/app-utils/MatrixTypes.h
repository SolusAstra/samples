#pragma once
#include <vector_types.h>
#include <vector_functions.h>


struct float3x3 {
    float3 u, v, w; // Rows

    __forceinline __device__ __host__ float3x3() : u(float3{ 0 }), v(float3{ 0 }), w(float3{ 0 }) {}

    __forceinline __device__ __host__ float3x3(const float c) :
        u(float3{ c, 0, 0 }), v(float3{ 0, c, 0 }), w(float3{ 0, 0, c }) {}

    __forceinline __device__ __host__ float3x3(const float3& i, const float3& j, const float3& k) :
        u(float3{ i.x, j.x, k.x }), v(float3{ i.y, j.y, k.y }), w(float3{ i.z, j.z, k.z }) {}

    __forceinline __device__ __host__ float3x3(const float3x3& m) :
        u(float3{ m.u.x, m.u.y, m.u.z }), v(float3{ m.v.x, m.v.y, m.v.z }), w(float3{ m.w.x, m.w.y, m.w.z }) {}

    __forceinline __device__ __host__ static float3x3 zero() { return float3x3(0); }
    __forceinline __device__ __host__ static float3x3 identity() { return float3x3(1); }

};

typedef struct float3x3 float3x3;

__forceinline __device__ __host__ float3x3 make_float3x3(const float3& u, const float3& v, const float3& w) { return float3x3(u, v, w); }

// Transpose
__forceinline __device__ __host__ float3x3 transpose(const float3x3& m) {
    return make_float3x3(make_float3(m.u.x, m.v.x, m.w.x), make_float3(m.u.y, m.v.y, m.w.y), make_float3(m.u.z, m.v.z, m.w.z));
}

// Basic operations
__forceinline __device__ __host__ float3x3 operator+(const float3x3& a, const float3x3& b) {
    return make_float3x3(
        make_float3(a.u.x + b.u.x, a.u.y + b.u.y, a.u.z + b.u.z),
        make_float3(a.v.x + b.v.x, a.v.y + b.v.y, a.v.z + b.v.z),
        make_float3(a.w.x + b.w.x, a.w.y + b.w.y, a.w.z + b.w.z));
}

__forceinline __device__ __host__ float3x3 operator-(const float3x3& a, const float3x3& b) {
    return make_float3x3(
        make_float3(a.u.x - b.u.x, a.u.y - b.u.y, a.u.z - b.u.z),
        make_float3(a.v.x - b.v.x, a.v.y - b.v.y, a.v.z - b.v.z),
        make_float3(a.w.x - b.w.x, a.w.y - b.w.y, a.w.z - b.w.z));
}



// Matrix multiplication 
/*
*           | u.x u.y u.z |       | x |
*       M = | v.x v.y v.z |   v = | y |
*           | w.x w.y w.z |       | z |
*
*                   | u.x * x + u.y * y + u.z * z |
*       w = M * v = | v.x * x + v.y * y + v.z * z |
*                   | w.x * x + w.y * y + w.z * z |
*/
__forceinline __device__ __host__ float3 operator*(const float3x3& M, const float3& v) {
    // Matrix-vector multiplication requries dot products of each row of the matrix with the vector

    return make_float3(
        M.u.x * v.x + M.u.y * v.y + M.u.z * v.z,
        M.v.x * v.x + M.v.y * v.y + M.v.z * v.z,
        M.w.x * v.x + M.w.y * v.y + M.w.z * v.z);
}


