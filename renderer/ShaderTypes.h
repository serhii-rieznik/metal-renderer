#pragma once

#include <simd/simd.h>

#ifdef __METAL_VERSION__
#   define pow metal::pow
#   define packed_float2(x) packed_float2 x
#   define packed_float3(x) packed_float3 x
#else
#   include <math.h>
#   define pow powf
#   define packed_float2(x) float x[2]
#   define packed_float3(x) float x[3]
#endif

#define ENABLE_TONE_MAPPING     false

#define ACCUMULATE_IMAGE        true
#define NEXT_EVENT_ESTIMATION   true
#define MANUAL_SRGB             true 

#define DISTANCE_EPSILON        0.001
#define ANGLE_EPSILON           0.00174533
#define PI                      3.1415926
#define NOISE_DIMENSIONS        64
#define ANIMATE_NOISE           1
#define MAX_FRAMES              0
#define MAX_PATH_LENGTH         8
#define CONTENT_SCALE           (1.0f / 2.0f)

struct SharedData
{
    unsigned int frameIndex;
    unsigned int lightTrianglesCount;
    float time;
};

struct Ray
{
    packed_float3(origin);
    float minDistance;
    packed_float3(direction);
    float maxDistance;
    packed_float3(throughput);
    packed_float3(radiance);
    unsigned int sourceIndex[4];
    unsigned int targetIndex;
    unsigned int bounce;
};

struct Intersection
{
    float distance;
    unsigned int triangleIndex;
    packed_float2 coordinates;
};

struct Vertex
{
    packed_float3(v);
    packed_float3(n);
    packed_float2(t);
};

struct Material
{
    packed_float3(diffuse);
    packed_float3(emissive);
};

struct LightTriangle
{
    float area;
    float pdf;
    float cdf;
    unsigned int index;
};

float toLinear(float value)
{
    return (value < 0.04045f) ? (value / 12.92) : pow((value + 0.055f) / 1.055f, 2.4);
}

float toSRGB(float value)
{
    if (value <= 0.0f) return 0.0f;
    if (value >= 1.0f) return 1.0f;
    return (value < 0.0031308) ? (12.92f * value) : (1.055f * pow(value, 1.0f / 2.4f) - 0.055);
}
