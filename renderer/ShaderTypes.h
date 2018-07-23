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
#define MANUAL_SRGB             false

#define ACCUMULATE_IMAGE        true

#define DISTANCE_EPSILON        0.0001
#define ANGLE_EPSILON           0.00003807693583
#define PI                      3.1415926
#define NOISE_DIMENSIONS        64
#define ANIMATE_NOISE           1

#define MAX_FRAMES              0
#define MAX_PATH_LENGTH         3

#define CONTENT_SCALE           (1.0f / 2.0f)

#define COMPARE_DISABLED        0
#define COMPARE_ABSOLUTE_VALUE  1 // just compare abs(color - ref) or abs(ref - color)
#define COMPARE_REF_TO_COLOR    2 // visible, if output is darker than reference
#define COMPARE_COLOR_TO_REF    3 // visible, if reference is darker than output
#define COMPARE_LUMINANCE       4 // red/green, red - output brighter, green - reference brighter
#define COMPARISON_MODE         COMPARE_DISABLED
#define COMPARISON_SCALE        10

#define MATERIAL_DIFFUSE    0
#define MATERIAL_MIRROR     1

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
    unsigned int targetIndex;
    unsigned int bounce;
    float materialPdf;
    float lightPdf;
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
    unsigned int materialType;
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

float halton(unsigned int index, unsigned int base)
{
    float f = 1.0f;
    float r = 0.0f;
    float fbase = float(base);
    
    while (index > 0)
    {
        f = f / fbase;
        r = r + f * (index % base);
        index = index / base;
    }
    
    return r;
}


float vanDerCorput(unsigned int t, unsigned int b)
{
    float r = 0.0;
    float base_inv = 1.0 / float(b);
    
    while (t > 0)
    {
        unsigned int d = ( t % b );
        r = r + float(d) * base_inv;
        base_inv = base_inv / b;
        t = t / b;
    }
    
    return r;
}
