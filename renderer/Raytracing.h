#pragma once

#include <simd/simd.h>

#ifndef __METAL_VERSION__
using uint = unsigned int;
using packed_uint3 = uint[3];
using packed_float3 = float[3];
#endif

#define ENABLE_TONE_MAPPING         false
#define MANUAL_SRGB                 false

#define ACCUMULATE_IMAGE            true

#define DISTANCE_EPSILON            0.0001
#define ANGLE_EPSILON               0.00003807693583
#define PI                          3.1415926
#define NOISE_DIMENSIONS            64
#define ANIMATE_NOISE               1

#define MAX_FRAMES                  0
#define MAX_PATH_LENGTH             8

#define CONTENT_SCALE               (1.0f / 2.0f)

#define COMPARE_DISABLED            0
#define COMPARE_ABSOLUTE_VALUE      1 // just compare abs(color - ref) or abs(ref - color)
#define COMPARE_REF_TO_COLOR        2 // visible, if output is darker than reference
#define COMPARE_COLOR_TO_REF        3 // visible, if reference is darker than output
#define COMPARE_LUMINANCE           4 // red/green, red - output brighter, green - reference brighter
#define COMPARISON_MODE             COMPARE_DISABLED
#define COMPARISON_SCALE            10

enum : uint
{
    MATERIAL_DIFFUSE,
    MATERIAL_MIRROR,
    MATERIAL_SMOOTH_PLASTIC,
    MATERIAL_SMOOTH_DIELECTRIC,
    
    MATERIAL_COUNT
};

#include "Spectrum.h"

struct SharedData
{
    uint frameIndex;
    uint lightTrianglesCount;
    float time;
};

struct Ray
{
    packed_float3 origin; // 3
    float minDistance; // 1
    
    packed_float3 direction; // 3
    float maxDistance; // 1
    
    Spectrum throughput; // 3
    Spectrum radiance; // 3
    
    vector_float4 params; // 3 // material pdf, light pdf, bounce, ior

    // ---------------------
    // 4 + 4 + 9 = 17 * 4 = 68 bytes
};

struct LightSamplingRay
{
    packed_float3 origin; // 3
    float minDistance; // 1
    
    packed_float3 direction; // 3
    float maxDistance; // 1
    
    Spectrum throughput; // 3
    uint targetIndex; // 1
    // ---------------------
    // 3x4 = 12 * 4 = 48 bytes
};

struct Intersection
{
    float distance;
    uint triangleIndex;
    packed_float2 coordinates;
};

struct Vertex
{
    packed_float3 v;
    packed_float3 n;
};

struct Material
{
    Spectrum diffuse;
    Spectrum emissive;
    float ior;
    uint materialType;
};

struct TriangleReference
{
    packed_uint3 tri;
    uint materialIndex;
    uint lightTriangleIndex;
};

struct LightTriangle
{
    Spectrum emissive;
    Vertex v1;
    Vertex v2;
    Vertex v3;
    float area;
    float pdf;
    float cdf;
    uint index;
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

float halton(uint index, uint base)
{
    float f = 1.0f;
    float r = 0.0f;
    float fbase = float(base);
    
    while (index > 0)
    {
        r += f * (index % base);
        f = f / fbase;
        index = index / base;
    }
    
    return r;
}

float vanDerCorput(uint index, uint base)
{
    float result = 0.0;
    float baseInv = 1.0f / float(base);
    
    while (index > 0)
    {
        result += float(index % base) * baseInv;
        baseInv *= baseInv;
        index = index / base;
    }
    
    return result;
}

float triangleSamplePDF(float area, float cosTheta, float distanceToSample)
{
    return (distanceToSample * distanceToSample) / (area * cosTheta);
}

float balanceHeuristic(float fPdf, float gPdf)
{
    float f2 = fPdf * fPdf;
    float g2 = gPdf * gPdf;
    return f2 / (f2 + g2);
}

#ifdef __METAL_VERSION__

float3 barycentric(float2 smp)
{
    float r1 = sqrt(smp.x);
    float r2 = smp.y;
    return float3(1.0f - r1, r1 * (1.0f - r2), r1 * r2);
}

void buildOrthonormalBasis(thread const float3& n, thread float3& u, thread float3& v)
{
    if (n.z < 0.0f)
    {
        float a = 1.0f / (1.0f - n.z);
        float b = n.x * n.y * a;
        u = float3(1.0f - n.x * n.x * a, -b, n.x);
        v = float3(b, n.y * n.y * a - 1.0f, -n.y);
    }
    else
    {
        float a = 1.0f / (1.0f + n.z);
        float b = -n.x * n.y * a;
        u = float3(1.0f - n.x * n.x * a, b, -n.x);
        v = float3(b, 1.0f - n.y * n.y * a, -n.y);
    }
}

float3 alignWithNormal(thread const float3& n, float cosTheta, float phi)
{
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    
    float3 u;
    float3 v;
    buildOrthonormalBasis(n, u, v);
    
    return (u * cos(phi) + v * sin(phi)) * sinTheta + n * cosTheta;
}

float3 generateDiffuseBounce(float2 smp, thread const float3& n)
{
    float cosTheta = sqrt(smp.y);
    float phi = smp.x * PI * 2.0;
    return alignWithNormal(n, cosTheta, phi);
}

float3 generateMirrorBounce(float3 wIn, float3 n)
{
    return reflect(wIn, n);
}

#endif

