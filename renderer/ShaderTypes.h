#pragma once

#include <simd/simd.h>

#ifdef __METAL_VERSION__
#   define packed_float2(x) packed_float2 x
#   define packed_float3(x) packed_float3 x
#else
#   define packed_float2(x) float x[2]
#   define packed_float3(x) float x[3]
#endif

struct Ray
{
    packed_float3(origin);
    float minDistance;
    packed_float3(direction);
    float maxDistance;
};

struct Intersection
{
    float distance;
    int index;
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

struct SharedData
{
    int frameIndex;
    float time;
};
