//
// Shaders.metal
// metal-rt Shared
//
// Created by Sergey Reznik on 6/18/18.
// Copyright © 2018 Serhii Rieznik. All rights reserved.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#import "ShaderTypes.h"

using namespace metal;

/*
 * Image write kernel
 */
kernel void imageWriteTest(texture2d<float, access::write> image [[texture(0)]],
                           uint2 threadId [[thread_position_in_grid]],
                           uint2 wholeSize [[threads_per_grid]],
                           uint2 groupId [[thread_position_in_threadgroup]],
                           uint2 groupSize [[threads_per_threadgroup]])
{
    // float2 gid = (float2)(groupId);
    // float2 gsz = (float2)(groupSize);
    float2 tid = (float2)(threadId);
    float2 tsz = (float2)(wholeSize);
    image.write(float4(tid / tsz, 0.0, 1.0), threadId);
}

/*
 * Texture blitting functions
 */
struct BlitVertexOut
{
    float4 pos [[position]];
    float2 coords;
};

constant float4 fullscreenTrianglePositions[3]
{
    {-1.0, -1.0, 0.0, 1.0},
    {3.0, -1.0, 0.0, 1.0},
    {-1.0, 3.0, 0.0, 1.0}
};

vertex BlitVertexOut blitVertex(uint vertexIndex [[vertex_id]])
{
    BlitVertexOut out;
    out.pos = fullscreenTrianglePositions[vertexIndex];
    out.coords = out.pos.xy * 0.5 + 0.5;
    return out;
}

fragment float4 blitFragment(BlitVertexOut in [[stage_in]], texture2d<float> image [[texture(0)]])
{
    constexpr sampler linearSampler(coord::normalized, filter::nearest);
    return image.sample(linearSampler, in.coords);
}

/*
 * Raytracing
 */
kernel void rayGenerator(device Ray* rays [[buffer(0)]],
                         uint2 threadId [[thread_position_in_grid]],
                         uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    float2 normalizedCoords = ((float2)(threadId) / (float2)(wholeSize)) * 2.0f - 1.0f;
    
    rays[rayIndex].origin = float3(normalizedCoords * 2.0f, -1.0f);
    rays[rayIndex].direction = float3(0.0f, 0.0f, 1.0f);
    rays[rayIndex].minDistance = 0.0f;
    rays[rayIndex].maxDistance = 10.0f;
}


kernel void intersectionHandler(texture2d<float, access::write> image [[texture(0)]],
                                device Intersection* intersections [[buffer(0)]],
                                uint2 threadId [[thread_position_in_grid]],
                                uint2 wholeSize [[threads_per_grid]])
{
    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    
    Intersection intersection = intersections[rayIndex];
    if (intersection.distance >= 0.0f)
    {
        float w = 1.0 - intersection.coordinates.x - intersection.coordinates.y;
        color = float4(intersection.coordinates, w, 1.0);
    }
    image.write(color, threadId);
}
