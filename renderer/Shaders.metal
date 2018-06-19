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
    float4 color = image.sample(linearSampler, in.coords);
    color = 1.0 - exp(-color);
    return color;
}

/*
 * Raytracing
 */
kernel void rayGenerator(device Ray* rays [[buffer(0)]],
                         constant SharedData& sharedData [[buffer(1)]],
                         uint2 threadId [[thread_position_in_grid]],
                         uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    float2 normalizedCoords = ((float2)(threadId) / (float2)(wholeSize)) * 2.0f - 1.0f;
    float aspect = float(wholeSize.y) / float(wholeSize.x);
    
    float t = sharedData.time;
    float ct = cos(t);
    float st = sin(t);
    float3 side = float3(ct, 0.0, st);
    float3 up = float3(0.0, 1.0, 0.0);
    float3 view = float3(st, 0.0, -ct);

    rays[rayIndex].origin = up - view * 2.35;
    rays[rayIndex].direction = normalize(side * normalizedCoords.x + up * (normalizedCoords.y * aspect) + view);
    rays[rayIndex].minDistance = 0.0f;
    rays[rayIndex].maxDistance = INFINITY;
}

template <class T>
T interpolate(thread const T& t0, thread const T& t1, thread const T& t2, thread const packed_float2& uv)
{
    return t0 * uv.x + t1 * uv.y + t2 * (1.0f - uv.x - uv.y);
}

kernel void intersectionHandler(texture2d<float, access::write> image [[texture(0)]],
                                device const Intersection* intersections [[buffer(0)]],
                                device const Vertex* vertexBuffer [[buffer(1)]],
                                device const uint* indexBuffer [[buffer(2)]],
                                device const uint* materialIndexBuffer [[buffer(3)]],
                                device const Material* materialBuffer [[buffer(4)]],
                                constant SharedData& sharedData [[buffer(5)]],
                                uint2 threadId [[thread_position_in_grid]],
                                uint2 wholeSize [[threads_per_grid]])
{
    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    
    Intersection intersection = intersections[rayIndex];
    if (intersection.distance >= 0.0f)
    {
        uint m = materialIndexBuffer[intersection.index];
        uint i0 = indexBuffer[3 * intersection.index + 0];
        uint i1 = indexBuffer[3 * intersection.index + 1];
        uint i2 = indexBuffer[3 * intersection.index + 2];
        device const Vertex& v0 = vertexBuffer[i0];
        device const Vertex& v1 = vertexBuffer[i1];
        device const Vertex& v2 = vertexBuffer[i2];
        float3 n = interpolate(v0.n, v1.n, v2.n, intersection.coordinates) * 0.5 + 0.5;
        color.xyz = materialBuffer[m].diffuse * (n.y * 0.5 + 0.5) + materialBuffer[m].emissive;
    }
    image.write(color, threadId);
}
