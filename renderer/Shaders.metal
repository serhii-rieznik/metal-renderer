#include <metal_stdlib>
#import "ShaderTypes.h"

using namespace metal;

#define ACCUMULATE_IMAGE true
#define ENABLE_TONE_MAPPING true
#define DISTANCE_EPSILON 0.001

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
#if (ENABLE_TONE_MAPPING)
    color = color / (1.0 + color); // 1.0 - exp(-color);
#endif
    return color;
}

/*
 * Raytracing
 */
kernel void rayGenerator(device Ray* rays [[buffer(0)]],
                         constant SharedData& sharedData [[buffer(1)]],
                         device const float4* noise [[buffer(2)]],
                         uint2 threadId [[thread_position_in_grid]],
                         uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    float2 normalizedCoords = ((float2)(threadId) / (float2)(wholeSize)) * 2.0f - 1.0f;
    float aspect = float(wholeSize.y) / float(wholeSize.x);
    
    float t = 0.0; // sin(sharedData.time) * 0.1;
    float ct = cos(t);
    float st = sin(t);
    float3 side = float3(ct, 0.0, st);
    float3 up = float3(0.0, 1.0, 0.0);
    float3 view = float3(st, 0.0, -ct);

    float4 noiseSample = noise[(threadId.x % NOISE_DIMENSIONS) + (threadId.y % NOISE_DIMENSIONS) * NOISE_DIMENSIONS];
    float2 dudv = 2.0f * (noiseSample.xy * 2.0 - 1.0) / float2(wholeSize);

    rays[rayIndex].origin = up - view * 2.35;
    rays[rayIndex].direction = normalize(side * (dudv.x + normalizedCoords.x) + up * (dudv.y + normalizedCoords.y * aspect) + view);
    rays[rayIndex].minDistance = DISTANCE_EPSILON;
    rays[rayIndex].maxDistance = INFINITY;
    rays[rayIndex].throughput = 1.0;
    rays[rayIndex].radiance = 0.0;
}

Vertex interpolate(device const Vertex& t0, device const Vertex& t1, device const Vertex& t2, packed_float2 uv)
{
    float w = (1.0f - uv.x - uv.y);
    Vertex result;
    result.v = t0.v * uv.x + t1.v * uv.y + t2.v * w;
    result.n = t0.n * uv.x + t1.n * uv.y + t2.n * w;
    result.t = t0.t * uv.x + t1.t * uv.y + t2.t * w;
    return result;
}

LightTriangle selectLightTriangle(float xi, device const LightTriangle* lightTriangles, int trianglesCount)
{
    int index = 0;
    for (; (index < trianglesCount) && (lightTriangles[index + 1].cdf <= xi); ++index);
    return lightTriangles[index];
}

float triangleSamplePDF(float area, float cosTheta, float distanceToSample)
{
    return (distanceToSample * distanceToSample) / (cosTheta * area);
}

kernel void intersectionHandler(texture2d<float, access::read_write> image [[texture(0)]],
                                device const Intersection* intersections [[buffer(0)]],
                                device const Vertex* vertexBuffer [[buffer(1)]],
                                device const packed_uint3* indexBuffer [[buffer(2)]],
                                device const uint* materialIndexBuffer [[buffer(3)]],
                                device const Material* materialBuffer [[buffer(4)]],
                                constant SharedData& sharedData [[buffer(5)]],
                                device const LightTriangle* lightTriangles [[buffer(6)]],
                                device Ray* rays [[buffer(7)]],
                                device const float4* noise [[buffer(8)]],
                                uint2 threadId [[thread_position_in_grid]],
                                uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    uint noiseIndex = (threadId.x % NOISE_DIMENSIONS) + (threadId.y % NOISE_DIMENSIONS) * NOISE_DIMENSIONS;

    device const Intersection& intersection = intersections[rayIndex];
    rays[rayIndex].maxDistance = -1.0;
    
    if (intersection.distance < 0.0f)
        return;
    
    device const packed_uint3& hitTriangle = indexBuffer[intersection.triangleIndex];
    device const Material& material = materialBuffer[materialIndexBuffer[intersection.triangleIndex]];
    device const float4& noiseSample = noise[noiseIndex];
    
    rays[rayIndex].radiance = material.emissive;

    Vertex hitVertex = interpolate(vertexBuffer[hitTriangle.x], vertexBuffer[hitTriangle.y], vertexBuffer[hitTriangle.z], intersection.coordinates);
    LightTriangle selectedLightTriangle = selectLightTriangle(noiseSample.x, lightTriangles, sharedData.lightTrianglesCount);
    if (selectedLightTriangle.index == intersection.triangleIndex)
        return;
    
    device const packed_uint3& lightTriangle = indexBuffer[selectedLightTriangle.index];
    Vertex lightVertex = interpolate(vertexBuffer[lightTriangle.x], vertexBuffer[lightTriangle.y], vertexBuffer[lightTriangle.z], noiseSample.zw);
    packed_float3 directionToLight = lightVertex.v - hitVertex.v;
    float distanceToLight = length(directionToLight);
    directionToLight /= distanceToLight;
    
    float LdotD = dot(directionToLight, lightVertex.n);
    if (LdotD >= 0.001)
        return;

    float trianglePdf = selectedLightTriangle.pdf;
    float samplePdf = triangleSamplePDF(selectedLightTriangle.area, -LdotD, distanceToLight);
    float LdotN = dot(directionToLight, hitVertex.n);
    
    rays[rayIndex].origin = hitVertex.v;
    rays[rayIndex].direction = directionToLight;
    rays[rayIndex].throughput = material.diffuse * LdotN / (trianglePdf * samplePdf);
    rays[rayIndex].minDistance = DISTANCE_EPSILON;
    rays[rayIndex].maxDistance = INFINITY;
    rays[rayIndex].targetIndex = selectedLightTriangle.index;
}

kernel void nextEventEstimationHandler(texture2d<float, access::read_write> image [[texture(0)]],
                                       device const Intersection* intersections [[buffer(0)]],
                                       device const Ray* rays [[buffer(1)]],
                                       constant SharedData& sharedData [[buffer(2)]],
                                       device const uint* materialIndexBuffer [[buffer(3)]],
                                       device const Material* materialBuffer [[buffer(4)]],
                                       device const float4* noise [[buffer(5)]],
                                       uint2 threadId [[thread_position_in_grid]],
                                       uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    
    device const Ray& ray = rays[rayIndex];
    device const Intersection& intersection = intersections[rayIndex];
    
    float4 color = float4(ray.radiance, 1.0);
    if ((intersection.distance > 0.0) && (intersection.triangleIndex == ray.targetIndex))
    {
        device const Material& material = materialBuffer[materialIndexBuffer[intersection.triangleIndex]];
        color.xyz += ray.throughput * material.emissive;
    }
    
    if (ACCUMULATE_IMAGE && (sharedData.frameIndex > 0))
    {
        float4 stored = image.read(threadId);
        color = mix(stored, color, 1.0 / (1.0 + sharedData.frameIndex));
        image.write(color, threadId);
    }
    else
    {
        image.write(color, threadId);
    }
}
