#include <metal_stdlib>
#import "ShaderTypes.h"

using namespace metal;

/*
 * Texture blitting functions
 */
struct BlitVertexOut
{
    float4 pos [[position]];
    float2 coords;
};

constant constexpr static const float4 fullscreenTrianglePositions[3]
{
    {-1.0, -1.0, 0.0, 1.0},
    { 3.0, -1.0, 0.0, 1.0},
    {-1.0,  3.0, 0.0, 1.0}
};

vertex BlitVertexOut blitVertex(uint vertexIndex [[vertex_id]])
{
    BlitVertexOut out;
    out.pos = fullscreenTrianglePositions[vertexIndex];
    out.coords = out.pos.xy * 0.5 + 0.5;
    return out;
}

fragment float4 blitFragment(BlitVertexOut in [[stage_in]]
                             , constant SharedData& sharedData [[buffer(0)]]
                             , texture2d<float> image [[texture(0)]]
#                            if (COMPARISON_MODE != COMPARE_DISABLED)
                             , texture2d<float> reference [[texture(1)]]
#                            endif
                             )
{
    constexpr sampler linearSampler(coord::normalized, filter::nearest);
    float4 color = image.sample(linearSampler, in.coords);
    
#if (ENABLE_TONE_MAPPING)
    color = 1.0 - exp(-color);
#endif
    
#if (MANUAL_SRGB)
    color.x = toSRGB(color.x);
    color.y = toSRGB(color.y);
    color.z = toSRGB(color.z);
#endif
    
#if (COMPARISON_MODE == COMPARE_ABSOLUTE_VALUE)
    float4 ref = reference.sample(linearSampler, in.coords);
    return abs(color - ref) * float(COMPARISON_SCALE);
#elif (COMPARISON_MODE == COMPARE_REF_TO_COLOR)
    float4 ref = reference.sample(linearSampler, in.coords);
    return max(0.0, ref - color) * float(COMPARISON_SCALE);
#elif (COMPARISON_MODE == COMPARE_COLOR_TO_REF)
    float4 ref = reference.sample(linearSampler, in.coords);
    return max(0.0, color - ref) * float(COMPARISON_SCALE);
#elif (COMPARISON_MODE == COMPARE_LUMINANCE)
    float4 ref = reference.sample(linearSampler, in.coords);
    float lumColor = dot(color.xyz, float3(0.2126, 0.7152, 0.0722));
    float lumRef = dot(ref.xyz, float3(0.2126, 0.7152, 0.0722));
    return float4(max(0.0, lumColor - lumRef), max(0.0, lumRef - lumColor), 0.0, 1.0) * float(COMPARISON_SCALE);
#else
    return color;
#endif
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
    float aspect = float(wholeSize.y) / float(wholeSize.x);
    
    float t = 0.0f; // sin(sharedData.time) * 0.1;
    float ct = cos(t);
    float st = sin(t);
    float3 side = float3(ct, 0.0, st);
    float3 up = float3(0.0, 1.0, 0.0);
    float3 view = float3(st, 0.0, -ct);

    float4 noiseSample = noise[(threadId.x % NOISE_DIMENSIONS) + (threadId.y % NOISE_DIMENSIONS) * NOISE_DIMENSIONS];
    float2 dudv = (noiseSample.xy * 2.0 - 1.0) / float2(wholeSize);

    float2 normalizedCoords = float2(2 * threadId.x, 2 * threadId.y) / float2(wholeSize.x - 1, wholeSize.y - 1) - 1.0f;

    rays[rayIndex].origin = up - view * 2.35;
    rays[rayIndex].direction = normalize(side * (dudv.x + normalizedCoords.x) + up * (dudv.y + normalizedCoords.y * aspect) + view);
    rays[rayIndex].minDistance = 0.0;
    rays[rayIndex].maxDistance = INFINITY;
    rays[rayIndex].throughput = 1.0;
    rays[rayIndex].radiance = 0.0;
    rays[rayIndex].bounce = 0;
    rays[rayIndex].targetIndex = -1;
}

Vertex interpolate(device const Vertex& t0, device const Vertex& t1, device const Vertex& t2, float3 uvw)
{
    Vertex result;
    result.v = t0.v * uvw.x + t1.v * uvw.y + t2.v * uvw.z;
    result.n = t0.n * uvw.x + t1.n * uvw.y + t2.n * uvw.z;
    result.t = t0.t * uvw.x + t1.t * uvw.y + t2.t * uvw.z;
    result.n = normalize(result.n);
    return result;
}

Vertex interpolate(device const Vertex& t0, device const Vertex& t1, device const Vertex& t2, float2 uv)
{
    return interpolate(t0, t1, t2, float3(uv, 1.0 - uv.x - uv.y));
}

LightTriangle selectLightTriangle(float xi, device const LightTriangle* lightTriangles, int trianglesCount)
{
    int index = 0;
    for (; (index < trianglesCount) && (lightTriangles[index + 1].cdf <= xi); ++index);
    return lightTriangles[index];
}

float triangleSamplePDF(float area, float cosTheta, float distanceToSample)
{
    return (distanceToSample * distanceToSample) / (area * cosTheta);
}

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

float3 generateDirection(float2 smp, thread const float3& n)
{
    float3 u;
    float3 v;
    buildOrthonormalBasis(n, u, v);
    
#if (DIFFUSE_COSINE_WEIGHTED)
    float cosTheta = sqrt(smp.y);
#else
    float cosTheta = smp.y;
#endif
    
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float phi = smp.x * PI * 2.0;
    return (u * cos(phi) + v * sin(phi)) * sinTheta + n * cosTheta;
}

float balanceHeuristic(float fPdf, float gPdf)
{
    float f2 = fPdf * fPdf;
    float g2 = gPdf * gPdf;
    return f2 / (f2 + g2);
}

kernel void intersectionHandler(texture2d<float, access::read_write> image [[texture(0)]],
                                device const Intersection* intersections [[buffer(0)]],
                                device const Vertex* vertexBuffer [[buffer(1)]],
                                device const packed_uint3* indexBuffer [[buffer(2)]],
                                device const uint* materialIndexBuffer [[buffer(3)]],
                                device const Material* materialBuffer [[buffer(4)]],
                                constant SharedData& sharedData [[buffer(5)]],
                                device const LightTriangle* lightTriangles [[buffer(6)]],
                                device Ray* primaryRays [[buffer(7)]],
                                device const float4* noise [[buffer(8)]],
                                device Ray* lightSamplingRays [[buffer(9)]],
                                uint2 threadId [[thread_position_in_grid]],
                                uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    lightSamplingRays[rayIndex].maxDistance = -1.0f;

    device Ray& ray = primaryRays[rayIndex];
    device const Intersection& intersection = intersections[rayIndex];
    if (intersection.distance < DISTANCE_EPSILON)
    {
        primaryRays[rayIndex].maxDistance = -1.0f;
        return;
    }
    
    device const packed_uint3& hitTriangle = indexBuffer[intersection.triangleIndex];
    device const Material& material = materialBuffer[materialIndexBuffer[intersection.triangleIndex]];

    uint noiseIndex =
        ((threadId.x + ray.bounce + sharedData.frameIndex / 3) % NOISE_DIMENSIONS) +
        ((threadId.y + ray.bounce + sharedData.frameIndex / 5) % NOISE_DIMENSIONS) * NOISE_DIMENSIONS;
    
    device const float4& noiseSample = noise[noiseIndex];

    Vertex hitVertex = interpolate(vertexBuffer[hitTriangle.x], vertexBuffer[hitTriangle.y], vertexBuffer[hitTriangle.z], intersection.coordinates);
    {
        if (ray.bounce == 0) // show directly visible emitters
        {
            ray.radiance += ray.throughput * material.emissive;
        }
        
        ray.origin = hitVertex.v + hitVertex.n * DISTANCE_EPSILON;
        ray.direction = generateDirection(noiseSample.xy, hitVertex.n);
        ray.minDistance = 0.0f;
        ray.maxDistance = INFINITY;
        ray.bounce += 1;

        float cosTheta = dot(ray.direction, hitVertex.n);
        float bsdf = (1.0f / PI) * cosTheta;

#   if (DIFFUSE_COSINE_WEIGHTED)
        float pdf = (1.0f / PI) * cosTheta;
#   else
        float pdf = 1.0f / (2.0f * PI);
#   endif
        
        ray.throughput *= material.diffuse * (bsdf / pdf);
    }
    
    if (ray.bounce + 1 > MAX_PATH_LENGTH)
        return;
    
    LightTriangle selectedLightTriangle = selectLightTriangle(noiseSample.z, lightTriangles, sharedData.lightTrianglesCount);
    if (selectedLightTriangle.index == intersection.triangleIndex)
        return;
    
    device const packed_uint3& lightTriangle = indexBuffer[selectedLightTriangle.index];
    Vertex lightVertex = interpolate(vertexBuffer[lightTriangle.x], vertexBuffer[lightTriangle.y], vertexBuffer[lightTriangle.z], barycentric(noiseSample.wx));
    packed_float3 directionToLight = lightVertex.v - hitVertex.v;
    float distanceToLight = length(directionToLight);
    directionToLight /= distanceToLight;
    
    if (distanceToLight < DISTANCE_EPSILON)
        return;

    float LdotD = -dot(directionToLight, lightVertex.n);
    if (LdotD < ANGLE_EPSILON)
        return;

    float cosTheta = dot(directionToLight, hitVertex.n);
    float lightSampleBsdf = (1.0f / PI) * cosTheta;
    float lightSamplePdf = selectedLightTriangle.pdf * triangleSamplePDF(selectedLightTriangle.area, LdotD, distanceToLight);
    
    lightSamplingRays[rayIndex].origin = hitVertex.v + hitVertex.n * DISTANCE_EPSILON;
    lightSamplingRays[rayIndex].direction = directionToLight;
    lightSamplingRays[rayIndex].minDistance = 0.0;
    lightSamplingRays[rayIndex].maxDistance = INFINITY;
    lightSamplingRays[rayIndex].targetIndex = selectedLightTriangle.index;
    lightSamplingRays[rayIndex].throughput = ray.throughput * (lightSampleBsdf / lightSamplePdf);
}

kernel void nextEventEstimationHandler(texture2d<float, access::read_write> image [[texture(0)]],
                                       device const Intersection* intersections [[buffer(0)]],
                                       device Ray* primaryRays [[buffer(1)]],
                                       constant SharedData& sharedData [[buffer(2)]],
                                       device const uint* materialIndexBuffer [[buffer(3)]],
                                       device const Material* materialBuffer [[buffer(4)]],
                                       device const Ray* lightSamplingRays [[buffer(5)]],
                                       uint2 threadId [[thread_position_in_grid]],
                                       uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    device const Intersection& intersection = intersections[rayIndex];
    if ((intersection.distance >= DISTANCE_EPSILON) && (intersection.triangleIndex == lightSamplingRays[rayIndex].targetIndex))
    {
        device const Material& material = materialBuffer[materialIndexBuffer[intersection.triangleIndex]];
        primaryRays[rayIndex].radiance += lightSamplingRays[rayIndex].throughput * material.emissive;
    }
}

kernel void accumulateImage(texture2d<float, access::read_write> image [[texture(0)]],
                            device Ray* sourceRays [[buffer(0)]],
                            constant SharedData& sharedData [[buffer(1)]],
                            uint2 threadId [[thread_position_in_grid]],
                            uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    float3 color = sourceRays[rayIndex].radiance;
    if (ACCUMULATE_IMAGE && (sharedData.frameIndex > 0))
    {
        float3 stored = image.read(threadId).xyz;
        color = mix(color, stored, float(sharedData.frameIndex) / float(sharedData.frameIndex + 1));
    }
    image.write(float4(color, 1.0), threadId);
}
