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

float balanceHeuristic(float fPdf, float gPdf)
{
    float f2 = fPdf * fPdf;
    float g2 = gPdf * gPdf;
    return f2 / (f2 + g2);
}

float materialBSDF(device const Material& material, thread const float3& wI, thread const float3& wO, thread const float3& n)
{
    float result = 0.0;
    
    switch (material.materialType)
    {
        case MATERIAL_MIRROR:
        {
            result = abs(dot(reflect(wI, n), wO) - 1.0) < ANGLE_EPSILON ? 1.0 : 0.0;
            break;
        }
            
        default:
        {
            result = 1.0 / PI;
            break;
        }
    }
    
    return result * max(0.0, dot(wO, n));
}

float materialPDF(device const Material& material, thread const float3& wI, thread const float3& wO, thread const float3& n)
{
    float result = 0.0;
    
    switch (material.materialType)
    {
        case MATERIAL_MIRROR:
        {
            result = abs(dot(reflect(wI, n), wO) - 1.0) < ANGLE_EPSILON ? 1.0 : 0.0;
            break;
        }
            
        default:
        {
            result = dot(wO, n) / PI;
            break;
        }
    }
    
    return result;
}

float3 generateNextBounce(device const Material& material, thread const float3& wI, thread const float3& n,
                          thread const float2& noiseSample, thread float& bsdf, thread float& pdf)
{
    float3 wO = 0.0;
    
    switch (material.materialType)
    {
        case MATERIAL_MIRROR:
        {
            wO = reflect(wI, n);
            bsdf = 1.0; // materialBSDF(material, wI, wO, n);
            pdf = 1.0; // materialPDF(material, wI, wO, n);
            break;
        }
            
        default:
        {
            wO = generateDiffuseBounce(noiseSample, n);
            bsdf = materialBSDF(material, wI, wO, n);
            pdf = materialPDF(material, wI, wO, n);
            break;
        }
    }
    
    return wO;
}

float lightTriangleSamplePDF(thread const LightTriangle& tri, thread const float3& source, thread const Vertex& sample,
                             thread packed_float3& directionToLight)
{
    directionToLight = sample.v - source;
    
    float distanceToLight = length(directionToLight);
    directionToLight = normalize(directionToLight);
    float LdotD = -dot(directionToLight, sample.n);
    bool validDirection = (distanceToLight >= DISTANCE_EPSILON) && (LdotD >= ANGLE_EPSILON);
    
    return validDirection ? tri.pdf * triangleSamplePDF(tri.area, LdotD, distanceToLight) : 0.0f;
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
                                device Ray* bsdfSamplingRays [[buffer(10)]],
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
    float3 wI = ray.direction;

    uint noiseIndex =
        ((threadId.x + ray.bounce + sharedData.frameIndex / 3) % NOISE_DIMENSIONS) +
        ((threadId.y + ray.bounce + sharedData.frameIndex / 5) % NOISE_DIMENSIONS) * NOISE_DIMENSIONS;
    
    device const float4& noiseSample = noise[noiseIndex];
    
    Vertex hitVertex = interpolate(vertexBuffer[hitTriangle.x], vertexBuffer[hitTriangle.y], vertexBuffer[hitTriangle.z], intersection.coordinates);
    {
        // light sampling
        LightTriangle selectedLightTriangle = selectLightTriangle(noiseSample.z, lightTriangles, sharedData.lightTrianglesCount);
        device const packed_uint3& lightTriangle = indexBuffer[selectedLightTriangle.index];
        Vertex lightVertex = interpolate(vertexBuffer[lightTriangle.x], vertexBuffer[lightTriangle.y], vertexBuffer[lightTriangle.z], barycentric(noiseSample.wx));
        
        packed_float3 directionToLight;
        float lPdf = lightTriangleSamplePDF(selectedLightTriangle, hitVertex.v, lightVertex, directionToLight);
        float bsdf = materialBSDF(material, wI, directionToLight, hitVertex.n);
        float mPdf = materialPDF(material, wI, directionToLight, hitVertex.n);
        float weight = balanceHeuristic(lPdf, mPdf);
        
        bool validLightTriangle = (lPdf > 0.0f) && (selectedLightTriangle.index != intersection.triangleIndex);

        lightSamplingRays[rayIndex].origin = hitVertex.v + hitVertex.n * DISTANCE_EPSILON;
        lightSamplingRays[rayIndex].direction = directionToLight;
        lightSamplingRays[rayIndex].minDistance = 0.0;
        lightSamplingRays[rayIndex].maxDistance = validLightTriangle ? INFINITY : -1.0;
        lightSamplingRays[rayIndex].targetIndex = selectedLightTriangle.index;
        lightSamplingRays[rayIndex].throughput = ray.throughput * material.diffuse * (weight * bsdf / lPdf);
    }
    
    {
        // bsdf sampling
        float pdf = 0.0f;
        float bsdf = 0.0f;
        bsdfSamplingRays[rayIndex].origin = hitVertex.v + hitVertex.n * DISTANCE_EPSILON;
        bsdfSamplingRays[rayIndex].direction = generateNextBounce(material, wI, hitVertex.n, noiseSample.zw, bsdf, pdf);
        bsdfSamplingRays[rayIndex].minDistance = 0.0;
        bsdfSamplingRays[rayIndex].maxDistance = INFINITY;
        bsdfSamplingRays[rayIndex].throughput = ray.throughput * material.diffuse * bsdf;
        bsdfSamplingRays[rayIndex].materialPdf = pdf;
        bsdfSamplingRays[rayIndex].lightPdf = (material.materialType == MATERIAL_MIRROR) ? 0.0 : 1.0;
        bsdfSamplingRays[rayIndex].targetIndex = intersection.triangleIndex;
    }
    
    {
        if (ray.bounce == 0) // show directly visible emitters
        {
            ray.radiance += ray.throughput * material.emissive;
        }
        
        float bsdf = 0.0f;
        float pdf = 0.0f;
        ray.origin = hitVertex.v + hitVertex.n * DISTANCE_EPSILON;
        ray.direction = generateNextBounce(material, wI, hitVertex.n, noiseSample.xy, bsdf, pdf);
        ray.throughput *= material.diffuse * (bsdf / pdf);
        ray.maxDistance = INFINITY;
        ray.minDistance = 0.0f;
        ray.bounce += 1;
    }
}

kernel void lightSamplingHandler(texture2d<float, access::read_write> image [[texture(0)]],
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

kernel void bsdfSamplingHandler(texture2d<float, access::read_write> image [[texture(0)]],
                                device const Intersection* intersections [[buffer(0)]],
                                device Ray* primaryRays [[buffer(1)]],
                                constant SharedData& sharedData [[buffer(2)]],
                                device const uint* materialIndexBuffer [[buffer(3)]],
                                device const Material* materialBuffer [[buffer(4)]],
                                device const Ray* bsdfSamplingRays [[buffer(5)]],
                                device const LightTriangle* lightTriangles [[buffer(6)]],
                                device const Vertex* vertexBuffer [[buffer(7)]],
                                device const packed_uint3* indexBuffer [[buffer(8)]],
                                uint2 threadId [[thread_position_in_grid]],
                                uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    device const Intersection& intersection = intersections[rayIndex];
    if (intersection.distance < DISTANCE_EPSILON) return;
    if (intersection.triangleIndex == bsdfSamplingRays[rayIndex].targetIndex) return;

    device const Material& material = materialBuffer[materialIndexBuffer[intersection.triangleIndex]];
    if (dot(material.emissive, material.emissive) < DISTANCE_EPSILON) return;
    
    LightTriangle selectedLightTriangle;
    for (uint i = 0; i < sharedData.lightTrianglesCount; ++i)
    {
        if (intersection.triangleIndex == lightTriangles[i].index)
        {
            selectedLightTriangle = lightTriangles[i];
            break;
        }
    }
    
    device const packed_uint3& lightTriangle = indexBuffer[selectedLightTriangle.index];
    Vertex lightVertex = interpolate(vertexBuffer[lightTriangle.x], vertexBuffer[lightTriangle.y], vertexBuffer[lightTriangle.z], intersection.coordinates);
    
    packed_float3 directionToLight;
    float mPdf = bsdfSamplingRays[rayIndex].materialPdf;
    float lPdf = bsdfSamplingRays[rayIndex].lightPdf * lightTriangleSamplePDF(selectedLightTriangle, bsdfSamplingRays[rayIndex].origin, lightVertex, directionToLight);
    float weight = balanceHeuristic(mPdf, lPdf);
    
    primaryRays[rayIndex].radiance += material.emissive * bsdfSamplingRays[rayIndex].throughput * weight;
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
