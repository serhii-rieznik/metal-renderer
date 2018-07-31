#include <metal_stdlib>
using namespace metal;

#import "Raytracing.h"
#import "KernelHelpers.h"

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

fragment float4 blitFragment(BlitVertexOut in [[stage_in]], constant SharedData& sharedData [[buffer(0)]],
                             texture2d<float> image [[texture(0)]]
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
    float lumColor = dot(color.xyz, 1.0 / 3.0); // float3(0.2126, 0.7152, 0.0722));
    float lumRef = dot(ref.xyz, 1.0 / 3.0); // float3(0.2126, 0.7152, 0.0722));
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
    float2 dudv = (noiseSample.xy * 2.0 - 1.0) / float2(wholeSize - 1);
    
    float2 normalizedCoords = float2(2 * threadId.x, 2 * threadId.y) / float2(wholeSize - 1) - 1.0f;
    
    rays[rayIndex].origin = up - view * 2.35;
    rays[rayIndex].direction = normalize(side * (dudv.x + normalizedCoords.x) + up * (dudv.y + normalizedCoords.y * aspect) + view);
    rays[rayIndex].minDistance = 0.0;
    rays[rayIndex].maxDistance = INFINITY;
    rays[rayIndex].bounce = 0;
    rays[rayIndex].targetIndex = -1;
    rays[rayIndex].lightPdf = 0.0;
    rays[rayIndex].materialPdf = 1.0;
    spectrum_set(rays[rayIndex].throughput, 1.0f);
    spectrum_set(rays[rayIndex].radiance, 0.0f);
}

kernel void intersectionHandler(device const Intersection* intersections [[buffer(0)]],
                                device const Vertex* vertexBuffer [[buffer(1)]],
                                device const packed_uint3* indexBuffer [[buffer(2)]],
                                device const TriangleReference* referenceBuffer [[buffer(3)]],
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
    spectrum_set(lightSamplingRays[rayIndex].throughput, 0.0f);
    
    device Ray& ray = primaryRays[rayIndex];
    device const Intersection& intersection = intersections[rayIndex];
    if (intersection.distance < DISTANCE_EPSILON)
    {
        primaryRays[rayIndex].maxDistance = -1.0f;
        return;
    }
    
    device const TriangleReference& ref = referenceBuffer[intersection.triangleIndex];
    device const packed_uint3& hitTriangle = indexBuffer[intersection.triangleIndex];
    device const Material& material = materialBuffer[ref.materialIndex];
    float3 wI = ray.direction;
    
    uint noiseIndex =
    ((threadId.x + ray.bounce + sharedData.frameIndex / 3) % NOISE_DIMENSIONS) +
    ((threadId.y + ray.bounce + sharedData.frameIndex / 5) % NOISE_DIMENSIONS) * NOISE_DIMENSIONS;
    
    device const float4& noiseSample = noise[noiseIndex];
    
    Vertex hitVertex = interpolate(vertexBuffer[hitTriangle.x], vertexBuffer[hitTriangle.y], vertexBuffer[hitTriangle.z], intersection.coordinates);
    
    if (material.materialType == MATERIAL_SMOOTH_PLASTIC)
    {
        // float f = fresnel(hitVertex.n, -wI, 1.0, material.ior);
        // spectrum_add_inplace(ray.radiance, spectrum_set(f));
    }
        
    // light sampling
    if (ray.bounce + 1 < MAX_PATH_LENGTH)
    {
        LightTriangle selectedLightTriangle = selectLightTriangle(noiseSample.z, lightTriangles, sharedData.lightTrianglesCount);
        Vertex lightVertex = interpolate(selectedLightTriangle.v1, selectedLightTriangle.v2, selectedLightTriangle.v3, barycentric(noiseSample.wx));
        
        packed_float3 directionToLight;
        float lightPdf = lightTriangleSamplePDF(selectedLightTriangle.pdf, selectedLightTriangle.area, hitVertex.v, lightVertex, directionToLight);
        float2 materialSample = sampleMaterial(material, wI, directionToLight, hitVertex.n, noiseSample);
        float materialBsdf = materialSample.x;
        float materialPdf = materialSample.y;
        float weight = balanceHeuristic(lightPdf, materialPdf);
        
        bool validLightTriangle = (lightPdf > 0.0f) && (selectedLightTriangle.index != intersection.triangleIndex);
        
        Spectrum emittedLight = spectrum_mul(selectedLightTriangle.emissive, 1.0f / lightPdf);
        Spectrum bsdf = spectrum_mul(material.diffuse, materialBsdf);
        Spectrum throughput0 = spectrum_mul(emittedLight, ray.throughput);
        Spectrum throughput1 = spectrum_mul(bsdf, weight);
        
        lightSamplingRays[rayIndex].origin = hitVertex.v + hitVertex.n * DISTANCE_EPSILON;
        lightSamplingRays[rayIndex].direction = directionToLight;
        lightSamplingRays[rayIndex].minDistance = 0.0;
        lightSamplingRays[rayIndex].maxDistance = validLightTriangle ? INFINITY : -1.0;
        lightSamplingRays[rayIndex].targetIndex = selectedLightTriangle.index;
        lightSamplingRays[rayIndex].throughput = spectrum_mul(throughput0, throughput1);
    }
    
    // BSDF sampling
    if (ref.lightTriangleIndex != uint(-1))
    {
        device const LightTriangle& selectedLightTriangle = lightTriangles[ref.lightTriangleIndex];
        
        device const packed_uint3& lightTriangle = indexBuffer[selectedLightTriangle.index];
        Vertex lightVertex = interpolate(vertexBuffer[lightTriangle.x], vertexBuffer[lightTriangle.y], vertexBuffer[lightTriangle.z], intersection.coordinates);
        
        packed_float3 directionToLight;
        float mPdf = ray.materialPdf;
        float lPdf = ray.lightPdf * lightTriangleSamplePDF(selectedLightTriangle.pdf, selectedLightTriangle.area, ray.origin, lightVertex, directionToLight);
        float weight = balanceHeuristic(mPdf, lPdf);
        
        Spectrum added_radiance = spectrum_mul(spectrum_mul(material.emissive, ray.throughput), weight * ray.materialPdf);
        spectrum_add_inplace(ray.radiance, added_radiance);
    }
    
    // generate new ray
    {
        float pdf = 0.0f;
        float bsdf = 0.0f;
        ray.origin = hitVertex.v + hitVertex.n * DISTANCE_EPSILON;
        ray.direction = generateNextBounce(material, wI, hitVertex.n, noiseSample, bsdf, pdf);
        ray.minDistance = 0.0;
        ray.maxDistance = INFINITY;
        ray.materialPdf = pdf;
        ray.lightPdf = (material.materialType == MATERIAL_DIFFUSE) ? 1.0 : 0.0;
        ray.bounce += 1;
        
        Spectrum throughputScale = spectrum_mul(material.diffuse, bsdf / pdf);
        spectrum_mul_inplace(ray.throughput, throughputScale);
    }
}

kernel void lightSamplingHandler(device const Intersection* intersections [[buffer(0)]],
                                 device Ray* primaryRays [[buffer(1)]],
                                 constant SharedData& sharedData [[buffer(2)]],
                                 device const TriangleReference* referenceBuffer [[buffer(3)]],
                                 device const Material* materialBuffer [[buffer(4)]],
                                 device const Ray* lightSamplingRays [[buffer(5)]],
                                 uint2 threadId [[thread_position_in_grid]],
                                 uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    device const Intersection& intersection = intersections[rayIndex];
    if ((intersection.distance >= DISTANCE_EPSILON) && (intersection.triangleIndex == lightSamplingRays[rayIndex].targetIndex))
    {
        spectrum_add_inplace(primaryRays[rayIndex].radiance, lightSamplingRays[rayIndex].throughput);
    }
}

kernel void accumulateImage(texture2d<float, access::read_write> image [[texture(0)]],
                            device Ray* sourceRays [[buffer(0)]],
                            constant SharedData& sharedData [[buffer(1)]],
                            uint2 threadId [[thread_position_in_grid]],
                            uint2 wholeSize [[threads_per_grid]])
{
    uint rayIndex = threadId.y * wholeSize.x + threadId.x;
    Spectrum color = sourceRays[rayIndex].radiance;
    if (ACCUMULATE_IMAGE && (sharedData.frameIndex > 0))
    {
        float factor = float(sharedData.frameIndex) / float(sharedData.frameIndex + 1);
        float3 stored = image.read(threadId).xyz;
        for (uint i = 0; i < SPECTRUM_SAMPLES; ++i)
            color.values[i] = mix(color.values[i], stored[i], factor);
    }
    image.write(float4(color.values[0], color.values[1], color.values[2], 1.0), threadId);
}
