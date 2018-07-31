#pragma once

#include <simd/simd.h>

#ifdef __METAL_VERSION__

float fresnel(thread const float3& n, thread const float3& wI, float etaIn, float etaOut)
{
    float result = 1.0;
    float cosThetaI = clamp(dot(n, wI), -1.0f, 1.0f);
    float scale = etaIn / etaOut;
    float cosThetaTSquared = 1.0 - (1.0 - cosThetaI * cosThetaI) * (scale * scale);
    if (cosThetaTSquared > 0.0)
    {
        cosThetaI = abs(cosThetaI);
        float cosThetaT = sqrt(cosThetaTSquared);
        float rS = (etaOut * cosThetaI - etaIn * cosThetaT) / (etaOut * cosThetaI - etaIn * cosThetaT);
        float rP = (etaIn * cosThetaI - etaOut * cosThetaT) / (etaIn * cosThetaI + etaOut * cosThetaT);
        result = 0.5 * (rS * rS + rP * rP);
    }
    return result;
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

Vertex interpolate(thread const Vertex& t0, thread const Vertex& t1, thread const Vertex& t2, float3 uvw)
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

Vertex interpolate(thread const Vertex& t0, thread const Vertex& t1, thread const Vertex& t2, float2 uv)
{
    return interpolate(t0, t1, t2, float3(uv, 1.0 - uv.x - uv.y));
}

LightTriangle selectLightTriangle(float xi, device const LightTriangle* lightTriangles, int trianglesCount)
{
    int index = 0;
    for (; (index < trianglesCount) && (lightTriangles[index + 1].cdf <= xi); ++index);
    return lightTriangles[index];
}

float2 sampleConcreteMaterialType(uint materialType, thread const float3& wI, thread const float3& wO, thread const float3& n)
{
    float cosTheta = dot(wO, n);
    
    switch (materialType)
    {
        case MATERIAL_MIRROR:
        {
            bool isMirrorDirection = abs(dot(reflect(wI, n), wO) - 1.0) < ANGLE_EPSILON;
            return isMirrorDirection ? float2(cosTheta, 1.0) : float2(0.0, 1.0);
        }
            
        case MATERIAL_DIFFUSE:
            return (1.0 / PI) * cosTheta;
            
        default:
            return float2(0.0, 1.0);
    }
}

float3 generateConcreteNextBounce(uint materialType, thread const float3& wI, thread const float3& n,
                          device const float4& noiseSample, thread float& bsdf, thread float& pdf)
{
    float3 wO = 0.0;
    
    switch (materialType)
    {
        case MATERIAL_MIRROR:
        {
            wO = reflect(wI, n);
            break;
        }
            
        default:
        {
            wO = generateDiffuseBounce(noiseSample.zw, n);
            break;
        }
    }
    
    float2 smp = sampleConcreteMaterialType(materialType, wI, wO, n);
    bsdf = smp.x;
    pdf = smp.y;

    return wO;
}

float2 sampleMaterial(device const Material& material, thread const float3& wI, thread const float3& wO, thread const float3& n,
                      device const float4& noiseSample)
{
    switch (material.materialType)
    {
        case MATERIAL_SMOOTH_PLASTIC:
        {
            float2 smp = 0.0;
            float f = fresnel(n, -wI, 1.0, material.ior);
            if (f < noiseSample.y)
            {
                smp = sampleConcreteMaterialType(MATERIAL_MIRROR, wI, wO, n);
                // smp.y = f;
            }
            else
            {
                smp = sampleConcreteMaterialType(MATERIAL_DIFFUSE, wI, wO, n);
                // smp.x *= 1.0 - f;
                // smp.y *= 1.0 - f;
            }
            return smp;
        }
            
        default:
            return sampleConcreteMaterialType(material.materialType, wI, wO, n);
    }
}

float3 generateNextBounce(device const Material& material, thread const float3& wI, thread const float3& n,
                          device const float4& noiseSample, thread float& bsdf, thread float& pdf)
{
    switch (material.materialType)
    {
        case MATERIAL_SMOOTH_PLASTIC:
        {
            float3 bounce;
            float f = fresnel(n, -wI, 1.0, material.ior);
            if (f < noiseSample.y)
            {
                bounce = generateConcreteNextBounce(MATERIAL_MIRROR, wI, n, noiseSample, bsdf, pdf);
            }
            else
            {
                bounce = generateConcreteNextBounce(MATERIAL_DIFFUSE, wI, n, noiseSample, bsdf, pdf);
            }
            return bounce;
        }
            
        default:
            return generateConcreteNextBounce(material.materialType, wI, n, noiseSample, bsdf, pdf);
    }
}

float lightTriangleSamplePDF(float trianglePdf, float triangleArea, thread const float3& source, thread const Vertex& sample,
                             thread packed_float3& directionToLight)
{
    directionToLight = sample.v - source;
    
    float distanceToLight = length(directionToLight);
    directionToLight = normalize(directionToLight);
    float LdotD = -dot(directionToLight, sample.n);
    bool validDirection = (distanceToLight >= DISTANCE_EPSILON) && (LdotD >= ANGLE_EPSILON);
    
    return validDirection ? trianglePdf * triangleSamplePDF(triangleArea, LdotD, distanceToLight) : 0.0f;
}

#endif

