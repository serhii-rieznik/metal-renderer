#pragma once

#include <simd/simd.h>

#ifdef __METAL_VERSION__

float fresnel(thread const float3& n, thread const float3& i, float etaOut, float etaIn)
{
    float result = 1.0;
    float etaScale = etaOut / etaIn;
    float cosThetaI = clamp(dot(n, i), -1.0f, 1.0f);
    float sinThetaTSquared = (etaScale * etaScale) * (1.0 - cosThetaI * cosThetaI);
    if (sinThetaTSquared < 1.0)
    {
        float cosThetaT = sqrt(1.0 - sinThetaTSquared);
        float rS = (etaIn * cosThetaI - etaOut * cosThetaT) / (etaIn * cosThetaI + etaOut * cosThetaT);
        float rP = (etaIn * cosThetaT - etaOut * cosThetaI) / (etaIn * cosThetaT + etaOut * cosThetaI);
        result = 0.5 * (rS * rS + rP * rP);
    }
    return result;
}

Vertex interpolate(device const Vertex& t0, device const Vertex& t1, device const Vertex& t2, float3 uvw)
{
    Vertex result;
    result.v = t0.v * uvw.x + t1.v * uvw.y + t2.v * uvw.z;
    result.n = normalize(t0.n * uvw.x + t1.n * uvw.y + t2.n * uvw.z);
    return result;
}

Vertex interpolate(thread const Vertex& t0, thread const Vertex& t1, thread const Vertex& t2, float3 uvw)
{
    Vertex result;
    result.v = t0.v * uvw.x + t1.v * uvw.y + t2.v * uvw.z;
    result.n = normalize(t0.n * uvw.x + t1.n * uvw.y + t2.n * uvw.z);
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

device const LightTriangle& selectLightTriangle(float xi, device const LightTriangle* lightTriangles, int trianglesCount)
{
    int index = 0;
    for (; (index < trianglesCount) && (lightTriangles[index + 1].cdf <= xi); ++index);
    return lightTriangles[index];
}

float2 sampleMaterial(device const Material& material, thread const float3& wI, thread const float3& wO,
                      thread const float3& n, thread const float4& noiseSample)
{
    float2 result;
    
    float cosTheta = dot(wO, n);
    
    switch (material.materialType)
    {
        case MATERIAL_MIRROR:
        {
            bool isMirrorDirection = abs(dot(reflect(wI, n), wO) - 1.0) < ANGLE_EPSILON;
            result = isMirrorDirection ? float2(cosTheta, 1.0) : float2(0.0, 1.0);
            break;
        }
            
        case MATERIAL_SMOOTH_PLASTIC:
        {
            float fI = fresnel(n, -wI, 1.0, material.ior);
            if (fI < noiseSample.y)
            {
                // diffuse reflectance
                result = (1.0 / PI) * cosTheta;
            }
            else
            {
                // specular reflectance
                bool isMirrorDirection = abs(dot(reflect(wI, n), wO) - 1.0) < ANGLE_EPSILON;
                result = isMirrorDirection ? float2(cosTheta, 1.0) : float2(0.0, 1.0);
            }
            break;
        }
            
        case MATERIAL_SMOOTH_DIELECTRIC:
        {
            float fI = fresnel(n, -wI, 1.0, material.ior);
            if (fI < noiseSample.y)
            {
                // transmittance
                result = 0.0; // (1.0 / PI) * cosTheta;
            }
            else
            {
                // specular reflectance
                bool isMirrorDirection = abs(dot(reflect(wI, n), wO) - 1.0) < ANGLE_EPSILON;
                result = isMirrorDirection ? float2(cosTheta, 1.0) : float2(0.0, 1.0);
            }
            break;
        }
            
        default: // MATERIAL_DIFFUSE
        {
            result = (1.0 / PI) * cosTheta;
            break;
        }
    }
    
    return result;
}

float3 generateNextBounce(device const Material& material, device const Ray& rI, thread const float3& n,
                          thread const float4& noiseSample, thread float& bsdf, thread float& pdf, thread float& ior)
{
    float3 wI = rI.direction;
    float3 wO;
    float2 properties;
    
    float currentIoR = rI.params.w;
    ior = currentIoR;
    
    switch (material.materialType)
    {
        case MATERIAL_MIRROR:
        {
            wO = reflect(wI, n);
            properties = float2(dot(wO, n), 1.0);
            break;
        }

        case MATERIAL_SMOOTH_PLASTIC:
        {
            float fI = fresnel(n, -wI, currentIoR, material.ior);
            if (fI < noiseSample.y)
            {
                wO = generateDiffuseBounce(noiseSample.zw, n);
                properties = (1.0 / PI) * dot(wO, n);
            }
            else
            {
                wO = reflect(wI, n);
                properties = float2(dot(wO, n), 1.0);
            }
            break;
        }
            
        case MATERIAL_SMOOTH_DIELECTRIC:
        {
            float fI = fresnel(n, -wI, currentIoR, material.ior);
            if (fI < noiseSample.y)
            {
                ior = material.ior;
                wO = wI;// generateDiffuseBounce(noiseSample.zw, n);
                properties = 1.0; // (1.0 / PI) * dot(wO, n);
            }
            else
            {
                wO = reflect(wI, n);
                properties = float2(dot(wO, n), 1.0);
            }
            break;
        }
            
        default: // MATERIAL_DIFFUSE
        {
            wO = generateDiffuseBounce(noiseSample.zw, n);
            properties = (1.0 / PI) * dot(wO, n);
        }
    }
    
    bsdf = properties.x;
    pdf = properties.y;
    
    return wO;
}

float lightTriangleSamplePDF(float trianglePdf, float triangleArea, thread const float3& source, thread const Vertex& sample, thread packed_float3& directionToLight)
{
    directionToLight = sample.v - source;
    
    float distanceToLight = length(directionToLight);
    directionToLight = normalize(directionToLight);
    float LdotD = -dot(directionToLight.xyz, sample.n.xyz);
    float validDirection = float(distanceToLight >= DISTANCE_EPSILON) * float(LdotD >= ANGLE_EPSILON);
    return validDirection * trianglePdf * triangleSamplePDF(triangleArea, LdotD, distanceToLight);
}

#endif

