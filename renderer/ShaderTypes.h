//
//  ShaderTypes.h
//  metal-rt Shared
//
//  Created by Sergey Reznik on 6/18/18.
//  Copyright © 2018 Serhii Rieznik. All rights reserved.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#pragma once

#ifdef __METAL_VERSION__
#else
#   define packed_float2 vector_float2
#   define packed_float3 vector_float3
#endif

#include <simd/simd.h>

struct Ray
{
    packed_float3 origin;
    float minDistance;
    packed_float3 direction;
    float maxDistance;
};

struct Intersection
{
    float distance;
    int index;
    packed_float2 coordinates;
};
