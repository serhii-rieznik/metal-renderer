#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <ModelIO/ModelIO.h>
#import <SceneKit/SceneKit.h>
#import <simd/simd.h>
#import "Renderer.h"
#import "ShaderTypes.h"
#import <vector>
#import <random>

static const NSUInteger MaxBuffersInFlight = 3;

#define ANIMATE_NOISE 1

@implementation Renderer
{
    dispatch_semaphore_t _inFlightSemaphore;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _blitPipeline;
    id<MTLComputePipelineState> _rayGenerator;
    id<MTLComputePipelineState> _intersectionHandler;
    id<MTLComputePipelineState> _nextEventHandler;
    id<MTLTexture> _image;
    id<MTLBuffer> _rayBuffer;
    id<MTLBuffer> _intersectionBuffer;
    id<MTLBuffer> _geometryBuffer;
    id<MTLBuffer> _indexBuffer;
    id<MTLBuffer> _materialIndexBuffer;
    id<MTLBuffer> _materialBuffer;
    id<MTLBuffer> _lightTriangles;
    
    struct PerFrameData
    {
        id<MTLBuffer> sharedData;
        id<MTLBuffer> noise;
    } _perFrameData[MaxBuffersInFlight];
 
    CFTimeInterval _startupTime;
    MPSTriangleAccelerationStructure* _accelerationStructure;
    MPSRayIntersector* _intersector;
    uint32_t _frameIndex;
    uint32_t _lightTrianglesCount;
    uint32_t _rayCount;
}

-(nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view;
{
    self = [super init];
    if(self)
    {
        _frameIndex = 0;
        _device = view.device;
        _inFlightSemaphore = dispatch_semaphore_create(MaxBuffersInFlight);
        _startupTime = CACurrentMediaTime();
        [self loadMetalWithView:view];
        [self initRaytracing];
    }
    
    return self;
}

- (void)loadMetalWithView:(nonnull MTKView *)view;
{
    view.depthStencilPixelFormat = MTLPixelFormatInvalid;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    view.sampleCount = 1;
    
    size_t noiseBufferSize = NOISE_DIMENSIONS * NOISE_DIMENSIONS * sizeof(vector_float4);
    for (uint32_t i = 0; i < MaxBuffersInFlight; ++i)
    {
        _perFrameData[i].sharedData = [_device newBufferWithLength:sizeof(SharedData) options:MTLResourceStorageModeManaged];
        _perFrameData[i].noise = [_device newBufferWithLength:noiseBufferSize options:MTLResourceStorageModeManaged];
    }
    
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    std::mt19937_64 rng;
    rng.seed(ss);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    float* noiseData[3] = {
        reinterpret_cast<float*>([_perFrameData[0].noise contents]),
        reinterpret_cast<float*>([_perFrameData[1].noise contents]),
        reinterpret_cast<float*>([_perFrameData[2].noise contents]) };
    
    for (NSUInteger i = 0, e = [_perFrameData[0].noise length] / sizeof(float); i < e; ++i)
    {
        float t = distribution(rng);
        *noiseData[0]++ = t; // distribution(rng);
        *noiseData[1]++ = t; // distribution(rng);
        *noiseData[2]++ = t; // distribution(rng);
    }
    [_perFrameData[0].noise didModifyRange:NSMakeRange(0, [_perFrameData[0].noise length])];
    [_perFrameData[1].noise didModifyRange:NSMakeRange(0, [_perFrameData[1].noise length])];
    [_perFrameData[2].noise didModifyRange:NSMakeRange(0, [_perFrameData[2].noise length])];

    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    
    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.label = @"MyPipeline";
    pipelineStateDescriptor.sampleCount = view.sampleCount;
    pipelineStateDescriptor.vertexFunction = [defaultLibrary newFunctionWithName:@"blitVertex"];
    pipelineStateDescriptor.fragmentFunction = [defaultLibrary newFunctionWithName:@"blitFragment"];
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    pipelineStateDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
    
    NSError *error = NULL;
    _blitPipeline = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
    if (_blitPipeline == nil)
    {
        NSLog(@"Failed to created blit pipeline state, error %@", error);
    }
    
    id<MTLFunction> rayGeneratorFunction = [defaultLibrary newFunctionWithName:@"rayGenerator"];
    _rayGenerator = [_device newComputePipelineStateWithFunction:rayGeneratorFunction error:&error];
    
    id<MTLFunction> intersectionHandlerFunction = [defaultLibrary newFunctionWithName:@"intersectionHandler"];
    _intersectionHandler = [_device newComputePipelineStateWithFunction:intersectionHandlerFunction error:&error];

    id<MTLFunction> nextEventHandlerFunction = [defaultLibrary newFunctionWithName:@"nextEventEstimationHandler"];
    _nextEventHandler = [_device newComputePipelineStateWithFunction:nextEventHandlerFunction error:&error];

    _commandQueue = [_device newCommandQueue];
}

template <class T>
id<MTLBuffer> createBuffer(id<MTLDevice> device, const std::vector<T>& v)
{
    return [device newBufferWithBytes:v.data() length:v.size() * sizeof(T) options:MTLResourceStorageModeManaged];
};

- (void)initRaytracing
{
    struct Vertex
    {
        float v[3];
        float n[3];
        float t[2];
    };
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> materialIndices;
    std::vector<Material> materials;
    std::vector<LightTriangle> lightTriangles;
    
    NSURL* assetUrl = [[NSBundle mainBundle] URLForResource:@"Media/CornellBox-Water" withExtension:@"obj"];
    SCNScene* scene = [SCNScene sceneWithURL:assetUrl options:nil error:nil];
    SCNNode* rootGeometryNode = [[[scene rootNode] childNodes] objectAtIndex:0]; // hopefully it is fine ^_^
    SCNGeometry* geometry = [rootGeometryNode geometry];
    
    NSInteger vertexCount = 0;
    const uint8_t* vertexData = nullptr;
    const uint8_t* normalData = nullptr;
    const uint8_t* texCoordData = nullptr;
    NSInteger vertexStride = 0;
    NSInteger normalStride = 0;
    NSInteger texCoordStride = 0;
    
    for (SCNMaterial* material in [geometry materials])
    {
        NSColor* diffuse = [[material diffuse] contents];
        NSColor* emissive = [[material emission] contents];

        materials.emplace_back();
        Material& mtl = materials.back();
        mtl.diffuse[0] = diffuse.redComponent;
        mtl.diffuse[1] = diffuse.greenComponent;
        mtl.diffuse[2] = diffuse.blueComponent;
        mtl.emissive[0] = emissive.redComponent;
        mtl.emissive[1] = emissive.greenComponent;
        mtl.emissive[2] = emissive.blueComponent;
    }

    for (SCNGeometrySource* source in [geometry geometrySources])
    {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>([[source data] bytes]) + [source dataOffset];
        if ([source semantic] == SCNGeometrySourceSemanticVertex)
        {
            vertexCount = [source vectorCount];
            vertexStride = [source dataStride];
            vertexData = ptr;
        }
        else if ([source semantic] == SCNGeometrySourceSemanticNormal)
        {
            normalStride = [source dataStride];
            normalData = ptr;
        }
        else if ([source semantic] == SCNGeometrySourceSemanticTexcoord)
        {
            texCoordStride = [source dataStride];
            texCoordData = ptr;
        }
    }

    vertices.resize(vertexCount);
    for (NSInteger i = 0; i < vertexCount; ++i)
    {
        Vertex& vert = vertices[i];
        {
            const vector_float3* ptr = reinterpret_cast<const vector_float3*>(vertexData + i * vertexStride);
            memcpy(vert.v, ptr, sizeof(float) * 3);
        }
        if (normalData)
        {
            const vector_float3* ptr = reinterpret_cast<const vector_float3*>(normalData + i * normalStride);
            memcpy(vert.n, ptr, sizeof(float) * 3);
        }
        
        if (texCoordData)
        {
            const vector_float2* ptr = reinterpret_cast<const vector_float2*>(texCoordData + i * texCoordStride);
            memcpy(vert.t, ptr, sizeof(float) * 2);
        }
    }

    // The index of the material used for a geometry element is equal to the index of that element modulo the number of materials.
    float totalArea = 0.0f;
    NSInteger elementIndex = 0;
    for (SCNGeometryElement* element in [geometry geometryElements])
    {
        NSInteger materialIndex = elementIndex % [[geometry materials] count];
        bool isEmitter =
            (materials[materialIndex].emissive[0] > 0.0f) ||
            (materials[materialIndex].emissive[1] > 0.0f) ||
            (materials[materialIndex].emissive[2] > 0.0f);
        
        if (([element bytesPerIndex] == 4) && ([element primitiveType] == SCNGeometryPrimitiveTypeTriangles))
        {
            NSInteger triangleCount = [element primitiveCount];
            
            materialIndices.reserve(materialIndices.size() + triangleCount);
            indices.reserve(indices.size() + 3 * triangleCount);
            
            const uint32_t* rawData = reinterpret_cast<const uint32_t*>([[element data] bytes]);
            for (NSInteger i = 0; i < triangleCount; ++i)
            {
                if (isEmitter)
                {
                    const Vertex& v1 = vertices[rawData[0]];
                    const Vertex& v2 = vertices[rawData[1]];
                    const Vertex& v3 = vertices[rawData[2]];
                    vector_float3 p1 = {v1.v[0], v1.v[1], v1.v[2]};
                    vector_float3 p2 = {v2.v[0], v2.v[1], v2.v[2]};
                    vector_float3 p3 = {v3.v[0], v3.v[1], v3.v[2]};
                    lightTriangles.emplace_back();
                    lightTriangles.back().index = static_cast<int>(materialIndices.size());
                    lightTriangles.back().area = 0.5f * simd_length(simd_cross(p2 - p1, p3 - p1));
                    totalArea += lightTriangles.back().area;
                }
                
                materialIndices.emplace_back(materialIndex);
                indices.emplace_back(*rawData++);
                indices.emplace_back(*rawData++);
                indices.emplace_back(*rawData++);
            }
        }
        else
        {
            NSLog(@"%@ : not implemented", element);
        }
        
        ++elementIndex;
    }
    
    NSLog(@"Light triangles");
    float cdf = 0.0f;
    for (LightTriangle& lt : lightTriangles)
    {
        lt.pdf = lt.area / totalArea;
        lt.cdf = cdf;
        NSLog(@"(%u, %.3f, %.3f, %.3f)", lt.index, lt.area, lt.pdf, lt.cdf);
        cdf += lt.pdf;
    }
    _lightTrianglesCount = static_cast<uint32_t>(lightTriangles.size());
    lightTriangles.emplace_back();
    lightTriangles.back().cdf = cdf;
    lightTriangles.back().pdf = 1.0f;
    lightTriangles.back().area = 0.0f;

    _geometryBuffer = createBuffer(_device, vertices);
    _indexBuffer = createBuffer(_device, indices);
    _materialIndexBuffer = createBuffer(_device, materialIndices);
    _materialBuffer = createBuffer(_device, materials);
    _lightTriangles = createBuffer(_device, lightTriangles);

    _accelerationStructure = [[MPSTriangleAccelerationStructure alloc] initWithDevice:_device];
    [_accelerationStructure setVertexBuffer:_geometryBuffer];
    [_accelerationStructure setIndexBuffer:_indexBuffer];
    [_accelerationStructure setVertexStride:sizeof(Vertex)];
    [_accelerationStructure setTriangleCount:indices.size() / 3];
    [_accelerationStructure setIndexType:MPSDataTypeUInt32];
    [_accelerationStructure rebuild];
    
    _intersector = [[MPSRayIntersector alloc] initWithDevice:_device];
    [_intersector setCullMode:MTLCullModeNone];
    [_intersector setRayStride:sizeof(Ray)];
    [_intersector setRayDataType:MPSRayDataTypeOriginMinDistanceDirectionMaxDistance];
    [_intersector setRayMaskOptions:MPSRayMaskOptionNone];
    [_intersector setIntersectionDataType:MPSIntersectionDataTypeDistancePrimitiveIndexCoordinates];
}

- (void)updateSharedData
{
    id<MTLBuffer> sharedDataBuffer = _perFrameData[_frameIndex % MaxBuffersInFlight].sharedData;
    SharedData* data = reinterpret_cast<SharedData*>([sharedDataBuffer contents]);
    data->frameIndex = _frameIndex;
    data->lightTrianglesCount = _lightTrianglesCount;
    data->time = CACurrentMediaTime() - _startupTime;
    [sharedDataBuffer didModifyRange:NSMakeRange(0, sizeof(SharedData))];
    
#if (ANIMATE_NOISE)
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    std::mt19937_64 rng;
    rng.seed(ss);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    id<MTLBuffer> noiseDataBuffer = _perFrameData[_frameIndex % MaxBuffersInFlight].noise;
    float* noiseData = reinterpret_cast<float*>([noiseDataBuffer contents]);
    for (NSUInteger i = 0, e = [noiseDataBuffer length] / sizeof(float); i < e; ++i)
        *noiseData++ = distribution(rng);
    [noiseDataBuffer didModifyRange:NSMakeRange(0, [noiseDataBuffer length])];
#endif
}

- (void)drawInMTKView:(nonnull MTKView *)view
{
    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        dispatch_semaphore_signal(block_sema);
    }];

    [self updateSharedData];

    MTLRenderPassDescriptor* renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor == nil)
    {
        [commandBuffer commit];
        return;
    }
    
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setLabel:@"Ray generator"];
    [computeEncoder pushDebugGroup:@"Generate rays"];
    {
        [computeEncoder setBuffer:_rayBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].sharedData offset:0 atIndex:1];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].noise offset:0 atIndex:2];
        [computeEncoder setComputePipelineState:_rayGenerator];
        [computeEncoder dispatchThreads:MTLSizeMake(view.drawableSize.width, view.drawableSize.height, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }
    [computeEncoder popDebugGroup];
    [computeEncoder endEncoding];
    
    [_intersector encodeIntersectionToCommandBuffer:commandBuffer
                                   intersectionType:MPSIntersectionTypeNearest
                                          rayBuffer:_rayBuffer
                                    rayBufferOffset:0
                                 intersectionBuffer:_intersectionBuffer
                           intersectionBufferOffset:0
                                           rayCount:_rayCount
                              accelerationStructure:_accelerationStructure];
    
    // handle intersections and do next event estimation
    computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setLabel:@"Intersection handler"];
    [computeEncoder pushDebugGroup:@"Handle intersection"];
    {
        [computeEncoder setTexture:_image atIndex:0];
        [computeEncoder setBuffer:_intersectionBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_geometryBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:_indexBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:_materialIndexBuffer offset:0 atIndex:3];
        [computeEncoder setBuffer:_materialBuffer offset:0 atIndex:4];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].sharedData offset:0 atIndex:5];
        [computeEncoder setBuffer:_lightTriangles offset:0 atIndex:6];
        [computeEncoder setBuffer:_rayBuffer offset:0 atIndex:7];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].noise offset:0 atIndex:8];
        [computeEncoder setComputePipelineState:_intersectionHandler];
        [computeEncoder dispatchThreads:MTLSizeMake(view.drawableSize.width, view.drawableSize.height, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }
    [computeEncoder popDebugGroup];
    [computeEncoder endEncoding];
    
    // next event estimation
    [_intersector encodeIntersectionToCommandBuffer:commandBuffer
                                   intersectionType:MPSIntersectionTypeNearest
                                          rayBuffer:_rayBuffer
                                    rayBufferOffset:0
                                 intersectionBuffer:_intersectionBuffer
                           intersectionBufferOffset:0
                                           rayCount:_rayCount
                              accelerationStructure:_accelerationStructure];
    
    // handle next event estimation
    computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setLabel:@"Next event estimation"];
    [computeEncoder pushDebugGroup:@"Next event estimation"];
    {
        [computeEncoder setTexture:_image atIndex:0];
        [computeEncoder setBuffer:_intersectionBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_rayBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].sharedData offset:0 atIndex:2];
        [computeEncoder setBuffer:_materialIndexBuffer offset:0 atIndex:3];
        [computeEncoder setBuffer:_materialBuffer offset:0 atIndex:4];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].noise offset:0 atIndex:5];
        /*
        [computeEncoder setBuffer:_geometryBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:_indexBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:_lightTriangles offset:0 atIndex:6];
        // */
        [computeEncoder setComputePipelineState:_nextEventHandler];
        [computeEncoder dispatchThreads:MTLSizeMake(view.drawableSize.width, view.drawableSize.height, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }
    [computeEncoder popDebugGroup];
    [computeEncoder endEncoding];

    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder pushDebugGroup:@"Blit"];
    {
        [renderEncoder setRenderPipelineState:_blitPipeline];
        [renderEncoder setFragmentTexture:_image atIndex:0];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [renderEncoder popDebugGroup];
    
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
    
    ++_frameIndex;
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    MTLTextureDescriptor* imageDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:size.width height:size.height mipmapped:NO];
    imageDescriptor.usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    _image = [_device newTextureWithDescriptor:imageDescriptor];
    
    _rayCount = (uint32_t)(size.width) * (uint32_t)(size.height);
    _rayBuffer = [_device newBufferWithLength:_rayCount * sizeof(Ray) options:MTLResourceStorageModePrivate];
    _intersectionBuffer = [_device newBufferWithLength:_rayCount * sizeof(Intersection) options:MTLResourceStorageModePrivate];
    _frameIndex = 0;

}

@end
