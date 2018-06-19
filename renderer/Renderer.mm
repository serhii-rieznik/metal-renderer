#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <ModelIO/ModelIO.h>
#import <SceneKit/SceneKit.h>
#import <simd/simd.h>
#import "Renderer.h"
#import "ShaderTypes.h"
#import <vector>

static const NSUInteger MaxBuffersInFlight = 3;

@implementation Renderer
{
    dispatch_semaphore_t _inFlightSemaphore;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _blitPipeline;
    id<MTLComputePipelineState> _rayGenerator;
    id<MTLComputePipelineState> _intersectionHandler;
    id<MTLTexture> _image;
    id<MTLBuffer> _rayBuffer;
    id<MTLBuffer> _intersectionBuffer;
    id<MTLBuffer> _geometryBuffer;
    id<MTLBuffer> _indexBuffer;
    id<MTLBuffer> _materialIndexBuffer;
    id<MTLBuffer> _materialBuffer;
    id<MTLBuffer> _sharedData[MaxBuffersInFlight];
 
    CFTimeInterval _startupTime;
    MPSTriangleAccelerationStructure* _accelerationStructure;
    MPSRayIntersector* _intersector;
    uint32_t _frameIndex;
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
        [self _loadMetalWithView:view];
        [self _initRaytracing];
    }
    
    return self;
}

- (void)_loadMetalWithView:(nonnull MTKView *)view;
{
    view.depthStencilPixelFormat = MTLPixelFormatInvalid;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    view.sampleCount = 1;
    
    for (uint32_t i = 0; i < MaxBuffersInFlight; ++i)
        _sharedData[i] = [_device newBufferWithLength:sizeof(_sharedData) options:MTLResourceStorageModeManaged];
    
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

    _commandQueue = [_device newCommandQueue];
}

- (void)_initRaytracing
{
    struct Vertex
    {
        float v[3];
        float n[3];
        float t[2];
    };
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> masks;
    std::vector<Material> materials;
    
    NSURL* assetUrl = [[NSBundle mainBundle] URLForResource:@"Media/cornellbox" withExtension:@"obj"];
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
    NSInteger elementIndex = 0;
    for (SCNGeometryElement* element in [geometry geometryElements])
    {
        NSInteger materialIndex = elementIndex % [[geometry materials] count];
        
        if (([element bytesPerIndex] == 4) && ([element primitiveType] == SCNGeometryPrimitiveTypeTriangles))
        {
            NSInteger triangleCount = [element primitiveCount];
            
            masks.reserve(masks.size() + triangleCount);
            indices.reserve(indices.size() + 3 * triangleCount);
            
            const uint32_t* rawData = reinterpret_cast<const uint32_t*>([[element data] bytes]);
            for (NSInteger i = 0; i < triangleCount; ++i)
            {
                masks.emplace_back(materialIndex);
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

    _geometryBuffer = [_device newBufferWithBytes:vertices.data() length:vertices.size() * sizeof(Vertex) options:MTLResourceStorageModeManaged];
    _indexBuffer = [_device newBufferWithBytes:indices.data() length:indices.size() * sizeof(uint32_t) options:MTLResourceStorageModeManaged];
    _materialIndexBuffer = [_device newBufferWithBytes:masks.data() length:masks.size() * sizeof(uint32_t) options:MTLResourceStorageModeManaged];
    _materialBuffer = [_device newBufferWithBytes:materials.data() length:materials.size() * sizeof(Material) options:MTLResourceStorageModeManaged];

    _accelerationStructure = [[MPSTriangleAccelerationStructure alloc] initWithDevice:_device];
    [_accelerationStructure setVertexBuffer:_geometryBuffer];
    [_accelerationStructure setIndexBuffer:_indexBuffer];
    [_accelerationStructure setVertexStride:sizeof(Vertex)];
    [_accelerationStructure setTriangleCount:indices.size() / 3];
    [_accelerationStructure setIndexType:MPSDataTypeUInt32];
    [_accelerationStructure rebuild];
    
    _intersector = [[MPSRayIntersector alloc] initWithDevice:_device];
    [_intersector setCullMode:MTLCullModeNone];
    [_intersector setRayDataType:MPSRayDataTypeOriginMinDistanceDirectionMaxDistance];
    [_intersector setRayMaskOptions:MPSRayMaskOptionNone];
    [_intersector setIntersectionDataType:MPSIntersectionDataTypeDistancePrimitiveIndexCoordinates];
}

- (void)_updateSharedData
{
    SharedData* data = reinterpret_cast<SharedData*>([_sharedData[_frameIndex] contents]);
    data->frameIndex = _frameIndex;
    data->time = CACurrentMediaTime() - _startupTime;
    [_sharedData[_frameIndex] didModifyRange:NSMakeRange(0, sizeof(SharedData))];
}

- (void)drawInMTKView:(nonnull MTKView *)view
{
    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        dispatch_semaphore_signal(block_sema);
    }];

    [self _updateSharedData];

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
        [computeEncoder setBuffer:_sharedData[_frameIndex] offset:0 atIndex:1];
        [computeEncoder setComputePipelineState:_rayGenerator];
        [computeEncoder dispatchThreads:MTLSizeMake(view.drawableSize.width, view.drawableSize.height, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }
    [computeEncoder popDebugGroup];
    [computeEncoder endEncoding];
    
    uint32_t rayCount = (uint32_t)(view.drawableSize.width) * (uint32_t)(view.drawableSize.height);
    
    [_intersector encodeIntersectionToCommandBuffer:commandBuffer
                                   intersectionType:MPSIntersectionTypeNearest
                                          rayBuffer:_rayBuffer
                                    rayBufferOffset:0
                                 intersectionBuffer:_intersectionBuffer
                           intersectionBufferOffset:0
                                           rayCount:rayCount
                              accelerationStructure:_accelerationStructure];
    
    // handle intersections
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
        [computeEncoder setBuffer:_sharedData[_frameIndex] offset:0 atIndex:5];
        [computeEncoder setComputePipelineState:_intersectionHandler];
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
    
    _frameIndex = (_frameIndex + 1) % MaxBuffersInFlight;
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    MTLTextureDescriptor* imageDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:size.width height:size.height mipmapped:NO];
    imageDescriptor.usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    _image = [_device newTextureWithDescriptor:imageDescriptor];
    
    uint32_t totalPixelsCount = (uint32_t)(size.width) * (uint32_t)(size.height);
    _rayBuffer = [_device newBufferWithLength:totalPixelsCount * sizeof(Ray) options:MTLResourceStorageModePrivate];
    _intersectionBuffer = [_device newBufferWithLength:totalPixelsCount * sizeof(Intersection) options:MTLResourceStorageModePrivate];

}

@end
