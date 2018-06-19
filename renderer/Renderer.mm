//
//  Renderer.m
//  metal-rt Shared
//
//  Created by Sergey Reznik on 6/18/18.
//  Copyright © 2018 Serhii Rieznik. All rights reserved.
//

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <ModelIO/ModelIO.h>
#import "Renderer.h"
#import "ShaderTypes.h"

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
 
    MPSTriangleAccelerationStructure* _accelerationStructure;
    MPSRayIntersector* _intersector;
}

-(nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view;
{
    self = [super init];
    if(self)
    {
        _device = view.device;
        _inFlightSemaphore = dispatch_semaphore_create(MaxBuffersInFlight);
        [self _loadMetalWithView:view];
        [self _initRaytracing];
    }
    
    return self;
}

- (void)_loadMetalWithView:(nonnull MTKView *)view;
{
    view.depthStencilPixelFormat = MTLPixelFormatInvalid;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.sampleCount = 1;
    
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
    vector_float4 vertices[] = {
        {-1.0, -1.0f, 0.0f, 1.0f},
        { 1.0, -1.0f, 0.0f, 1.0f},
        {-1.0,  1.0f, 0.0f, 1.0f},
        { 1.0,  1.0f, 0.0f, 1.0f},
    };
    
    uint32_t indices[] = {
        0, 1, 2,
        1, 3, 2
    };
    
    uint32_t triangleCount = (sizeof(indices) / sizeof(indices[6])) / 3;
    
    id<MTLBuffer> vertexBuffer = [_device newBufferWithBytes:vertices length:sizeof(vertices) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> indexBuffer = [_device newBufferWithBytes:indices length:sizeof(indices) options:MTLResourceStorageModeManaged];
    
    _accelerationStructure = [[MPSTriangleAccelerationStructure alloc] initWithDevice:_device];
    [_accelerationStructure setVertexBuffer:vertexBuffer];
    [_accelerationStructure setIndexBuffer:indexBuffer];
    [_accelerationStructure setVertexStride:sizeof(simd_float4)];
    [_accelerationStructure setTriangleCount:triangleCount];
    [_accelerationStructure rebuild];
    
    _intersector = [[MPSRayIntersector alloc] initWithDevice:_device];
    [_intersector setCullMode:MTLCullModeNone];
    [_intersector setRayDataType:MPSRayDataTypeOriginMinDistanceDirectionMaxDistance];
    [_intersector setRayMaskOptions:MPSRayMaskOptionNone];
    [_intersector setIntersectionDataType:MPSIntersectionDataTypeDistancePrimitiveIndexCoordinates];
}

- (void)drawInMTKView:(nonnull MTKView *)view
{
    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        dispatch_semaphore_signal(block_sema);
    }];
    
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
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    MTLTextureDescriptor* imageDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:size.width height:size.height mipmapped:NO];
    imageDescriptor.usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    _image = [_device newTextureWithDescriptor:imageDescriptor];
    
    uint32_t totalPixelsCount = (uint32_t)(size.width) * (uint32_t)(size.height);
    _rayBuffer = [_device newBufferWithLength:totalPixelsCount * sizeof(Ray) options:MTLResourceStorageModePrivate];
    _intersectionBuffer = [_device newBufferWithLength:totalPixelsCount * sizeof(Intersection) options:MTLResourceStorageModePrivate];

}

@end
