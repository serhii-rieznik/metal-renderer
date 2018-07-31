#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <ModelIO/ModelIO.h>
#import <SceneKit/SceneKit.h>
#import <simd/simd.h>
#import "Renderer.h"
#import "Raytracing.h"
#import <vector>
#import <random>

#if (COMPARISON_MODE != COMPARE_DISABLED)
#   import <OpenEXR/ImfArray.h>
#   import <OpenEXR/ImfInputFile.h>
#endif

static const NSInteger ThreadGroupSize = 16;
static const NSUInteger MaxBuffersInFlight = 3;
// static const NSString* sceneName = @"CornellBox-Water-mirror";
static const NSString* sceneName = @"CornellBox-Water-plastic";
// static const NSString* sceneName = @"CornellBox-Water";
// static const NSString* sceneName = @"cornellbox";
// static const NSString* sceneName = @"white-box";

@implementation Renderer
{
    dispatch_semaphore_t _inFlightSemaphore;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _blitPipeline;
    id<MTLComputePipelineState> _rayGenerator;
    id<MTLComputePipelineState> _intersectionHandler;
    id<MTLComputePipelineState> _lightSamplingHandler;
    id<MTLComputePipelineState> _finalHandler;
    id<MTLTexture> _image;
    id<MTLTexture> _referenceImage;
    
    id<MTLBuffer> _primaryRayBuffer; // rays
    id<MTLBuffer> _lightSamplingBuffer; // rays
    
    id<MTLBuffer> _intersectionBuffer; // intersections
    
    id<MTLBuffer> _vertexBuffer; // v, n, t
    id<MTLBuffer> _indexBuffer; // int32
    
    id<MTLBuffer> _materialBuffer; // per triangle: material / light index
    
    id<MTLBuffer> _referenceBuffer; // triangle # -> material #
    id<MTLBuffer> _lightTriangles; // emissive triangles
    
    MTKTextureLoader* _textureLoader;
    
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
    uint32_t _dispatchSizeX;
    uint32_t _dispatchSizeY;
    float _time;
    float _frameTime;
    float _averageRaysPerSecond;
    float _averageFrameTime;
    BOOL _grabImage;
}

-(nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view;
{
    self = [super init];
    if(self)
    {
        _grabImage = NO;
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
#if (MANUAL_SRGB)
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
#else
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
#endif
    view.depthStencilPixelFormat = MTLPixelFormatInvalid;
    view.preferredFramesPerSecond = 120;
    view.sampleCount = 1;
    
    _textureLoader = [[MTKTextureLoader alloc] initWithDevice:_device];
    
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

    NSError *error = NULL;
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    
    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.label = @"MyPipeline";
    pipelineStateDescriptor.sampleCount = view.sampleCount;
    pipelineStateDescriptor.vertexFunction = [defaultLibrary newFunctionWithName:@"blitVertex"];
    pipelineStateDescriptor.fragmentFunction = [defaultLibrary newFunctionWithName:@"blitFragment"];
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    pipelineStateDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
    _blitPipeline = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
    
    auto createComputeEncoder = [&](NSString* function) {
        id<MTLFunction> rayGeneratorFunction = [defaultLibrary newFunctionWithName:function];
        return [_device newComputePipelineStateWithFunction:rayGeneratorFunction error:&error];
    };
    
    _rayGenerator = createComputeEncoder(@"rayGenerator");
    _intersectionHandler = createComputeEncoder(@"intersectionHandler");
    _lightSamplingHandler = createComputeEncoder(@"lightSamplingHandler");
    _finalHandler = createComputeEncoder(@"accumulateImage");

    _commandQueue = [_device newCommandQueue];
}

template <class T>
id<MTLBuffer> createBuffer(id<MTLDevice> device, const std::vector<T>& v)
{
    return [device newBufferWithBytes:v.data() length:v.size() * sizeof(T) options:MTLResourceStorageModeManaged];
};

- (void)loadReferenceImage
{
#if (COMPARISON_MODE != COMPARE_DISABLED)
    NSString* imageName = [NSString stringWithFormat:@"Media/reference/%@-%u", sceneName, MAX_PATH_LENGTH];
    NSURL* imageUrl = [[NSBundle mainBundle] URLForResource:imageName withExtension:@"exr"];
    if (imageUrl != nil)
    {
        /*
         CGImageSourceRef imageSource = CGImageSourceCreateWithURL((__bridge CFURLRef)imageUrl, nil);
         CGImageRef image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil);
         size_t width = CGImageGetWidth(image);
         size_t height = CGImageGetHeight(image);
         size_t bpp = CGImageGetBitsPerPixel(image);
         assert(bpp == 48);
         
         size_t bitsPerComponent = CGImageGetBitsPerComponent(image);
         assert(bitsPerComponent == 16);
         
         size_t rowSize = CGImageGetBytesPerRow(image);
         assert(rowSize == width * 6);
         rowSize = width * 8;
         
         MTLPixelFormat pixelFormat = MTLPixelFormatRGBA16Float;
         
         CGDataProviderRef imageDataProvider = CGImageGetDataProvider(image);
         CFDataRef imageData = CGDataProviderCopyData(imageDataProvider);
         const uint16_t* srcData = reinterpret_cast<const uint16_t*>(CFDataGetBytePtr(imageData));
         
         #define SWAP_ORDER(A) (((A & 0x00FF) << 8) | ((A & 0xFF00) >> 8))
         
         uint16_t* rgbaData = (uint16_t*)calloc(width * height, sizeof(uint64_t));
         for (size_t y = 0; y < height; ++y)
         {
         for (size_t x = 0; x < width; ++x)
         {
         size_t i = x + y * width;
         size_t j = x + (height - y - 1) * width;
         rgbaData[4 * i + 0] = srcData[3 * j + 0];
         rgbaData[4 * i + 1] = srcData[3 * j + 1];
         rgbaData[4 * i + 2] = srcData[3 * j + 2];
         }
         }
         // */
        
        //*
        using namespace Imf_2_2;
        using namespace Imath;
        const char* pathToFile = [[imageUrl path] cStringUsingEncoding:NSUTF8StringEncoding];
        InputFile file(pathToFile);
        Box2i dw = file.header().dataWindow();
        int width  = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;
        Array2D<float> rPixels;
        Array2D<float> gPixels;
        Array2D<float> bPixels;
        rPixels.resizeErase(height, width);
        gPixels.resizeErase(height, width);
        bPixels.resizeErase(height, width);
        FrameBuffer frameBuffer;
        frameBuffer.insert("R", Slice(FLOAT, (char*)(&rPixels[0][0] - dw.min.x - dw.min.y * width), sizeof(rPixels[0][0]) * 1, sizeof(rPixels[0][0]) * width, 1, 1, 0.0f));
        frameBuffer.insert("G", Slice(FLOAT, (char*)(&gPixels[0][0] - dw.min.x - dw.min.y * width), sizeof(gPixels[0][0]) * 1, sizeof(gPixels[0][0]) * width, 1, 1, 0.0f));
        frameBuffer.insert("B", Slice(FLOAT, (char*)(&bPixels[0][0] - dw.min.x - dw.min.y * width), sizeof(bPixels[0][0]) * 1, sizeof(bPixels[0][0]) * width, 1, 1, 0.0f));
        file.setFrameBuffer(frameBuffer);
        file.readPixels(dw.min.y, dw.max.y);
        
        vector_float4* rgbaData = (vector_float4*)calloc(width * height, 4 * sizeof(float));
        int i = 0;
        for (int v = 0; v < height; ++v)
        {
            for (int u = 0; u < width; ++u)
            {
                rgbaData[i].x = rPixels[height - v - 1][u];
                rgbaData[i].y = gPixels[height - v - 1][u];
                rgbaData[i].z = bPixels[height - v - 1][u];
                rgbaData[i].w = 1.0f;
                ++i;
            }
        }
        // */
        
        size_t rowSize = width * sizeof(float) * 4;
        MTLPixelFormat pixelFormat = MTLPixelFormatRGBA32Float;
        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat width:width height:height mipmapped:NO];
        _referenceImage = [_device newTextureWithDescriptor:texDesc];
        [_referenceImage replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:rgbaData bytesPerRow:rowSize];
        
        // CFRelease(imageData);
        
        free(rgbaData);
    }
#endif
}

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
    std::vector<Material> materials;
    std::vector<TriangleReference> references;
    std::vector<LightTriangle> lightTriangles;
    
    [self loadReferenceImage];
    
    NSURL* assetUrl = [[NSBundle mainBundle] URLForResource:[NSString stringWithFormat:@"Media/%@", sceneName] withExtension:@"obj"];
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
        NSColor* specular = [[material specular] contents];

        materials.emplace_back();
        Material& mtl = materials.back();
        mtl.diffuse[0] = diffuse.redComponent;
        mtl.diffuse[1] = diffuse.greenComponent;
        mtl.diffuse[2] = diffuse.blueComponent;
        mtl.emissive[0] = emissive.redComponent;
        mtl.emissive[1] = emissive.greenComponent;
        mtl.emissive[2] = emissive.blueComponent;
        mtl.ior = specular.blueComponent;

        float roughness = specular.redComponent;
        float metallness = specular.greenComponent;
        
        if (metallness > 0.0f)
        {
            if (roughness == 0.0f)
            {
                mtl.materialType = MATERIAL_MIRROR;
            }
            else
            {
                // plastic
            }
        }
        else if (roughness == 1.0f)
        {
            mtl.materialType = MATERIAL_DIFFUSE;
        }
        else if (mtl.ior <= 0.0)
        {
            mtl.ior = std::abs(mtl.ior);
            mtl.materialType = (roughness == 0.0) ? MATERIAL_SMOOTH_PLASTIC : MATERIAL_DIFFUSE /* TODO : rough plastic */;
        }
        else
        {
            mtl.materialType = (roughness == 0.0) ? MATERIAL_SMOOTH_DIELECTRIC : MATERIAL_DIFFUSE /* TODO : rough dielectric */;
        }
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
            
            references.reserve(references.size() + triangleCount);
            indices.reserve(indices.size() + 3 * triangleCount);
            
            const uint32_t* rawData = reinterpret_cast<const uint32_t*>([[element data] bytes]);
            for (NSInteger i = 0; i < triangleCount; ++i)
            {
                uint32_t lightTriangleIndex = uint32_t(-1);
                if (isEmitter)
                {
                    const Vertex& v1 = vertices[rawData[0]];
                    const Vertex& v2 = vertices[rawData[1]];
                    const Vertex& v3 = vertices[rawData[2]];
                    vector_float3 p1 = {v1.v[0], v1.v[1], v1.v[2]};
                    vector_float3 p2 = {v2.v[0], v2.v[1], v2.v[2]};
                    vector_float3 p3 = {v3.v[0], v3.v[1], v3.v[2]};
                    lightTriangleIndex = uint32_t(lightTriangles.size());
                    lightTriangles.emplace_back();
                    lightTriangles.back().index = static_cast<int>(references.size());
                    memcpy(&(lightTriangles.back().v1), &(v1), sizeof(Vertex));
                    memcpy(&(lightTriangles.back().v2), &(v2), sizeof(Vertex));
                    memcpy(&(lightTriangles.back().v3), &(v3), sizeof(Vertex));
                    lightTriangles.back().area = 0.5f * simd_length(simd_cross(p2 - p1, p3 - p1));
                    lightTriangles.back().emissive[0] = materials[materialIndex].emissive[0];
                    lightTriangles.back().emissive[1] = materials[materialIndex].emissive[1];
                    lightTriangles.back().emissive[2] = materials[materialIndex].emissive[2];
                    totalArea += lightTriangles.back().area;
                }
                
                references.emplace_back();
                references.back().materialIndex = uint32_t(materialIndex);
                references.back().lightTriangleIndex = lightTriangleIndex;
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
        NSLog(@"(%.3f, %.3f, %.3f)", lt.area, lt.pdf, lt.cdf);
        cdf += lt.pdf;
    }
    
    _lightTrianglesCount = static_cast<uint32_t>(lightTriangles.size());
    lightTriangles.emplace_back();
    lightTriangles.back().cdf = cdf;
    lightTriangles.back().pdf = 1.0f;
    lightTriangles.back().area = 0.0f;

    _vertexBuffer = createBuffer(_device, vertices);
    _indexBuffer = createBuffer(_device, indices);
    _referenceBuffer = createBuffer(_device, references);
    _materialBuffer = createBuffer(_device, materials);
    _lightTriangles = createBuffer(_device, lightTriangles);

    _accelerationStructure = [[MPSTriangleAccelerationStructure alloc] initWithDevice:_device];
    [_accelerationStructure setVertexBuffer:_vertexBuffer];
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
    float currentTime = CACurrentMediaTime() - _startupTime;
    _frameTime = currentTime - _time;
    _time = currentTime;
    
    id<MTLBuffer> sharedDataBuffer = _perFrameData[_frameIndex % MaxBuffersInFlight].sharedData;
    SharedData* data = reinterpret_cast<SharedData*>([sharedDataBuffer contents]);
    data->frameIndex = _frameIndex;
    data->lightTrianglesCount = _lightTrianglesCount;
    data->time = _time;
    [sharedDataBuffer didModifyRange:NSMakeRange(0, sizeof(SharedData))];
    
#if (ANIMATE_NOISE)
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff) ^ (_frameIndex + 1), uint32_t(timeSeed >> 32) ^ (_frameIndex + 3)};
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

- (void)performRaytracing:(id<MTLCommandBuffer>)commandBuffer
{
    [self updateSharedData];
    
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setLabel:@"Ray generator"];
    [computeEncoder pushDebugGroup:@"Generate rays"];
    {
        [computeEncoder setBuffer:_primaryRayBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].sharedData offset:0 atIndex:1];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].noise offset:0 atIndex:2];
        [computeEncoder setComputePipelineState:_rayGenerator];
        [computeEncoder dispatchThreads:MTLSizeMake(_dispatchSizeX, _dispatchSizeY, 1) threadsPerThreadgroup:MTLSizeMake(ThreadGroupSize, ThreadGroupSize, 1)];
    }
    [computeEncoder popDebugGroup];
    [computeEncoder endEncoding];
    
    for (uint32_t i = 0; i < MAX_PATH_LENGTH; ++i)
    {
        [_intersector encodeIntersectionToCommandBuffer:commandBuffer intersectionType:MPSIntersectionTypeNearest
                                              rayBuffer:_primaryRayBuffer rayBufferOffset:0
                                     intersectionBuffer:_intersectionBuffer intersectionBufferOffset:0
                                               rayCount:_rayCount accelerationStructure:_accelerationStructure];
        
        // handle intersections and do next event estimation
        computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setLabel:@"Intersection handler"];
        {
            [computeEncoder setTexture:_image atIndex:0];
            [computeEncoder setBuffer:_intersectionBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:_vertexBuffer offset:0 atIndex:1];
            [computeEncoder setBuffer:_indexBuffer offset:0 atIndex:2];
            [computeEncoder setBuffer:_referenceBuffer offset:0 atIndex:3];
            [computeEncoder setBuffer:_materialBuffer offset:0 atIndex:4];
            [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].sharedData offset:0 atIndex:5];
            [computeEncoder setBuffer:_lightTriangles offset:0 atIndex:6];
            [computeEncoder setBuffer:_primaryRayBuffer offset:0 atIndex:7];
            [computeEncoder setBuffer:_perFrameData[(_frameIndex + i) % MaxBuffersInFlight].noise offset:0 atIndex:8];
            [computeEncoder setBuffer:_lightSamplingBuffer offset:0 atIndex:9];
            [computeEncoder setComputePipelineState:_intersectionHandler];
            [computeEncoder dispatchThreads:MTLSizeMake(_dispatchSizeX, _dispatchSizeY, 1) threadsPerThreadgroup:MTLSizeMake(ThreadGroupSize, ThreadGroupSize, 1)];
        }
        [computeEncoder endEncoding];
        
        [_intersector encodeIntersectionToCommandBuffer:commandBuffer
                                       intersectionType:MPSIntersectionTypeNearest
                                              rayBuffer:_lightSamplingBuffer
                                        rayBufferOffset:0
                                     intersectionBuffer:_intersectionBuffer
                               intersectionBufferOffset:0
                                               rayCount:_rayCount
                                  accelerationStructure:_accelerationStructure];
        
        // handle next event estimation
        computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setLabel:@"Light sampling"];
        {
            [computeEncoder setTexture:_image atIndex:0];
            [computeEncoder setBuffer:_intersectionBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:_primaryRayBuffer offset:0 atIndex:1];
            [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].sharedData offset:0 atIndex:2];
            [computeEncoder setBuffer:_referenceBuffer offset:0 atIndex:3];
            [computeEncoder setBuffer:_materialBuffer offset:0 atIndex:4];
            [computeEncoder setBuffer:_lightSamplingBuffer offset:0 atIndex:5];
            [computeEncoder setComputePipelineState:_lightSamplingHandler];
            [computeEncoder dispatchThreads:MTLSizeMake(_dispatchSizeX, _dispatchSizeY, 1) threadsPerThreadgroup:MTLSizeMake(ThreadGroupSize, ThreadGroupSize, 1)];
        }
        [computeEncoder endEncoding];
    }
    
    // final image gathering
    computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setLabel:@"Final image accumulation"];
    [computeEncoder pushDebugGroup:@"Final"];
    {
        [computeEncoder setTexture:_image atIndex:0];
        [computeEncoder setBuffer:_primaryRayBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].sharedData offset:0 atIndex:1];
        [computeEncoder setComputePipelineState:_finalHandler];
        [computeEncoder dispatchThreads:MTLSizeMake(_dispatchSizeX, _dispatchSizeY, 1) threadsPerThreadgroup:MTLSizeMake(ThreadGroupSize, ThreadGroupSize, 1)];
    }
    [computeEncoder popDebugGroup];
    [computeEncoder endEncoding];
}

- (void)drawInMTKView:(nonnull MTKView *)view
{
#if (MAX_FRAMES > 0)
    if (_frameIndex >= MAX_FRAMES) return;
#endif
    
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

    [self performRaytracing:commandBuffer];

    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder pushDebugGroup:@"Blit"];
    {
        [renderEncoder setRenderPipelineState:_blitPipeline];
        [renderEncoder setFragmentTexture:_image atIndex:0];
        [renderEncoder setFragmentTexture:(_referenceImage == nil) ? _image : _referenceImage atIndex:1];
        [renderEncoder setFragmentBuffer:_perFrameData[_frameIndex % MaxBuffersInFlight].sharedData offset:0 atIndex:0];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [renderEncoder popDebugGroup];
    
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
    
    if (_grabImage)
    {
        
    }
    
    float raysPerSecond = float(_dispatchSizeX * _dispatchSizeY) / _frameTime;
    _averageRaysPerSecond = 0.5f * (_averageRaysPerSecond + raysPerSecond);
    _averageFrameTime = 0.5f * (_averageFrameTime + _frameTime);
    
    ++_frameIndex;
    [[NSApp mainWindow] setTitle:[NSString stringWithFormat:@"Frame: %u [%0.2f Mrays/s, %.2f ms/frame]",
                                  _frameIndex, _averageRaysPerSecond / 1.0e+6, _averageFrameTime * 1000.f]];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    _dispatchSizeX = uint32_t(CONTENT_SCALE * size.width);
    _dispatchSizeY = uint32_t(CONTENT_SCALE * size.height);

    MTLTextureDescriptor* imageDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:_dispatchSizeX height:_dispatchSizeY mipmapped:NO];
    imageDescriptor.usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    imageDescriptor.storageMode = MTLStorageModeManaged;
    _image = [_device newTextureWithDescriptor:imageDescriptor];
        
    _rayCount = _dispatchSizeX * _dispatchSizeY;
    _primaryRayBuffer = [_device newBufferWithLength:_rayCount * sizeof(Ray) options:MTLResourceStorageModePrivate];
    _lightSamplingBuffer = [_device newBufferWithLength:_rayCount * sizeof(Ray) options:MTLResourceStorageModePrivate];
    _intersectionBuffer = [_device newBufferWithLength:_rayCount * sizeof(Intersection) options:MTLResourceStorageModePrivate];
    _frameIndex = 0;
    _averageRaysPerSecond = 0;
    _averageFrameTime = 0.0f;
}

- (void)saveCurrentImage
{
    _grabImage = YES;
}

@end
