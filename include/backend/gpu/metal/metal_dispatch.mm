#include "backend/gpu/metal/metal_dispatch.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#import <Foundation/Foundation.h>
#pragma clang diagnostic pop
#import <Metal/Metal.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "backend/gpu/metal/metal_kernels_source.h"
#include "scalars_macros.h"

namespace scalars::metal_backend
{
namespace
{
const char* dtype_suffix(dtype type)
{
    switch (type)
    {
    case dtype::float32:
        return "float";
    case dtype::half:
        return "half";
    case dtype::bfloat16:
        return "bfloat16";
    case dtype::complex64:
        return "complex64";
    }
    throw std::invalid_argument("Metal dispatch: unsupported dtype");
}

// Minimal reduce-kind tag + identity element, local to this translation unit
// (Scalars has no expressions/ layer to import these from).
enum class reduce_kind
{
    sum,
    min,
    max
};

template <reduce_kind Kind>
constexpr float reduce_identity()
{
    if constexpr (Kind == reduce_kind::sum)
    {
        return 0.0f;
    }
    else if constexpr (Kind == reduce_kind::min)
    {
        return std::numeric_limits<float>::infinity();
    }
    else
    {
        return -std::numeric_limits<float>::infinity();
    }
}

id<MTLDevice> device()
{
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

id<MTLCommandQueue> command_queue()
{
    static id<MTLCommandQueue> q = [device() newCommandQueue];
    return q;
}

MTLCompileOptions* compile_options()
{
    MTLCompileOptions* options = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *))
    {
        options.mathMode                   = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        return options;
    }
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    return options;
}

// Compiled once, lazily, from the embedded kernels.metal source (kMetalKernelSource).
id<MTLLibrary> library()
{
    static id<MTLLibrary> lib = []
    {
        NSString*      src = [NSString stringWithUTF8String:kMetalKernelSource];
        NSError*       err = nil;
        id<MTLLibrary> l = [device() newLibraryWithSource:src options:compile_options() error:&err];
        if (l == nil)
        {
            throw std::runtime_error(
                "Metal kernel library compile failed: " +
                std::string([[err localizedDescription] UTF8String]));
        }
        return l;
    }();
    return lib;
}

std::mutex& pipeline_cache_mutex()
{
    static std::mutex m;
    return m;
}
std::unordered_map<std::string, id<MTLComputePipelineState>>& pipeline_cache()
{
    static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
    return cache;
}

id<MTLComputePipelineState> pipeline_for(const std::string& function_name)
{
    std::lock_guard<std::mutex> lock(pipeline_cache_mutex());
    auto                        it = pipeline_cache().find(function_name);
    if (it != pipeline_cache().end())
    {
        return it->second;
    }

    NSString*       ns_name = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> fn      = [library() newFunctionWithName:ns_name];
    if (fn == nil)
    {
        throw std::runtime_error("Metal kernel function not found: " + function_name);
    }

    NSError*                    err = nil;
    id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil)
    {
        throw std::runtime_error(
            "Metal pipeline creation failed for " + function_name + ": " +
            std::string([[err localizedDescription] UTF8String]));
    }

    pipeline_cache().emplace(function_name, pso);
    return pso;
}

// Pointer registry backing allocate()/deallocate()/buffer_for(): Scalars has no
// Memory package to resolve an arbitrary host pointer back to its MTLBuffer, so
// this backend owns that mapping itself. Every pointer handed to dispatch()/
// dispatch_fill()/dispatch_fused()/reduce_sum/min/max() must have come from
// allocate() below -- offsets are always 0 (one buffer per allocation, no
// sub-buffer packing).
std::mutex& buffer_registry_mutex()
{
    static std::mutex m;
    return m;
}
std::unordered_map<const void*, id<MTLBuffer>>& buffer_registry()
{
    static std::unordered_map<const void*, id<MTLBuffer>> registry;
    return registry;
}

id<MTLBuffer> buffer_for(const void* host_ptr)
{
    std::lock_guard<std::mutex> lock(buffer_registry_mutex());
    auto                        it = buffer_registry().find(host_ptr);
    if (it == buffer_registry().end())
    {
        throw std::invalid_argument(
            "Metal dispatch: pointer was not returned by scalars::metal_backend::allocate()");
    }
    return it->second;
}

NSUInteger buffer_offset_for(const void* /*host_ptr*/)
{
    return 0;
}

std::size_t next_pow2(std::size_t n)
{
    std::size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

// No-op: Scalars has no Profiler package. Kept as a named call site so `run()`
// and `reduce_buffer()` below read the same as the upstream Metal backend design.
void record_completed_command_buffer(
    SCALARS_UNUSED std::string_view name, SCALARS_UNUSED id<MTLCommandBuffer> cb)
{
}

// Dispatches `pso` with `n` threads, `threadsPerThreadgroup` capped at the pipeline's
// max — synchronous (waits for completion before returning; no stream/async support
// in the starter backend, see the Metal backend design notes).
void run(
    id<MTLComputePipelineState> pso,
    void (^encode_args)(id<MTLComputeCommandEncoder>),
    std::size_t      n,
    std::string_view event_name)
{
    id<MTLCommandBuffer>         cb  = [command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    encode_args(enc);

    NSUInteger tg_size = pso.maxTotalThreadsPerThreadgroup;
    if (tg_size > n)
    {
        tg_size = n;
    }
    if (tg_size == 0)
    {
        tg_size = 1;
    }
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError)
    {
        throw std::runtime_error(
            "Metal kernel dispatch failed: " +
            std::string([[cb.error localizedDescription] UTF8String]));
    }
    record_completed_command_buffer(event_name, cb);
}

id<MTLComputePipelineState> pipeline_for_fused_source(
    std::string const& source, std::string const& function_name)
{
    std::lock_guard<std::mutex> lock(pipeline_cache_mutex());
    auto&                       cache = pipeline_cache();
    const std::string           key   = function_name + "\n" + source;
    auto                        it    = cache.find(key);
    if (it != cache.end())
    {
        return it->second;
    }

    NSString*      src = [NSString stringWithUTF8String:source.c_str()];
    NSError*       err = nil;
    id<MTLLibrary> lib = [device() newLibraryWithSource:src options:compile_options() error:&err];
    if (lib == nil)
    {
        throw std::runtime_error(
            "Metal fused kernel compile failed: " +
            std::string([[err localizedDescription] UTF8String]));
    }
    id<MTLFunction> fn =
        [lib newFunctionWithName:[NSString stringWithUTF8String:function_name.c_str()]];
    if (fn == nil)
    {
        throw std::runtime_error("Metal fused kernel function not found: " + function_name);
    }
    id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil)
    {
        throw std::runtime_error(
            "Metal fused pipeline creation failed: " +
            std::string([[err localizedDescription] UTF8String]));
    }
    cache.emplace(key, pso);
    return pso;
}

std::size_t reduce_threadgroup_size(id<MTLComputePipelineState> pso, std::size_t n)
{
    std::size_t tg_size = 256;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup)
    {
        tg_size = pso.maxTotalThreadsPerThreadgroup;
    }
    std::size_t p = 1;
    while ((p << 1) <= tg_size)
    {
        p <<= 1;
    }
    tg_size = p;
    if (n < tg_size)
    {
        tg_size = next_pow2(n);
    }
    if (tg_size == 0)
    {
        tg_size = 1;
    }
    return tg_size;
}

float reduce_buffer(
    const char* kernel_name, const void* buffer, std::size_t n_elems, float identity, dtype type)
{
    if (n_elems == 0)
    {
        return identity;
    }
    if (n_elems > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()))
    {
        throw std::invalid_argument("Metal reduce: n_elems exceeds uint32 grid limit");
    }

    std::string function_name = std::string(kernel_name) + "_" + dtype_suffix(type);
    const void* in    = buffer;
    void*       owned = nullptr;

    while (true)
    {
        id<MTLComputePipelineState> pso      = pipeline_for(function_name);
        std::size_t const           tg_size  = reduce_threadgroup_size(pso, n_elems);
        std::size_t const           n_groups = (n_elems + tg_size - 1) / tg_size;
        if (n_groups == 0)
        {
            throw std::runtime_error("Metal reduce: empty grid");
        }

        void*    out = allocate(n_groups * sizeof(float));
        uint32_t n32 = static_cast<uint32_t>(n_elems);

        id<MTLCommandBuffer>         cb  = [command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(in) offset:buffer_offset_for(in) atIndex:0];
        [enc setBuffer:buffer_for(out) offset:buffer_offset_for(out) atIndex:1];
        [enc setBytes:&n32 length:sizeof(n32) atIndex:2];
        [enc setThreadgroupMemoryLength:(tg_size * sizeof(float)) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_groups), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(tg_size), 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError)
        {
            if (owned != nullptr)
            {
                deallocate(owned);
            }
            deallocate(out);
            throw std::runtime_error(
                std::string("Metal ") + kernel_name +
                " dispatch failed: " + std::string([[cb.error localizedDescription] UTF8String]));
        }
        record_completed_command_buffer(kernel_name, cb);

        if (owned != nullptr)
        {
            deallocate(owned);
        }

        if (n_groups == 1)
        {
            float const result = *static_cast<float*>(out);
            deallocate(out);
            return result;
        }

        owned   = out;
        in      = out;
        n_elems = n_groups;
        // Every subsequent pass reads the FP32 partials, never reduced storage.
        function_name = std::string(kernel_name) + "_float";
    }
}

}  // namespace

bool device_available()
{
    return device() != nil;
}

void* allocate(std::size_t n_bytes)
{
    if (n_bytes == 0 || device() == nil)
    {
        return nullptr;
    }
    id<MTLBuffer> buf = [device() newBufferWithLength:n_bytes options:MTLResourceStorageModeShared];
    if (buf == nil)
    {
        throw std::bad_alloc();
    }
    void* ptr = [buf contents];
    {
        std::lock_guard<std::mutex> lock(buffer_registry_mutex());
        buffer_registry().emplace(ptr, buf);
    }
    return ptr;
}

void deallocate(void* ptr)
{
    if (ptr == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(buffer_registry_mutex());
    buffer_registry().erase(ptr);
}

void dispatch(const char* kernel_name,
    const void* const*    in_buffers,
    int                   n_in,
    void*                 out_buffer,
    std::size_t           n_elems,
    dtype                 type)
{
    if (n_elems == 0)
    {
        return;
    }
    if (n_elems > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()))
    {
        throw std::invalid_argument("Metal dispatch: n_elems exceeds uint32 grid limit");
    }

    std::string                 function_name = std::string(kernel_name) + "_" + dtype_suffix(type);
    id<MTLComputePipelineState> pso           = pipeline_for(function_name);

    uint32_t n32 = static_cast<uint32_t>(n_elems);

    run(
        pso,
        ^(id<MTLComputeCommandEncoder> enc) {
          for (int i = 0; i < n_in; ++i)
          {
              [enc setBuffer:buffer_for(in_buffers[i])
                      offset:buffer_offset_for(in_buffers[i])
                     atIndex:i];
          }
          [enc setBuffer:buffer_for(out_buffer) offset:buffer_offset_for(out_buffer) atIndex:n_in];
          [enc setBytes:&n32 length:sizeof(n32) atIndex:n_in + 1];
        },
        n_elems,
        function_name);
}

void dispatch_fill(void* out_buffer, float value, std::size_t n_elems, dtype type)
{
    if (n_elems == 0)
    {
        return;
    }
    if (n_elems > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()))
    {
        throw std::invalid_argument("Metal dispatch: n_elems exceeds uint32 grid limit");
    }

    std::string                 function_name = std::string("fill_") + dtype_suffix(type);
    id<MTLComputePipelineState> pso           = pipeline_for(function_name);
    uint32_t                    n32 = static_cast<uint32_t>(n_elems);

    run(
        pso,
        ^(id<MTLComputeCommandEncoder> enc) {
          [enc setBuffer:buffer_for(out_buffer) offset:buffer_offset_for(out_buffer) atIndex:0];
          [enc setBytes:&value length:sizeof(value) atIndex:1];
          [enc setBytes:&n32 length:sizeof(n32) atIndex:2];
        },
        n_elems,
        function_name);
}

void dispatch_fused(std::string const& kernel_source,
    const void* const*                 in_buffers,
    int                                n_in,
    float const*                       scalars,
    int                                n_scalars,
    void*                              out_buffer,
    std::size_t                        n_elems,
    dtype                              type)
{
    if (n_elems == 0)
    {
        return;
    }
    if (n_elems > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()))
    {
        throw std::invalid_argument("Metal dispatch: n_elems exceeds uint32 grid limit");
    }

    std::string                 function_name = std::string("fused_") + dtype_suffix(type);
    id<MTLComputePipelineState> pso = pipeline_for_fused_source(kernel_source, function_name);
    uint32_t                    n32 = static_cast<uint32_t>(n_elems);

    run(
        pso,
        ^(id<MTLComputeCommandEncoder> enc) {
          int idx = 0;
          for (int i = 0; i < n_in; ++i)
          {
              [enc setBuffer:buffer_for(in_buffers[i])
                      offset:buffer_offset_for(in_buffers[i])
                     atIndex:idx++];
          }
          for (int j = 0; j < n_scalars; ++j)
          {
              [enc setBytes:&scalars[j] length:sizeof(float) atIndex:idx++];
          }
          [enc setBuffer:buffer_for(out_buffer) offset:buffer_offset_for(out_buffer) atIndex:idx++];
          [enc setBytes:&n32 length:sizeof(n32) atIndex:idx];
        },
        n_elems,
        function_name);
}

float reduce_sum(const void* buffer, std::size_t n_elems, dtype type)
{
    return reduce_buffer("reduce_sum", buffer, n_elems, reduce_identity<reduce_kind::sum>(), type);
}

float reduce_min(const void* buffer, std::size_t n_elems, dtype type)
{
    return reduce_buffer("reduce_min", buffer, n_elems, reduce_identity<reduce_kind::min>(), type);
}

float reduce_max(const void* buffer, std::size_t n_elems, dtype type)
{
    return reduce_buffer("reduce_max", buffer, n_elems, reduce_identity<reduce_kind::max>(), type);
}

void dispatch(const char* name, const void* const* inputs, int n_in, void* out, std::size_t n)
{
    dispatch(name, inputs, n_in, out, n, dtype::float32);
}
void dispatch_fill(void* out, float value, std::size_t n)
{
    dispatch_fill(out, value, n, dtype::float32);
}
void dispatch_fused(std::string const& source,
    const void* const*                 inputs,
    int                                n_in,
    float const*                       scalars,
    int                                n_scalars,
    void*                              out,
    std::size_t                        n)
{
    dispatch_fused(source, inputs, n_in, scalars, n_scalars, out, n, dtype::float32);
}
float reduce_sum(const void* buffer, std::size_t n)
{
    return reduce_sum(buffer, n, dtype::float32);
}
float reduce_min(const void* buffer, std::size_t n)
{
    return reduce_min(buffer, n, dtype::float32);
}
float reduce_max(const void* buffer, std::size_t n)
{
    return reduce_max(buffer, n, dtype::float32);
}

}  // namespace scalars::metal_backend
