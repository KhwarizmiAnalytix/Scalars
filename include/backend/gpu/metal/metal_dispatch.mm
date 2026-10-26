#include "backend/gpu/metal/metal_dispatch.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#import <Foundation/Foundation.h>
#pragma clang diagnostic pop
#import <Metal/Metal.h>

#include <unistd.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

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
enum class reduce_kind : std::uint8_t
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

// Returns the element size in bytes for each dtype.
std::size_t bytes_per_elem(dtype type)
{
    switch (type)
    {
    case dtype::float32:  return 4;
    case dtype::half:     return 2;
    case dtype::bfloat16: return 2;
    case dtype::complex64: return 8;
    }
    throw std::invalid_argument("Metal dispatch: unsupported dtype");
}

// newBufferWithBytesNoCopy accepts only a page-aligned address and a length
// that is a multiple of the page size. Callers allocate that span; kernels
// still observe the unrounded element count.
std::size_t page_multiple(std::size_t bytes)
{
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
    {
        throw std::runtime_error("Metal dispatch: page size unavailable");
    }
    const auto page_size = static_cast<std::size_t>(page);
    if (bytes == 0)
    {
        return page_size;
    }
    const std::size_t remainder = bytes % page_size;
    if (remainder == 0)
    {
        return bytes;
    }
    if (bytes > std::numeric_limits<std::size_t>::max() - (page_size - remainder))
    {
        throw std::invalid_argument("Metal dispatch: buffer length exceeds page alignment limit");
    }
    return bytes + (page_size - remainder);
}

// One MTLBuffer object per host address. Metal tracks hazards by object
// identity, so a later kernel in the same batch must bind the same buffer.
using buffer_cache = std::unordered_map<const void*, id<MTLBuffer>>;

// Wraps a page-aligned Metal shared-storage pointer without copying.
// Throws if ptr is not page-aligned (requirement of newBufferWithBytesNoCopy).
id<MTLBuffer> make_buffer(buffer_cache& cache, const void* ptr, std::size_t bytes)
{
    if (ptr == nullptr)
    {
        throw std::invalid_argument(
            "Metal dispatch: pointer is not page-aligned Metal shared storage");
    }
    const std::size_t length = page_multiple(bytes);
    const auto        found  = cache.find(ptr);
    if (found != cache.end())
    {
        if (found->second.length < length)
        {
            throw std::invalid_argument(
                "Metal dispatch: shared storage is smaller than the requested span");
        }
        return found->second;
    }
    id<MTLBuffer> buf =
        [device() newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                    length:length
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
    if (buf == nil)
    {
        throw std::invalid_argument(
            "Metal dispatch: pointer is not page-aligned Metal shared storage");
    }
    cache.emplace(ptr, buf);
    return buf;
}

std::size_t next_pow2(std::size_t n)
{
    std::size_t p = 1;
    while (p < n)
    {
        p <<= 1;
    }
    return p;
}

// No-op: Scalars has no Profiler package. Retain a named completion hook for
// dispatches, explicit batches, and reductions.
void record_completed_command_buffer(
    SCALARS_UNUSED std::string_view name, SCALARS_UNUSED id<MTLCommandBuffer> cb)
{
}

// Encoders share a command buffer so dependent operations need no host waits.
void encode_run(id<MTLCommandBuffer> cb,
    id<MTLComputePipelineState>      pso,
    void (^encode_args)(id<MTLComputeCommandEncoder>),
    std::size_t n)
{
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    try
    {
        encode_args(enc);
    }
    catch (...)
    {
        [enc endEncoding];
        throw;
    }

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
}

id<MTLComputePipelineState> pipeline_for_fused_source(
    std::string_view source, std::string const& function_name)
{
    std::lock_guard<std::mutex> lock(pipeline_cache_mutex());
    auto&                       cache = pipeline_cache();
    const std::string           key   = function_name + "\n" + std::string(source);
    auto                        it    = cache.find(key);
    if (it != cache.end())
    {
        return it->second;
    }

    NSString* src = [[NSString alloc] initWithBytes:source.data()
                                             length:source.size()
                                           encoding:NSUTF8StringEncoding];
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

struct reduction_workspace
{
    id<MTLBuffer> buffers[2] = {nil, nil};

    id<MTLBuffer> get(int index, std::size_t bytes)
    {
        if (buffers[index] == nil || buffers[index].length < bytes)
        {
            buffers[index] =
                [device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (buffers[index] == nil)
            {
                throw std::bad_alloc();
            }
        }
        return buffers[index];
    }
};

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

    @autoreleasepool
    {
        std::string          function_name = std::string(kernel_name) + "_" + dtype_suffix(type);
        buffer_cache         inputs;
        id<MTLBuffer>        in = make_buffer(inputs, buffer, n_elems * bytes_per_elem(type));
        id<MTLCommandBuffer> cb            = [command_queue() commandBuffer];
        // Each calling thread owns its scratch buffers. Every pass is encoded in
        // order, with Metal's tracked resource hazards between the encoders.
        thread_local reduction_workspace workspace;
        int                              pass = 0;

        while (true)
        {
            id<MTLComputePipelineState> pso      = pipeline_for(function_name);
            std::size_t const           tg_size  = reduce_threadgroup_size(pso, n_elems);
            std::size_t const           n_groups = (n_elems + tg_size - 1) / tg_size;
            if (n_groups == 0)
            {
                throw std::runtime_error("Metal reduce: empty grid");
            }

            id<MTLBuffer> out = workspace.get(pass % 2, n_groups * sizeof(float));
            uint32_t      n32 = static_cast<uint32_t>(n_elems);

            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:in offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBytes:&n32 length:sizeof(n32) atIndex:2];
            [enc setThreadgroupMemoryLength:(tg_size * sizeof(float)) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_groups), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(tg_size), 1, 1)];
            [enc endEncoding];
            if (n_groups == 1)
            {
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status == MTLCommandBufferStatusError)
                {
                    throw std::runtime_error(
                        std::string("Metal ") + kernel_name + " dispatch failed: " +
                        std::string([[cb.error localizedDescription] UTF8String]));
                }
                record_completed_command_buffer(kernel_name, cb);
                return *static_cast<const float*>(out.contents);
            }

            in      = out;
            n_elems = n_groups;
            ++pass;
            // Every subsequent pass reads the FP32 partials, never reduced storage.
            function_name = std::string(kernel_name) + "_float";
        }
    }
}

}  // namespace

bool device_available()
{
    return device() != nil;
}

namespace
{
void encode_dispatch(id<MTLCommandBuffer> cb,
    const char*                           kernel_name,
    const void* const*                    in_buffers,
    int                                   n_in,
    void*                                 out_buffer,
    std::size_t                           n_elems,
    dtype                                 type,
    buffer_cache&                         cache)
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

    uint32_t          n32       = static_cast<uint32_t>(n_elems);
    std::size_t const buf_bytes = n_elems * bytes_per_elem(type);
    // The block must capture a pointer. Capturing the cache reference would
    // copy the map and give later kernels a different MTLBuffer.
    buffer_cache* const buffers = &cache;

    encode_run(
        cb,
        pso,
        ^(id<MTLComputeCommandEncoder> enc) {
          for (int i = 0; i < n_in; ++i)
          {
              [enc setBuffer:make_buffer(*buffers, in_buffers[i], buf_bytes) offset:0 atIndex:i];
          }
          [enc setBuffer:make_buffer(*buffers, out_buffer, buf_bytes) offset:0 atIndex:n_in];
          [enc setBytes:&n32 length:sizeof(n32) atIndex:n_in + 1];
        },
        n_elems);
}

void encode_fill(id<MTLCommandBuffer> cb,
    void*                              out_buffer,
    float                              value,
    std::size_t                        n_elems,
    dtype                              type,
    buffer_cache&                      cache)
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
    uint32_t                    n32       = static_cast<uint32_t>(n_elems);
    std::size_t const           buf_bytes = n_elems * bytes_per_elem(type);
    buffer_cache* const         buffers   = &cache;

    encode_run(
        cb,
        pso,
        ^(id<MTLComputeCommandEncoder> enc) {
          [enc setBuffer:make_buffer(*buffers, out_buffer, buf_bytes) offset:0 atIndex:0];
          [enc setBytes:&value length:sizeof(value) atIndex:1];
          [enc setBytes:&n32 length:sizeof(n32) atIndex:2];
        },
        n_elems);
}

void encode_fused(id<MTLCommandBuffer> cb,
    std::string_view                   kernel_source,
    const void* const*                 in_buffers,
    int                                n_in,
    float const*                       scalars,
    int                                n_scalars,
    void*                              out_buffer,
    std::size_t                        n_elems,
    dtype                              type,
    buffer_cache&                      cache)
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
    id<MTLComputePipelineState> pso           = pipeline_for_fused_source(kernel_source, function_name);
    uint32_t                    n32       = static_cast<uint32_t>(n_elems);
    std::size_t const           buf_bytes = n_elems * bytes_per_elem(type);
    buffer_cache* const         buffers   = &cache;

    encode_run(
        cb,
        pso,
        ^(id<MTLComputeCommandEncoder> enc) {
          int idx = 0;
          for (int i = 0; i < n_in; ++i)
          {
              [enc setBuffer:make_buffer(*buffers, in_buffers[i], buf_bytes) offset:0 atIndex:idx++];
          }
          for (int j = 0; j < n_scalars; ++j)
          {
              [enc setBytes:&scalars[j] length:sizeof(float) atIndex:idx++];
          }
          [enc setBuffer:make_buffer(*buffers, out_buffer, buf_bytes) offset:0 atIndex:idx++];
          [enc setBytes:&n32 length:sizeof(n32) atIndex:idx];
        },
        n_elems);
}

template <class Encode> void synchronous_dispatch(Encode encode)
{
    @autoreleasepool
    {
        buffer_cache         cache;
        id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
        encode(cb, cache);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError)
        {
            throw std::runtime_error("Metal kernel dispatch failed: " +
                                     std::string([[cb.error localizedDescription] UTF8String]));
        }
        record_completed_command_buffer("dispatch", cb);
    }
}
}  // namespace

struct execution_context::impl
{
    id<MTLCommandBuffer>              current = nil;
    buffer_cache                      buffers;
    id<MTLSharedEvent>                event          = nil;
    uint64_t                          last_submitted = 0;
    std::vector<id<MTLCommandBuffer>> pending;

    id<MTLSharedEvent> batch_event()
    {
        if (event == nil)
        {
            event = [device() newSharedEvent];
            if (event == nil)
            {
                throw std::runtime_error("Metal shared event creation failed");
            }
        }
        return event;
    }

    id<MTLCommandBuffer> buffer()
    {
        if (current == nil)
        {
            current = [command_queue() commandBuffer];
            if (current == nil)
            {
                throw std::runtime_error("Metal command buffer creation failed");
            }
            // Later batches wait for the preceding submit. Metal does not track
            // hazards across command buffers.
            if (last_submitted != 0)
            {
                [current encodeWaitForEvent:batch_event() value:last_submitted];
            }
        }
        return current;
    }
};

execution_context::execution_context() : impl_(std::make_unique<impl>()) {}

execution_context::~execution_context()
{
    // Avoid allocating or throwing in destruction, including after an encode
    // error. Retained command buffers keep bound resources alive until drained.
    if (impl_->current != nil)
    {
        [impl_->current commit];
        [impl_->current waitUntilCompleted];
    }
    for (id<MTLCommandBuffer> cb : impl_->pending)
    {
        [cb waitUntilCompleted];
    }
}

void execution_context::dispatch(
    const char* name, const void* const* inputs, int n_in, void* out, std::size_t n, dtype type)
{
    if (n != 0)
    {
        @autoreleasepool
        {
            encode_dispatch(impl_->buffer(), name, inputs, n_in, out, n, type, impl_->buffers);
        }
    }
}

void execution_context::dispatch_fill(void* out, float value, std::size_t n, dtype type)
{
    if (n != 0)
    {
        @autoreleasepool
        {
            encode_fill(impl_->buffer(), out, value, n, type, impl_->buffers);
        }
    }
}

void execution_context::dispatch_fused(std::string_view source,
    const void* const*                                   inputs,
    int                                                   n_in,
    const float*                                          scalars,
    int                                                   n_scalars,
    void*                                                 out,
    std::size_t                                           n,
    dtype                                                 type)
{
    if (n != 0)
    {
        @autoreleasepool
        {
            encode_fused(
                impl_->buffer(), source, inputs, n_in, scalars, n_scalars, out, n, type, impl_->buffers);
        }
    }
}

void execution_context::submit()
{
    if (impl_->current != nil)
    {
        // Allocate before commit, leaving the current batch intact on failure.
        id<MTLSharedEvent> event = impl_->batch_event();
        impl_->pending.push_back(impl_->current);
        const uint64_t value = impl_->last_submitted + 1;
        [impl_->current encodeSignalEvent:event value:value];
        [impl_->current commit];
        impl_->last_submitted = value;
        impl_->current        = nil;
    }
}

void execution_context::wait()
{
    submit();
    std::string error;
    for (id<MTLCommandBuffer> cb : impl_->pending)
    {
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError && error.empty())
        {
            error = [[cb.error localizedDescription] UTF8String];
        }
        record_completed_command_buffer("batch", cb);
    }
    impl_->pending.clear();
    impl_->buffers.clear();
    if (!error.empty())
    {
        throw std::runtime_error("Metal kernel dispatch failed: " + error);
    }
}

void dispatch(
    const char* name, const void* const* inputs, int n_in, void* out, std::size_t n, dtype type)
{
    if (n != 0)
    {
        synchronous_dispatch([&](id<MTLCommandBuffer> cb, buffer_cache& buffers)
            { encode_dispatch(cb, name, inputs, n_in, out, n, type, buffers); });
    }
}

void dispatch_fill(void* out, float value, std::size_t n, dtype type)
{
    if (n != 0)
    {
        synchronous_dispatch([&](id<MTLCommandBuffer> cb, buffer_cache& buffers)
            { encode_fill(cb, out, value, n, type, buffers); });
    }
}

void dispatch_fused(std::string_view    source,
    const void* const*                  inputs,
    int                                 n_in,
    const float*                        scalars,
    int                                 n_scalars,
    void*                               out,
    std::size_t                         n,
    dtype                               type)
{
    if (n != 0)
    {
        synchronous_dispatch([&](id<MTLCommandBuffer> cb, buffer_cache& buffers) {
            encode_fused(cb, source, inputs, n_in, scalars, n_scalars, out, n, type, buffers);
        });
    }
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
void dispatch_fused(std::string_view    source,
    const void* const*                  inputs,
    int                                 n_in,
    float const*                        scalars,
    int                                 n_scalars,
    void*                               out,
    std::size_t                         n)
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
