#pragma once

// Plain C++ surface over the fixed Metal kernel set (backend/gpu/metal/kernels.metal) —
// no Objective-C types cross this boundary, so a header-only caller compiled by ordinary
// clang++ can call into it without becoming Objective-C++ itself.
//
// in_buffers/out_buffer must be host-visible pointers into page-aligned Metal
// shared storage (MTLResourceStorageModeShared). The allocation length must be
// a multiple of the system page size and at least the byte size of n_elems.
// dispatch() rounds that byte size up to a page multiple and wraps it with
// newBufferWithBytesNoCopy before binding with setBuffer:offset:.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace scalars::metal_backend
{
// Reduced buffers contain the two-byte encodings of scalars::half/bfloat16.
// All operation math and reduction partials use float. complex64 buffers
// contain scalars::complex<float> (= MSL float2: x is real, y is imag), the
// same interleaved layout the CPU complex backends use; there is no
// complex128 (MSL has no double type, matching the backend's existing
// float32-only -- no double -- support), and complex64 has no reduce_sum/
// min/max (see kernels.metal's complex64 block for why).
enum class dtype : std::uint8_t
{
    float32,
    half,
    bfloat16,
    complex64
};

bool device_available();

// Thread-confined batch of dependent launches. dispatch* only encodes work;
// submit() sends the current batch to the GPU without waiting, and wait()
// submits any remaining work, waits for all submissions, and reports errors.
// Do not read, write, or release host-visible buffers until wait() completes.
// Encoded command buffers retain the Metal buffer objects they bind. The
// caller keeps the backing storage alive for that wait.
// Destruction submits and drains outstanding work without throwing; use wait()
// to observe GPU errors. Separate contexts may be used by separate threads.
class execution_context
{
public:
    execution_context();
    ~execution_context();
    execution_context(const execution_context&)            = delete;
    execution_context& operator=(const execution_context&) = delete;

    void dispatch(const char* kernel_name,
        const void* const*    in_buffers,
        int                   n_in,
        void*                 out_buffer,
        std::size_t           n_elems,
        dtype                 type = dtype::float32);
    void dispatch_fill(
        void* out_buffer, float value, std::size_t n_elems, dtype type = dtype::float32);
    void dispatch_fused(std::string_view   kernel_source,
        const void* const*                 in_buffers,
        int                                n_in,
        const float*                       scalars,
        int                                n_scalars,
        void*                              out_buffer,
        std::size_t                        n_elems,
        dtype                              type = dtype::float32);
    void submit();
    void wait();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

// Dispatches the kernel named kernel_name (e.g. "add" -> "add_float") over n_elems
// threads. in_buffers holds n_in input pointers, consumed in the same order the
// corresponding .metal kernel declares its [[buffer(k)]] arguments. Synchronous
// (waits for GPU completion before returning).
void dispatch(const char* kernel_name,
    const void* const*    in_buffers,
    int                   n_in,
    void*                 out_buffer,
    std::size_t           n_elems);

void dispatch_fill(void* out_buffer, float value, std::size_t n_elems);

// One fused element-wise kernel compiled from `kernel_source` (MSL). Tensor
// leaves bind as device buffers 0..n_in-1, scalars as constant buffers
// n_in..n_in+n_scalars-1 (setBytes), then out and n. The kernel function
// must be named fused_float. Source is compiled once and cached.
void dispatch_fused(std::string_view    kernel_source,
    const void* const*                  in_buffers,
    int                                 n_in,
    float const*                        scalars,
    int                                 n_scalars,
    void*                               out_buffer,
    std::size_t                         n_elems);

// Multi-block reductions. Each threadgroup writes one partial; the host
// launcher encodes all ping-pong passes in one command buffer, reusing per-thread
// scratch storage and waiting once for the result. n_elems is not limited
// to maxTotalThreadsPerThreadgroup. Empty input returns the matching
// accumulate / hmin / hmax identity (0, +infinity, -infinity).
float reduce_sum(const void* buffer, std::size_t n_elems);
float reduce_min(const void* buffer, std::size_t n_elems);
float reduce_max(const void* buffer, std::size_t n_elems);

// Typed variants preserve the float-only entry points above. The type applies
// to every input and the output; fill values and fused scalar arguments stay
// FP32. A fused source must define fused_float, fused_half, or fused_bfloat16.
void dispatch(const char* kernel_name,
    const void* const*    in_buffers,
    int                   n_in,
    void*                 out_buffer,
    std::size_t           n_elems,
    dtype                 type);
void dispatch_fill(void* out_buffer, float value, std::size_t n_elems, dtype type);
void dispatch_fused(std::string_view    kernel_source,
    const void* const*                  in_buffers,
    int                                 n_in,
    float const*                        scalars,
    int                                 n_scalars,
    void*                               out_buffer,
    std::size_t                         n_elems,
    dtype                               type);
// Reduced sums are never narrowed between threadgroups. All return FP32.
float reduce_sum(const void* buffer, std::size_t n_elems, dtype type);
float reduce_min(const void* buffer, std::size_t n_elems, dtype type);
float reduce_max(const void* buffer, std::size_t n_elems, dtype type);

}  // namespace scalars::metal_backend
