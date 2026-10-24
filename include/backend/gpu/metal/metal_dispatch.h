#pragma once

// Plain C++ surface over the fixed Metal kernel set (backend/gpu/metal/kernels.metal) —
// no Objective-C types cross this boundary, so a header-only caller compiled by ordinary
// clang++ can call into it without becoming Objective-C++ itself.
//
// in_buffers/out_buffer are raw host pointers previously returned by allocate() below —
// host-visible pointers into shared-storage MTLBuffers tracked by this backend's own
// pointer registry (see metal_dispatch.mm). dispatch() resolves them via buffer_for()
// before binding with setBuffer:offset:.

#include <cstddef>
#include <string>

namespace scalars::metal_backend
{
bool device_available();

// Allocates n_bytes of Metal shared-storage memory and returns a host-visible
// pointer suitable for the in_buffers/out_buffer arguments below. Paired with
// deallocate(). Returns nullptr if n_bytes == 0 or no Metal device is available.
void* allocate(std::size_t n_bytes);
void  deallocate(void* ptr);

// Dispatches the kernel named kernel_name (e.g. "add" -> "add_float") over n_elems
// threads. in_buffers holds n_in input pointers, consumed in the same order the
// corresponding .metal kernel declares its [[buffer(k)]] arguments. Synchronous
// (waits for GPU completion before returning).
void dispatch(
    const char*        kernel_name,
    const void* const* in_buffers,
    int                n_in,
    void*              out_buffer,
    std::size_t        n_elems);

void dispatch_fill(void* out_buffer, float value, std::size_t n_elems);

// One fused element-wise kernel compiled from `kernel_source` (MSL). Tensor
// leaves bind as device buffers 0..n_in-1, scalars as constant buffers
// n_in..n_in+n_scalars-1 (setBytes), then out and n. The kernel function
// must be named fused_float. Source is compiled once and cached.
void dispatch_fused(
    std::string const& kernel_source,
    const void* const* in_buffers,
    int                n_in,
    float const*       scalars,
    int                n_scalars,
    void*              out_buffer,
    std::size_t        n_elems);

// Multi-block reductions. Each threadgroup writes one partial; the host
// launcher ping-pongs until a single value remains. n_elems is not limited
// to maxTotalThreadsPerThreadgroup. Empty input returns the matching
// accumulate / hmin / hmax identity (0, +FLT_MAX, -FLT_MAX).
float reduce_sum(const void* buffer, std::size_t n_elems);
float reduce_min(const void* buffer, std::size_t n_elems);
float reduce_max(const void* buffer, std::size_t n_elems);

}  // namespace scalars::metal_backend
