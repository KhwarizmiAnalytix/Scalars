#pragma once

// Backend-neutral gpu*/cuda*/hip* aliases: gpuGetDeviceCount, gpuStreamCreate,
// gpuStreamDestroy, gpuStreamSynchronize, gpuSuccess. Lets callers write one
// code path instead of their own CUDA/HIP/Metal #if ladder.
//
// Metal has no stream concept, so it only gets gpuGetDeviceCount here (routed
// through the metal_device_probe.mm shim used by Testing/Cxx, since callers
// like TestTensorGpu.cpp must stay plain .cpp) — guard stream-based code with
// SCALARS_HAS_CUDA || SCALARS_HAS_HIP yourself.

#if SCALARS_HAS_CUDA
#include <cuda_runtime.h>
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuSuccess cudaSuccess
#elif SCALARS_HAS_HIP
#include <hip/hip_runtime.h>
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuSuccess hipSuccess
#elif SCALARS_HAS_METAL
// Implemented in Testing/Cxx/metal_device_probe.mm (Objective-C++) — callers
// needing this branch (e.g. TestTensorGpu.cpp) must stay plain .cpp themselves.
extern "C" int xsigma_metal_device_count();
#define gpuGetDeviceCount(pn) (*(pn) = xsigma_metal_device_count())
#endif
