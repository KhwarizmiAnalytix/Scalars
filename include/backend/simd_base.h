#pragma once

#include "scalars_macros.h"

#if SCALARS_VECTORIZED
#include "common/intrinsics.h"
#endif

namespace scalars
{

#if SCALARS_VECTORIZED
#if __SCALARS_COMPILER_PGI__
using prefetch_ptr_type = const void*;
#else
using prefetch_ptr_type = const char*;
#endif
#endif

// Shared declarations only: leaf backends must not include the dispatcher.
template <typename T> struct simd
{
};

}  // namespace scalars
