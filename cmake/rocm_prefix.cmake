# =============================================================================
# Scalars ROCm / HIP prefix
#
# hip-config.cmake lives under /opt/rocm (or $ROCM_PATH), which is not a CMake
# default search path. Debian/Ubuntu also install a /usr/bin/hipcc alternatives
# wrapper; taking dirname(hipcc) then yields HIP_PATH=/usr, which is not an SDK
# root and makes find_package(hip) fail.
#
# Ported from Vectorization's Cmake/tools/rocm_prefix.cmake
# (xsigma_setup_rocm_prefix -> scalars_setup_rocm_prefix).
# =============================================================================

include_guard(GLOBAL)

# Prepend a real ROCm SDK root to CMAKE_PREFIX_PATH and correct HIP_PATH/ROCM_PATH
# when they point at /usr. Idempotent.
macro(scalars_setup_rocm_prefix)
    if(NOT SCALARS_ROCM_PREFIX)
        set(_scalars_rocm_roots)
        foreach(_cand IN ITEMS "$ENV{ROCM_PATH}" "$ENV{HIP_PATH}" "/opt/rocm")
            if(_cand
               AND NOT _cand STREQUAL "/usr"
               AND NOT _cand STREQUAL "/usr/local"
               AND NOT _cand STREQUAL "/"
            )
                list(APPEND _scalars_rocm_roots "${_cand}")
            endif()
        endforeach()
        file(GLOB _scalars_rocm_versioned "/opt/rocm-*")
        list(APPEND _scalars_rocm_roots ${_scalars_rocm_versioned})

        set(_scalars_rocm_found "")
        foreach(_root IN LISTS _scalars_rocm_roots)
            if(IS_DIRECTORY "${_root}")
                if(EXISTS "${_root}/lib/cmake/hip/hip-config.cmake"
                   OR EXISTS "${_root}/lib64/cmake/hip/hip-config.cmake"
                   OR EXISTS "${_root}/hip/lib/cmake/hip/hip-config.cmake"
                   OR EXISTS "${_root}/include/hip/hip_runtime.h"
                   OR EXISTS "${_root}/hip/include/hip/hip_runtime.h"
                )
                    set(_scalars_rocm_found "${_root}")
                    break()
                endif()
            endif()
        endforeach()

        if(_scalars_rocm_found)
            list(PREPEND CMAKE_PREFIX_PATH "${_scalars_rocm_found}")
            list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
            set(SCALARS_ROCM_PREFIX "${_scalars_rocm_found}")
            if(NOT DEFINED ENV{ROCM_PATH} OR "$ENV{ROCM_PATH}" STREQUAL "/usr")
                set(ENV{ROCM_PATH} "${_scalars_rocm_found}")
            endif()
            if(NOT DEFINED ENV{HIP_PATH} OR "$ENV{HIP_PATH}" STREQUAL "/usr")
                set(ENV{HIP_PATH} "${_scalars_rocm_found}")
            endif()
            message(STATUS "Scalars: ROCm/HIP SDK root: ${_scalars_rocm_found}")
            # hip-config.cmake looks at AMDGPU_TARGETS / GPU_TARGETS, not only
            # CMAKE_HIP_ARCHITECTURES. Without these, it falls back to gfx906 and
            # may inject --offload-arch into host CXX flags.
            if(CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "native")
                if(NOT AMDGPU_TARGETS)
                    set(AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}")
                endif()
                if(NOT GPU_TARGETS)
                    set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}")
                endif()
            endif()
        endif()
        unset(_scalars_rocm_roots)
        unset(_scalars_rocm_versioned)
        unset(_scalars_rocm_found)
        unset(_cand)
        unset(_root)
    endif()
endmacro()
