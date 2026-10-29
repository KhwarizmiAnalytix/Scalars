# GPU backend selection and toolchain plumbing for Scalars, ported from
# Vectorization's include/CMakeLists.txt GPU sections (VECTORIZATION_ -> SCALARS_,
# Vectorization target -> Scalars target).
#
# SCALARS_GPU_BACKEND (cache: none|cuda|hip|metal) selects the GPU backend used by
# include/backend/gpu/*. Independent of SCALARS_CPU_BACKEND -- any CPU tier can be
# combined with any GPU backend.
#
# SCALARS_USE_HIP (ifdef-style) and SCALARS_HIP_VERSION (integer, major*100+minor,
# matching PyTorch's TORCH_HIP_VERSION convention) are what include/types/bfloat16.h,
# complex.h, and half.h test. SCALARS_HAS_CUDA/HIP/METAL (0/1) are for the ported
# backend/ tree's own dispatch.
include_guard(GLOBAL)

# hip::device publishes Clang device flags for the legacy "compile HIP as CXX"
# workflow (`-x hip`, `--offload-arch`, `--hip-link`). CMake's HIP language
# already supplies those for HIP sources. On a host g++ they land on every .cpp
# file and the compile fails.
function(_scalars_drop_hip_cxx_device_flags target)
    if(NOT TARGET "${target}")
        return()
    endif()
    foreach(_prop IN ITEMS INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES
                           INTERFACE_LINK_OPTIONS
    )
        get_property(
            _prop_set
            TARGET "${target}"
            PROPERTY "${_prop}"
            SET
        )
        if(NOT _prop_set)
            continue()
        endif()
        get_property(_opts TARGET "${target}" PROPERTY "${_prop}")
        # hip_add_interface_compile_flags(hip::device -x hip) stores
        # "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-x;hip>", which CMake splits into
        # "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-x" and "hip>". Match that pair, the
        # unsplit spelling, and the offload/link flags.
        set(_kept "")
        set(_dropped 0)
        set(_i 0)
        list(LENGTH _opts _n)
        while(_i LESS _n)
            list(GET _opts ${_i} _opt)
            set(_consume 1)
            set(_poison 0)
            if(_opt STREQUAL "-x" OR _opt MATCHES "(^|SHELL:)-x$")
                math(EXPR _j "${_i} + 1")
                if(_j LESS _n)
                    list(GET _opts ${_j} _next)
                    if(_next STREQUAL "hip" OR _next MATCHES "^hip>")
                        set(_poison 1)
                        set(_consume 2)
                    endif()
                endif()
            endif()
            if(NOT _poison AND _opt MATCHES
                               "--offload-arch|--hip-link|-xhip|-x[ ;]hip|SHELL:-x;hip"
            )
                set(_poison 1)
            endif()
            if(_poison)
                math(EXPR _dropped "${_dropped} + ${_consume}")
            else()
                list(APPEND _kept "${_opt}")
            endif()
            math(EXPR _i "${_i} + ${_consume}")
        endwhile()
        if(_dropped GREATER 0)
            list(LENGTH _kept _kept_len)
            if(_kept_len EQUAL 0)
                set_property(TARGET "${target}" PROPERTY "${_prop}" "")
            else()
                set_property(TARGET "${target}" PROPERTY "${_prop}" ${_kept})
            endif()
            message(STATUS "Scalars: dropped ${_dropped} host-incompatible ${_prop} from ${target}")
        endif()
    endforeach()
endfunction()

set(SCALARS_GPU_BACKEND "none" CACHE STRING "GPU backend for Scalars: none, cuda, hip, metal")
set_property(CACHE SCALARS_GPU_BACKEND PROPERTY STRINGS none cuda hip metal)
if(NOT SCALARS_GPU_BACKEND MATCHES "^(none|cuda|hip|metal)$")
    message(
        FATAL_ERROR
            "SCALARS_GPU_BACKEND must be one of: none, cuda, hip, metal (got '${SCALARS_GPU_BACKEND}')"
    )
endif()
if(SCALARS_GPU_BACKEND STREQUAL "metal" AND NOT APPLE)
    message(FATAL_ERROR "SCALARS_GPU_BACKEND=metal is only supported on Apple platforms.")
endif()

set(SCALARS_HAS_CUDA 0)
set(SCALARS_HAS_HIP 0)
set(SCALARS_HAS_METAL 0)
set(_scalars_hip_version 0)

# ---- HIP GPU backend --------------------------------------------------------
if(SCALARS_GPU_BACKEND STREQUAL "hip")
    if(WIN32)
        message(
            FATAL_ERROR
                "SCALARS_GPU_BACKEND=hip is not supported on Windows (HIP/ROCm is Unix-only here). "
                "Use SCALARS_GPU_BACKEND=cuda on Windows, or build on Linux for HIP."
        )
    endif()

    if(NOT DEFINED CMAKE_HIP_COMPILER AND DEFINED ENV{ROCM_PATH})
        set(CMAKE_HIP_COMPILER "$ENV{ROCM_PATH}/bin/hipcc" CACHE FILEPATH "HIP compiler")
    endif()

    include("${CMAKE_CURRENT_LIST_DIR}/rocm_prefix.cmake")
    scalars_setup_rocm_prefix()
    find_package(hip QUIET)
    if(NOT hip_FOUND)
        message(
            WARNING "Scalars: SCALARS_GPU_BACKEND=hip but the HIP package was not found via "
                    "find_package(hip). Set ROCM_PATH or CMAKE_PREFIX_PATH to your ROCm install."
        )
    endif()

    enable_language(HIP)
    _scalars_drop_hip_cxx_device_flags(hip::device)
    _scalars_drop_hip_cxx_device_flags(hip::host)
    set(SCALARS_HAS_HIP 1)

    set(SCALARS_GPU_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"
        CACHE STRING "HIP GPU architectures (gfxXXX)"
    )
    if(CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "native")
        set(SCALARS_GPU_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
            CACHE STRING "HIP GPU architectures (gfxXXX)" FORCE
        )
    endif()

    # SCALARS_HIP_VERSION mirrors PyTorch's TORCH_HIP_VERSION: major*100+minor.
    if(hip_FOUND AND DEFINED hip_VERSION)
        string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _scalars_hip_ver_match "${hip_VERSION}")
        if(_scalars_hip_ver_match)
            math(EXPR _scalars_hip_version "${CMAKE_MATCH_1} * 100 + ${CMAKE_MATCH_2}")
        endif()
        unset(_scalars_hip_ver_match)
    endif()
    message(
        STATUS
            "Scalars: HIP GPU backend enabled (archs: ${SCALARS_GPU_ARCHITECTURES}, SCALARS_HIP_VERSION=${_scalars_hip_version})"
    )
endif()

# ---- CUDA GPU backend --------------------------------------------------------
if(SCALARS_GPU_BACKEND STREQUAL "cuda")
    # clang++ GNU driver on Windows (CMAKE_CXX_SIMULATE_ID=MSVC) generates
    # MSVC-style CUDA link flags: -LIBPATH: for the search path and bare .lib
    # names for the runtime. The GNU driver does not forward -LIBPATH: to
    # lld-link and treats bare .lib names as input files, producing "no such
    # file" errors at link time. Disable automatic CUDA runtime linking here
    # and wire it up with full paths in scalars_apply_gpu_backend instead.
    if(WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
        set(CMAKE_CUDA_RUNTIME_LIBRARY None CACHE STRING "CUDA runtime library type" FORCE)
    endif()
    enable_language(CUDA)
    set(SCALARS_HAS_CUDA 1)

    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
       AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native"
    )
        set(_scalars_cuda_archs "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(_scalars_cuda_archs "75;80;86;89;90")
    endif()
    set(SCALARS_CUDA_ARCHITECTURES "${_scalars_cuda_archs}"
        CACHE STRING "CUDA compute capability list (e.g. 75;80;86 for Turing/Ampere/Ada)" FORCE
    )
    unset(_scalars_cuda_archs)
    message(
        STATUS
            "Scalars: CUDA GPU backend enabled (archs: ${SCALARS_CUDA_ARCHITECTURES}, compiler: ${CMAKE_CUDA_COMPILER_ID})"
    )
endif()

# ---- Metal GPU backend --------------------------------------------------------
# Metal has no CMake "language"/unified host+device compiler pass -- MSL kernel
# source (backend/gpu/metal/kernels.metal) is embedded as a string at configure
# time and compiled at runtime via newLibraryWithSource:, so this only needs
# Objective-C++ (for the .mm dispatch glue) plus the Metal/Foundation frameworks.
if(SCALARS_GPU_BACKEND STREQUAL "metal")
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD ${SCALARS_CXX_STANDARD})
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(SCALARS_HAS_METAL 1)

    find_library(SCALARS_METAL_FRAMEWORK Metal REQUIRED)
    find_library(SCALARS_FOUNDATION_FRAMEWORK Foundation REQUIRED)

    set_property(
        DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                                  "${SCALARS_SOURCE_DIR}/include/backend/gpu/metal/kernels.metal"
    )
    file(READ "${SCALARS_SOURCE_DIR}/include/backend/gpu/metal/kernels.metal"
         SCALARS_METAL_KERNEL_SOURCE
    )
    configure_file(
        "${SCALARS_SOURCE_DIR}/include/backend/gpu/metal/metal_kernels_source.h.in"
        "${SCALARS_BINARY_DIR}/include/backend/gpu/metal/metal_kernels_source.h" @ONLY
    )
    message(STATUS "Scalars: Metal GPU backend enabled (framework: ${SCALARS_METAL_FRAMEWORK})")
endif()

set(SCALARS_HIP_VERSION "${_scalars_hip_version}")
unset(_scalars_hip_version)

# scalars_apply_gpu_backend(<target>)
#
# Applies SCALARS_HAS_CUDA/HIP/METAL, SCALARS_USE_HIP/SCALARS_HIP_VERSION, GPU
# architecture target properties, and (for metal) the extra .mm source and
# framework links to <target>. Must run after add_library(<target> ...).
function(scalars_apply_gpu_backend target)
    target_compile_definitions(
        ${target} PUBLIC SCALARS_HAS_CUDA=${SCALARS_HAS_CUDA} SCALARS_HAS_HIP=${SCALARS_HAS_HIP}
                         SCALARS_HAS_METAL=${SCALARS_HAS_METAL}
    )

    if(SCALARS_GPU_BACKEND STREQUAL "hip")
        target_compile_definitions(
            ${target} PUBLIC SCALARS_USE_HIP SCALARS_HIP_VERSION=${SCALARS_HIP_VERSION}
        )
        set_target_properties(
            ${target} PROPERTIES HIP_STANDARD ${SCALARS_CXX_STANDARD}
                                 HIP_ARCHITECTURES "${SCALARS_GPU_ARCHITECTURES}"
        )
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)
        if(TARGET hip::host)
            target_link_libraries(${target} PUBLIC hip::host)
        endif()
        if(TARGET hip::device)
            target_link_libraries(${target} PRIVATE hip::device)
        endif()
    elseif(SCALARS_GPU_BACKEND STREQUAL "cuda")
        set_target_properties(
            ${target}
            PROPERTIES CUDA_STANDARD ${SCALARS_CXX_STANDARD}
                       CUDA_STANDARD_REQUIRED ON
                       CUDA_ARCHITECTURES "${SCALARS_CUDA_ARCHITECTURES}"
                       CUDA_SEPARABLE_COMPILATION OFF
        )
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math>)
            target_compile_options(
                ${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++${SCALARS_CXX_STANDARD}>
            )
        elseif(MSVC)
            # cl.exe's traditional preprocessor trips a hard error in CCCL's
            # headers (cuda/std/__cccl/preprocessor.h); nvcc's MSVC host needs
            # the standard-conforming preprocessor instead. -fPIC is a Unix
            # concept and is not passed to the MSVC host.
            target_compile_options(
                ${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math
                                  -Xcompiler=/Zc:preprocessor>
            )
        else()
            target_compile_options(
                ${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-fPIC>
            )
        endif()
        # CMAKE_CUDA_RUNTIME_LIBRARY was set to None above for this case to
        # prevent MSVC-style -LIBPATH:/-bare-.lib link flags that the GNU driver
        # does not forward to lld-link. Link the CUDA runtime explicitly here
        # using full paths, which clang++ accepts alongside other .lib objects.
        if(WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
           AND CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES)
            list(GET CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES 0 _scalars_cuda_lib_dir)
            foreach(_scalars_cuda_lib IN ITEMS cudart cudadevrt)
                if(EXISTS "${_scalars_cuda_lib_dir}/${_scalars_cuda_lib}.lib")
                    target_link_libraries(
                        ${target} PRIVATE "${_scalars_cuda_lib_dir}/${_scalars_cuda_lib}.lib"
                    )
                endif()
            endforeach()
            unset(_scalars_cuda_lib_dir)
        endif()
    elseif(SCALARS_GPU_BACKEND STREQUAL "metal")
        target_sources(
            ${target} PRIVATE "${SCALARS_SOURCE_DIR}/include/backend/gpu/metal/metal_dispatch.mm"
        )
        set_source_files_properties(
            "${SCALARS_SOURCE_DIR}/include/backend/gpu/metal/metal_dispatch.mm"
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc"
        )
        target_include_directories(
            ${target} PUBLIC $<BUILD_INTERFACE:${SCALARS_BINARY_DIR}/include>
        )
        target_link_libraries(
            ${target} PUBLIC "${SCALARS_METAL_FRAMEWORK}" "${SCALARS_FOUNDATION_FRAMEWORK}"
        )
    endif()
endfunction()
