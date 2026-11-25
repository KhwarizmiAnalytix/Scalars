# CPU SIMD backend selection for Scalars. Compiler probes live in utils.cmake,
# ported from Vectorization's include/Cmake/utils.cmake.
#
# SCALARS_CPU_BACKEND (cache: no|sse|avx|avx2|avx512|neon|sve) selects the CPU SIMD
# backend used by include/backend/cpu/*. C++ preprocessor defines:
#   sse -> SCALARS_HAS_SSE=1; avx/avx2 -> SCALARS_HAS_AVX=1; avx2 -> SCALARS_HAS_AVX2=1;
#   avx512 -> SCALARS_HAS_AVX512=1; neon -> SCALARS_HAS_NEON=1; sve -> SCALARS_HAS_SVE=1.
# SCALARS_CPU_CAPABILITY_AVX2 / SCALARS_CPU_CAPABILITY_AVX512 are defined (ifdef-style,
# no value) exactly when backend is avx2 / avx512 -- these are what
# include/types/half.h and include/common/vec_half.h test via #if defined(...).
include_guard(GLOBAL)

# Host-aware default: x86 family -> avx2, aarch64 -> neon, anything else -> no.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _scalars_simd_proc)
if(_scalars_simd_proc MATCHES "^(x86_64|amd64|i686|i386)$")
    set(_scalars_simd_is_x86 TRUE)
    set(_scalars_simd_is_aarch64 FALSE)
    set(_scalars_cpu_backend_default "avx2")
elseif(_scalars_simd_proc MATCHES "^(aarch64|arm64)$")
    set(_scalars_simd_is_x86 FALSE)
    set(_scalars_simd_is_aarch64 TRUE)
    set(_scalars_cpu_backend_default "neon")
else()
    set(_scalars_simd_is_x86 FALSE)
    set(_scalars_simd_is_aarch64 FALSE)
    set(_scalars_cpu_backend_default "no")
endif()

set(SCALARS_CPU_BACKEND "${_scalars_cpu_backend_default}"
    CACHE STRING "CPU SIMD backend for Scalars. Options: no, sse, avx, avx2, avx512, neon, sve"
)
set_property(
    CACHE SCALARS_CPU_BACKEND
    PROPERTY STRINGS
             no
             sse
             avx
             avx2
             avx512
             neon
             sve
)
if(NOT SCALARS_CPU_BACKEND MATCHES "^(no|sse|avx|avx2|avx512|neon|sve)$")
    message(FATAL_ERROR "SCALARS_CPU_BACKEND must be one of: no, sse, avx, avx2, avx512, neon, sve "
                        "(got '${SCALARS_CPU_BACKEND}')"
    )
endif()

# SCALARS_MATH_BACKEND (cache: auto|svml|sleef|accelerate) selects the
# vector math library behind the CPU packets' transcendental functions:
#   auto       -- the built-in vectorized port of Eigen's packet math
#                 (include/backend/cpu/detail/vector_math.h), on every CPU
#                 tier, even when compiler SVML intrinsics are available.
#   svml       -- bundled Intel SVML from ThirdParty/svml (x86 only).
#   sleef      -- SLEEF built from ThirdParty/sleef.
#   accelerate -- Apple Accelerate vForce (Apple only).
# A library that cannot be used falls back (with a warning) to auto.
# SCALARS_MATH_BACKEND_RESOLVED holds the provider actually used: one of
# svml, sleef, accelerate, or native (the built-in port).
set(SCALARS_MATH_BACKEND "auto"
    CACHE STRING "CPU vector math backend for Scalars. Options: auto, svml, sleef, accelerate"
)
# "native" was folded into "auto". Migrate a stale cache entry instead of failing.
if(SCALARS_MATH_BACKEND STREQUAL "native")
    message(WARNING "SCALARS_MATH_BACKEND=native was removed; auto now selects the built-in "
                    "vector math. Using auto."
    )
    set(SCALARS_MATH_BACKEND "auto"
        CACHE STRING "CPU vector math backend for Scalars. Options: auto, svml, sleef, accelerate"
              FORCE
    )
endif()
set_property(
    CACHE SCALARS_MATH_BACKEND
    PROPERTY STRINGS
             auto
             svml
             sleef
             accelerate
)
if(NOT SCALARS_MATH_BACKEND MATCHES "^(auto|svml|sleef|accelerate)$")
    message(FATAL_ERROR "SCALARS_MATH_BACKEND must be one of: auto, svml, sleef, accelerate "
                        "(got '${SCALARS_MATH_BACKEND}')"
    )
endif()
# The per-library switches this replaces. Drop them so a stale cache entry
# (including one auto-enabled by an older configure) cannot shadow the enum.
foreach(_scalars_legacy_math IN ITEMS SVML SLEEF ACCELERATE NATIVE_MATH)
    if(DEFINED SCALARS_ENABLE_${_scalars_legacy_math})
        if(SCALARS_ENABLE_${_scalars_legacy_math})
            message(
                WARNING
                    "SCALARS_ENABLE_${_scalars_legacy_math} was replaced by SCALARS_MATH_BACKEND "
                    "and is ignored. Set -DSCALARS_MATH_BACKEND=<auto|svml|sleef|accelerate>."
            )
        endif()
        unset(SCALARS_ENABLE_${_scalars_legacy_math} CACHE)
        unset(SCALARS_ENABLE_${_scalars_legacy_math})
    endif()
endforeach()
unset(_scalars_legacy_math)
if(SCALARS_MATH_BACKEND STREQUAL "auto")
    set(SCALARS_MATH_BACKEND_RESOLVED "native")
else()
    set(SCALARS_MATH_BACKEND_RESOLVED "${SCALARS_MATH_BACKEND}")
endif()

option(SCALARS_ENABLE_MKL "Enable Intel MKL VML vector math backend (x86 only, static link)" OFF)
mark_as_advanced(SCALARS_ENABLE_MKL)
option(USE_NATIVE_ARCH "Prefer host CPU instruction tuning (-march=native, Clang/GCC only)" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/packages")

# Install the custom find modules so an installed package's ScalarsConfig.cmake
# can call find_dependency(SVML) / find_dependency(MKL) to re-find them (see
# scalars_config.cmake.in); otherwise a consumer's find_package(Scalars) has
# no way to recreate the SVML::SVML / MKL::MKL imported targets that Scalars
# links against PUBLIC.
install(FILES "${CMAKE_CURRENT_LIST_DIR}/packages/FindSVML.cmake"
              "${CMAKE_CURRENT_LIST_DIR}/packages/FindMKL.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Scalars/Modules"
)

set(SCALARS_HAS_MKL 0)
if(SCALARS_ENABLE_MKL)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _scalars_mkl_proc)
    if(NOT _scalars_mkl_proc MATCHES "^(x86_64|amd64|i686|i386)$")
        message(WARNING "SCALARS_ENABLE_MKL=ON is only supported on x86_64; disabling.")
        set(SCALARS_ENABLE_MKL OFF
            CACHE BOOL "Enable Intel MKL VML vector math backend (x86 only, static link)" FORCE
        )
    else()
        set(MKL_THREADING "SEQ" CACHE STRING "MKL threading flavour used by Scalars" FORCE)
        set(ENABLE_STATIC_MKL OFF CACHE BOOL "Link MKL statically for Scalars" FORCE)
        find_package(MKL QUIET)
        if(MKL_FOUND)
            set(SCALARS_HAS_MKL 1)
            message(
                STATUS
                    "Scalars: MKL VML found (version ${MKL_VERSION}). Include: ${MKL_INCLUDE_DIR}"
            )
            if(NOT TARGET MKL::MKL)
                add_library(MKL::MKL INTERFACE IMPORTED GLOBAL)
                target_include_directories(MKL::MKL INTERFACE "${MKL_INCLUDE_DIR}")
                # --start-group / --end-group resolves circular references among the
                # static MKL archives. Shared libraries do not need it.
                if(UNIX AND NOT APPLE AND ENABLE_STATIC_MKL)
                    target_link_libraries(
                        MKL::MKL INTERFACE -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group
                                           ${CMAKE_DL_LIBS}
                    )
                else()
                    target_link_libraries(MKL::MKL INTERFACE ${MKL_LIBRARIES} ${CMAKE_DL_LIBS})
                endif()
            endif()
        else()
            message(
                WARNING "SCALARS_ENABLE_MKL=ON but MKL was not found -- disabling MKL VML backend. "
                        "Set MKLROOT or INTEL_MKL_DIR to your MKL installation root."
            )
            set(SCALARS_ENABLE_MKL OFF
                CACHE BOOL "Enable Intel MKL VML vector math backend (x86 only, static link)" FORCE
            )
        endif()
    endif()
    unset(_scalars_mkl_proc)
endif()

# SVE vector register width. include/backend/cpu/sve/* only supports 128 bits.
# Set before utils.cmake, which passes it to -msve-vector-bits.
set(SCALARS_SVE_VECTOR_BITS "128" CACHE STRING "SVE vector register width in bits: 128, 256, 512")
set_property(CACHE SCALARS_SVE_VECTOR_BITS PROPERTY STRINGS 128 256 512)
if(NOT SCALARS_SVE_VECTOR_BITS MATCHES "^(128|256|512)$")
    message(
        FATAL_ERROR
            "SCALARS_SVE_VECTOR_BITS must be 128, 256, or 512 (got '${SCALARS_SVE_VECTOR_BITS}')"
    )
endif()

include("${CMAKE_CURRENT_LIST_DIR}/utils.cmake")

# Settle SCALARS_MATH_BACKEND_RESOLVED. SVML requires the imported target, not
# just the request: utils.cmake only probes for it on x86 and only warns when
# the ThirdParty binaries are missing.
if(SCALARS_MATH_BACKEND_RESOLVED STREQUAL "svml" AND NOT TARGET SVML::SVML)
    if(NOT _scalars_simd_is_x86)
        message(WARNING "SCALARS_MATH_BACKEND=svml is only supported on x86; "
                        "falling back to the built-in vector math (auto)."
        )
    endif()
    set(SCALARS_MATH_BACKEND_RESOLVED "")
endif()
# Apple Accelerate vForce is a system framework -- no ThirdParty build needed,
# just a link flag applied in scalars_apply_simd_compile_definitions() below.
if(SCALARS_MATH_BACKEND_RESOLVED STREQUAL "accelerate" AND NOT APPLE)
    message(WARNING "SCALARS_MATH_BACKEND=accelerate is only supported on Apple platforms; "
                    "falling back to the built-in vector math (auto)."
    )
    set(SCALARS_MATH_BACKEND_RESOLVED "")
endif()
# Build/link ThirdParty SLEEF (vendored submodule) when requested. include/backend/cpu/{sse,avx,avx512,neon}/svml.h and
# backend/cpu/x86/math_fallback.h branch on SCALARS_HAS_SLEEF.
if(SCALARS_MATH_BACKEND_RESOLVED STREQUAL "sleef")
    include("${CMAKE_CURRENT_LIST_DIR}/scalars_dependencies.cmake")
    scalars_setup_sleef()
    if(NOT TARGET Sleef::sleef)
        set(SCALARS_MATH_BACKEND_RESOLVED "")
    endif()
endif()
# No usable library: use the built-in port, exactly as auto does.
if(SCALARS_MATH_BACKEND_RESOLVED STREQUAL "")
    set(SCALARS_MATH_BACKEND_RESOLVED "native")
endif()

set(SCALARS_HAS_SVML 0)
set(SCALARS_HAS_SLEEF 0)
set(SCALARS_HAS_ACCELERATE 0)
set(SCALARS_HAS_NATIVE_MATH 0)
if(SCALARS_MATH_BACKEND_RESOLVED STREQUAL "svml")
    set(SCALARS_HAS_SVML 1)
elseif(SCALARS_MATH_BACKEND_RESOLVED STREQUAL "sleef")
    set(SCALARS_HAS_SLEEF 1)
elseif(SCALARS_MATH_BACKEND_RESOLVED STREQUAL "accelerate")
    set(SCALARS_HAS_ACCELERATE 1)
elseif(SCALARS_MATH_BACKEND_RESOLVED STREQUAL "native")
    set(SCALARS_HAS_NATIVE_MATH 1)
endif()
message(STATUS "Scalars: vector math backend: ${SCALARS_MATH_BACKEND_RESOLVED} "
               "(SCALARS_MATH_BACKEND=${SCALARS_MATH_BACKEND})."
)

if(SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS)
    set(SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS 1)
else()
    set(SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS 0)
endif()

if(NOT SCALARS_CPU_BACKEND STREQUAL "no"
   AND NOT _scalars_has_sse
   AND NOT _scalars_has_avx
   AND NOT _scalars_has_avx2
   AND NOT _scalars_has_avx512
   AND NOT _scalars_has_neon
   AND NOT _scalars_has_sve
)
    message(
        WARNING
            "SCALARS_CPU_BACKEND '${SCALARS_CPU_BACKEND}' is not supported by this compiler/architecture. "
            "Falling back to SCALARS_CPU_BACKEND=no."
    )
    set(SCALARS_CPU_BACKEND "no"
        CACHE STRING "CPU SIMD backend for Scalars. Options: no, sse, avx, avx2, avx512, neon, sve"
              FORCE
    )
endif()

set(SCALARS_HAS_SSE ${_scalars_has_sse})
set(SCALARS_HAS_AVX ${_scalars_has_avx})
set(SCALARS_HAS_AVX2 ${_scalars_has_avx2})
set(SCALARS_HAS_AVX512 ${_scalars_has_avx512})
set(SCALARS_HAS_NEON ${_scalars_has_neon})
set(SCALARS_HAS_SVE ${_scalars_has_sve})

# 1 when any CPU SIMD backend/cpu/* tree is active (GPU does not affect this) --
# gates backend/simd.h's system-intrinsic-header include block.
if(_scalars_has_sse
   OR _scalars_has_avx
   OR _scalars_has_avx2
   OR _scalars_has_avx512
   OR _scalars_has_neon
   OR _scalars_has_sve
)
    set(SCALARS_VECTORIZED 1)
else()
    set(SCALARS_VECTORIZED 0)
endif()

unset(_scalars_simd_proc)
unset(_scalars_simd_is_x86)
unset(_scalars_simd_is_aarch64)
unset(_scalars_cpu_backend_default)
unset(_scalars_has_sse)
unset(_scalars_has_avx)
unset(_scalars_has_avx2)
unset(_scalars_has_avx512)
unset(_scalars_has_neon)
unset(_scalars_has_sve)

# scalars_apply_simd_compile_definitions(<target>)
#
# Applies the one-hot SCALARS_HAS_* defines, the ifdef-style
# SCALARS_CPU_CAPABILITY_AVX2/AVX512 defines, and SCALARS_SIMD_COMPILER_FLAGS
# as PUBLIC settings on <target>.
function(scalars_apply_simd_compile_definitions target)
    target_compile_definitions(
        ${target}
        PUBLIC
            SCALARS_HAS_SSE=${SCALARS_HAS_SSE}
            SCALARS_HAS_AVX=${SCALARS_HAS_AVX}
            SCALARS_HAS_AVX2=${SCALARS_HAS_AVX2}
            SCALARS_HAS_AVX512=${SCALARS_HAS_AVX512}
            SCALARS_HAS_NEON=${SCALARS_HAS_NEON}
            SCALARS_HAS_SVE=${SCALARS_HAS_SVE}
            SCALARS_HAS_SVML=${SCALARS_HAS_SVML}
            SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS=${SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS}
            SCALARS_HAS_MKL=${SCALARS_HAS_MKL}
            SCALARS_HAS_SLEEF=${SCALARS_HAS_SLEEF}
            SCALARS_HAS_ACCELERATE=${SCALARS_HAS_ACCELERATE}
            SCALARS_HAS_NATIVE_MATH=${SCALARS_HAS_NATIVE_MATH}
            SCALARS_VECTORIZED=${SCALARS_VECTORIZED}
            SCALARS_PACKET_SIZE=${SCALARS_PACKET_SIZE}
    )
    if(SCALARS_CPU_BACKEND STREQUAL "avx2")
        target_compile_definitions(${target} PUBLIC SCALARS_CPU_CAPABILITY_AVX2)
    endif()
    if(SCALARS_CPU_BACKEND STREQUAL "avx512")
        target_compile_definitions(${target} PUBLIC SCALARS_CPU_CAPABILITY_AVX512)
    endif()
    # Link ThirdParty SVML for this build and stage its runtimes next to the
    # output library. SVML::SVML is linked PUBLIC, unrestricted -- a consumer
    # of the installed package also needs it: SVML calls are emitted by
    # inline headers (include/backend/cpu/*/svml.h), so they end up compiled
    # directly into the consumer's own translation units, not just inside
    # libScalars itself. scalars_config.cmake.in calls find_dependency(SVML)
    # (via the installed FindSVML.cmake module) before including
    # ScalarsTargets.cmake, which is what makes this resolve for a consumer.
    # libintlc.so.5 is a direct DT_NEEDED of libsvml; Linux ld.so does not use
    # an executable RUNPATH to find a shared library's own dependencies, so
    # the library searches $ORIGIN, where the installed copies live.
    if(SCALARS_HAS_SVML)
        if(UNIX AND NOT APPLE)
            # Nothing in Scalars calls libirc/libintlc symbols directly, so the
            # linker's default --as-needed drops them from this target's own
            # DT_NEEDED. DT_RUNPATH only resolves the DT_NEEDED entries of the
            # object that carries it, not of that object's dependencies in
            # turn -- so if libintlc.so.5 is only a transitive dependency of
            # libsvml, ld.so has no RUNPATH to find it when libsvml is loaded.
            # Forcing it onto this target's own DT_NEEDED makes ld.so resolve
            # and load it via this target's RUNPATH up front; libsvml's later
            # request for the same soname then reuses the already-loaded copy.
            # The bracketing --no-as-needed/--as-needed flags stay in
            # BUILD_INTERFACE: exported, they would be positional raw linker
            # flags in a consumer's link line, which CMake may reorder or
            # deduplicate, and the trailing --as-needed would apply to every
            # library the consumer links after Scalars.
            target_link_libraries(${target} PUBLIC "$<BUILD_INTERFACE:-Wl$<COMMA>--no-as-needed>")
        endif()
        target_link_libraries(${target} PUBLIC SVML::SVML)
        if(TARGET SVML::IRC)
            target_link_libraries(${target} PUBLIC SVML::IRC)
        endif()
        if(TARGET SVML::INTLC)
            target_link_libraries(${target} PUBLIC SVML::INTLC)
        endif()
        if(UNIX AND NOT APPLE)
            target_link_libraries(${target} PUBLIC "$<BUILD_INTERFACE:-Wl$<COMMA>--as-needed>")
        endif()
        foreach(_svml_rt IN LISTS SVML_RUNTIME_LIBRARIES)
            if(_svml_rt)
                add_custom_command(
                    TARGET ${target}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_svml_rt}"
                            "$<TARGET_FILE_DIR:${target}>"
                    COMMENT "Staging SVML runtime: ${_svml_rt}"
                    VERBATIM
                )
            endif()
        endforeach()
        if(SVML_RUNTIME_LIBRARIES)
            # Windows loads the DLL from the executable directory. Unix loaders
            # find the staged objects next to libScalars via $ORIGIN.
            if(WIN32)
                install(FILES ${SVML_RUNTIME_LIBRARIES} DESTINATION ${CMAKE_INSTALL_BINDIR})
            else()
                install(FILES ${SVML_RUNTIME_LIBRARIES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
            endif()
        endif()
        if(WIN32 AND SVML_LIBRARIES)
            # SVML_RUNTIME_LIBRARIES is only the DLL on Windows (see
            # FindSVML.cmake); the .lib import library a consumer links
            # against is never staged otherwise, so find_dependency(SVML)
            # from an installed ScalarsConfig.cmake would find the DLL but
            # not the import library.
            install(FILES ${SVML_LIBRARIES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()
        if(UNIX AND NOT APPLE)
            set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "\$ORIGIN")
            set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "\$ORIGIN")
            if(SVML_LIBRARY_SVML)
                get_filename_component(_svml_rt_dir "${SVML_LIBRARY_SVML}" DIRECTORY)
                # INTERFACE only: ${target} itself already resolves SVML via its
                # own $ORIGIN rpath plus the staged copy_if_different above, so
                # applying this PUBLIC would additionally bake the build
                # machine's absolute ThirdParty/svml path into *this* target's
                # own RUNPATH -- which then survives into the installed
                # library. In-tree consumers built in the same tree (e.g.
                # ScalarsCxxTests), which have no staged copy of their own,
                # still need it and get it via this INTERFACE propagation.
                target_link_options(
                    ${target} INTERFACE
                    "$<BUILD_INTERFACE:-Wl$<COMMA>-rpath$<COMMA>${_svml_rt_dir}>"
                )
                unset(_svml_rt_dir)
            endif()
        endif()
        unset(_svml_rt)
        message(
            STATUS
                "Scalars: linked SVML::SVML (ThirdParty); runtime files: ${SVML_RUNTIME_LIBRARIES}"
        )
    endif()
    if(SCALARS_ENABLE_MKL)
        if(MKL_FOUND AND TARGET MKL::MKL)
            target_link_libraries(${target} PUBLIC MKL::MKL)
            message(STATUS "Scalars: linked MKL VML via MKL::MKL.")
        else()
            message(FATAL_ERROR "SCALARS_ENABLE_MKL=ON but the MKL::MKL target is not available. "
                                "Set MKLROOT or INTEL_MKL_DIR to your MKL installation root."
            )
        endif()
    endif()
    if(SCALARS_HAS_SLEEF)
        target_link_libraries(${target} PUBLIC Sleef::sleef)
    endif()
    if(SCALARS_HAS_ACCELERATE)
        target_link_libraries(${target} PUBLIC "-framework Accelerate")
        message(STATUS "Scalars: linked Apple Accelerate framework (vForce).")
    endif()
    if(SCALARS_SIMD_COMPILER_FLAGS)
        # Branch on the C++ compiler's flag syntax, which is what utils.cmake
        # probed with, not on MSVC: enable_language(CUDA) with nvcc on Windows
        # sets MSVC even when the C++ compiler is the GNU-style clang driver.
        if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            separate_arguments(
                _scalars_simd_flags_list WINDOWS_COMMAND "${SCALARS_SIMD_COMPILER_FLAGS}"
            )
            # nvcc treats a bare /arch or /D option as another input file and stops
            # with "A single input file is required". -Xcompiler forwards them to
            # cl.exe, so /D__F16C__ still enables F16C half conversion in .cu files.
            foreach(_flag IN LISTS _scalars_simd_flags_list)
                target_compile_options(
                    ${target} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>"
                                     "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_flag}>"
                )
            endforeach()
            unset(_flag)
            set(_scalars_cuda_host_simd_flags "${_scalars_simd_flags_list}")
        elseif(WIN32 AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            # GNU-style C++ compiler (clang) with nvcc on Windows: nvcc's host
            # compiler is always cl.exe, which ignores -mavx2/-mf16c/-mfma
            # (warning D9002), so __AVX2__ stays undefined in the host pass and
            # scalars_macros.h rejects the translation unit. Give the CUDA host
            # pass the cl.exe spelling of the same ISA level instead.
            separate_arguments(
                _scalars_simd_flags_list UNIX_COMMAND "${SCALARS_SIMD_COMPILER_FLAGS}"
            )
            set(_scalars_simd_cuda_host_flags "")
            if(SCALARS_CPU_BACKEND STREQUAL "sse")
                list(APPEND _scalars_simd_cuda_host_flags /arch:SSE2)
            elseif(SCALARS_CPU_BACKEND STREQUAL "avx")
                list(APPEND _scalars_simd_cuda_host_flags /arch:AVX /D__F16C__)
            elseif(SCALARS_CPU_BACKEND STREQUAL "avx2")
                list(APPEND _scalars_simd_cuda_host_flags /arch:AVX2 /D__F16C__)
            elseif(SCALARS_CPU_BACKEND STREQUAL "avx512")
                list(APPEND _scalars_simd_cuda_host_flags /arch:AVX512 /D__F16C__)
            endif()
            if("-mfma" IN_LIST _scalars_simd_flags_list)
                list(APPEND _scalars_simd_cuda_host_flags /D__FMA__)
            endif()
            foreach(_flag IN LISTS _scalars_simd_flags_list)
                target_compile_options(
                    ${target} PUBLIC "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_flag}>"
                )
            endforeach()
            foreach(_flag IN LISTS _scalars_simd_cuda_host_flags)
                target_compile_options(
                    ${target} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>"
                )
            endforeach()
            unset(_flag)
            set(_scalars_cuda_host_simd_flags "${_scalars_simd_cuda_host_flags}")
            unset(_scalars_simd_cuda_host_flags)
        else()
            separate_arguments(
                _scalars_simd_flags_list UNIX_COMMAND "${SCALARS_SIMD_COMPILER_FLAGS}"
            )
            target_compile_options(${target} PUBLIC ${_scalars_simd_flags_list})
            set(_scalars_cuda_host_simd_flags "${_scalars_simd_flags_list}")
        endif()
        unset(_scalars_simd_flags_list)
    endif()
    # Host-pass SIMD flags as the CUDA compiler sees them, for summary.cmake.
    set(SCALARS_CUDA_HOST_SIMD_FLAGS "${_scalars_cuda_host_simd_flags}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED SCALARS_PACKET_SIZE)
    set(SCALARS_PACKET_SIZE
        "1"
        CACHE
            STRING
            "SIMD registers processed per unrolled loop iteration in the ported backend/ kernels (1,2,3,...)"
    )
endif()
if(NOT SCALARS_PACKET_SIZE MATCHES "^[1-9][0-9]*$")
    message(
        FATAL_ERROR
            "SCALARS_PACKET_SIZE must be a positive decimal integer (got '${SCALARS_PACKET_SIZE}')"
    )
endif()

message(STATUS "Scalars: SIMD compiler flags: ${SCALARS_SIMD_COMPILER_FLAGS}")
