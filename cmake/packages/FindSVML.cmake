# FindSVML.cmake — Locate Intel SVML (Short Vector Math Library) ThirdParty binaries
#
# Resolves the ThirdParty/svml directory relative to this find module:
#   <root>/Cmake/packages/FindSVML.cmake  →  <root>/ThirdParty/svml
#
# Imported targets (created only when SVML_FOUND):
#   SVML::SVML    — main SVML library (Windows: import lib + DLL; Unix: .so)
#   SVML::IRC     — IRC companion runtime         (Unix only)
#   SVML::INTLC   — INTLC companion runtime       (Unix only)
#
# Output variables:
#   SVML_FOUND               — TRUE when the core SVML library is located
#   SVML_LIBRARIES           — all link-time library paths
#   SVML_RUNTIME_LIBRARIES   — runtime files (DLLs / .so) to stage in the build tree

# ---------------------------------------------------------------------------
# Locate the ThirdParty/svml root from this file's position
# ---------------------------------------------------------------------------
get_filename_component(_svml_mod_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_svml_tp_root "${_svml_mod_dir}/../.." ABSOLUTE)
set(_svml_tp "${_svml_tp_root}/ThirdParty/svml")

# This module is also installed to <prefix>/lib/cmake/Scalars/Modules (see
# scalars_config.cmake.in), so find_dependency(SVML) can re-find SVML for an
# installed package's consumers. In that layout ThirdParty/svml does not
# exist; simd.cmake instead staged the runtime libraries (and, on Windows,
# the import library) flat into the install prefix's lib directory. Search
# both layouts.
get_filename_component(_svml_installed_libdir "${_svml_mod_dir}/../../.." ABSOLUTE)
get_filename_component(_svml_installed_bindir "${_svml_installed_libdir}/../bin" ABSOLUTE)

set(SVML_LIBRARIES)
set(SVML_RUNTIME_LIBRARIES)

# ---------------------------------------------------------------------------
# Windows — link via import library; stage the DLL at runtime
# ---------------------------------------------------------------------------
if(WIN32)
  set(_svml_lib_dir "${_svml_tp}/windows/lib")
  set(_svml_bin_dir "${_svml_tp}/windows/svml")

  find_library(SVML_LIBRARY_SVML
    NAMES svml_dispmd
    PATHS "${_svml_lib_dir}" "${_svml_installed_libdir}"
    NO_DEFAULT_PATH
  )
  find_file(SVML_DLL_SVML
    NAMES svml_dispmd.dll
    PATHS "${_svml_bin_dir}" "${_svml_installed_bindir}"
    NO_DEFAULT_PATH
  )

  if(SVML_LIBRARY_SVML)
    list(APPEND SVML_LIBRARIES "${SVML_LIBRARY_SVML}")
  endif()
  if(SVML_DLL_SVML)
    list(APPEND SVML_RUNTIME_LIBRARIES "${SVML_DLL_SVML}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Unix — shared objects; all three must be staged at runtime
# ---------------------------------------------------------------------------
if(UNIX)
  set(_svml_lib_dir "${_svml_tp}/unix/lib")

  find_library(SVML_LIBRARY_SVML   NAMES svml
    PATHS "${_svml_lib_dir}" "${_svml_installed_libdir}" NO_DEFAULT_PATH)
  find_library(SVML_LIBRARY_IRC    NAMES irc
    PATHS "${_svml_lib_dir}" "${_svml_installed_libdir}" NO_DEFAULT_PATH)
  # libintlc.so is a GNU ld script ("INPUT (libintlc.so.5)"). mold does not
  # search the script's directory for that name and fails the link with
  # "library not found: libintlc.so.5". Link the real shared object. Its SONAME
  # is libintlc.so.5, which is also libsvml's DT_NEEDED entry. A cached hit on
  # the script (from an older configure) is replaced; an explicit ELF path is kept.
  set(_svml_intlc_real "${_svml_lib_dir}/libintlc.so.5")
  if(NOT EXISTS "${_svml_intlc_real}")
    set(_svml_intlc_real "${_svml_installed_libdir}/libintlc.so.5")
  endif()
  set(_svml_intlc_head "")
  if(SVML_LIBRARY_INTLC AND EXISTS "${SVML_LIBRARY_INTLC}")
    file(READ "${SVML_LIBRARY_INTLC}" _svml_intlc_head LIMIT 32)
  endif()
  if(NOT SVML_LIBRARY_INTLC OR _svml_intlc_head MATCHES "^[ \t]*INPUT")
    if(EXISTS "${_svml_intlc_real}")
      set(SVML_LIBRARY_INTLC "${_svml_intlc_real}"
          CACHE FILEPATH "Path to the Intel libintlc runtime" FORCE)
    else()
      find_library(SVML_LIBRARY_INTLC NAMES intlc
        PATHS "${_svml_lib_dir}" "${_svml_installed_libdir}" NO_DEFAULT_PATH)
    endif()
  endif()
  unset(_svml_intlc_head)
  unset(_svml_intlc_real)

  foreach(_svml_var IN ITEMS SVML_LIBRARY_SVML SVML_LIBRARY_IRC SVML_LIBRARY_INTLC)
    if(${_svml_var})
      list(APPEND SVML_LIBRARIES        "${${_svml_var}}")
      list(APPEND SVML_RUNTIME_LIBRARIES "${${_svml_var}}")
    endif()
  endforeach()
  unset(_svml_var)

  # Stage libintlc.so.5 for the dynamic loader. libsvml's DT_NEEDED names that
  # SONAME, and a linker script is not a runtime dependency. Skip the append
  # when the loop above already recorded the real shared object.
  if(EXISTS "${_svml_lib_dir}/libintlc.so.5"
     AND NOT "${_svml_lib_dir}/libintlc.so.5" IN_LIST SVML_RUNTIME_LIBRARIES)
    list(APPEND SVML_RUNTIME_LIBRARIES "${_svml_lib_dir}/libintlc.so.5")
  endif()
endif()

# ---------------------------------------------------------------------------
# Standard args — only SVML_LIBRARY_SVML is strictly required
# ---------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SVML DEFAULT_MSG SVML_LIBRARY_SVML)

# ---------------------------------------------------------------------------
# Create IMPORTED targets when the package is found
# ---------------------------------------------------------------------------
if(SVML_FOUND)
  if(NOT TARGET SVML::SVML)
    if(WIN32)
      # SHARED IMPORTED lets CMake track both the import lib and the DLL
      add_library(SVML::SVML SHARED IMPORTED GLOBAL)
      set_target_properties(SVML::SVML PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_IMPLIB   "${SVML_LIBRARY_SVML}"
        IMPORTED_LOCATION "${SVML_DLL_SVML}"
      )
    else()
      add_library(SVML::SVML UNKNOWN IMPORTED GLOBAL)
      set_target_properties(SVML::SVML PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${SVML_LIBRARY_SVML}"
      )
    endif()
  endif()

  if(UNIX)
    if(SVML_LIBRARY_IRC AND NOT TARGET SVML::IRC)
      add_library(SVML::IRC UNKNOWN IMPORTED GLOBAL)
      set_target_properties(SVML::IRC PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${SVML_LIBRARY_IRC}"
      )
    endif()

    if(SVML_LIBRARY_INTLC AND NOT TARGET SVML::INTLC)
      add_library(SVML::INTLC UNKNOWN IMPORTED GLOBAL)
      set_target_properties(SVML::INTLC PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${SVML_LIBRARY_INTLC}"
      )
    endif()
  endif()
endif()

mark_as_advanced(SVML_LIBRARY_SVML SVML_LIBRARY_IRC SVML_LIBRARY_INTLC SVML_DLL_SVML)

unset(_svml_mod_dir)
unset(_svml_tp_root)
unset(_svml_tp)
unset(_svml_lib_dir)
unset(_svml_bin_dir)
unset(_svml_installed_libdir)
unset(_svml_installed_bindir)
