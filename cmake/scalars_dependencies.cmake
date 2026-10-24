# GoogleTest resolution for standalone and embedded tests, following LinearAlgebra.
include_guard(GLOBAL)

set(SCALARS_THIRD_PARTY_DIR "${SCALARS_SOURCE_DIR}/ThirdParty"
    CACHE PATH "Root of Scalars' bundled third-party sources"
)

function(scalars_setup_gtest)
    if(TARGET Gtest::gtest AND TARGET Gtest::gtest_main)
        return()
    endif()
    if(NOT TARGET GTest::gtest_main AND NOT TARGET gtest_main)
        if(COMMAND xsigma_add_googletest)
            xsigma_add_googletest()
        else()
            # Reuse an installed package before downloading a test-only dependency.
            find_package(GTest CONFIG QUIET)
        endif()
    endif()
    if(NOT TARGET GTest::gtest_main AND NOT TARGET gtest_main)
        set(INSTALL_GTEST OFF)
        set(BUILD_GMOCK OFF)
        set(gtest_force_shared_crt ON)
        if(EXISTS "${SCALARS_THIRD_PARTY_DIR}/googletest/CMakeLists.txt")
            add_subdirectory(
                "${SCALARS_THIRD_PARTY_DIR}/googletest"
                "${SCALARS_BINARY_DIR}/ThirdParty/googletest_build" EXCLUDE_FROM_ALL
            )
        else()
            include(FetchContent)
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.18.0
                GIT_SHALLOW TRUE
            )
            FetchContent_MakeAvailable(googletest)
        endif()
    endif()
    foreach(_name IN ITEMS gtest gtest_main)
        if(TARGET GTest::${_name})
            set(_target GTest::${_name})
        elseif(TARGET ${_name})
            set(_target ${_name})
        else()
            message(FATAL_ERROR "GoogleTest must provide ${_name}")
        endif()
        if(NOT TARGET Gtest::${_name})
            add_library(Gtest::${_name} INTERFACE IMPORTED)
            set_target_properties(Gtest::${_name} PROPERTIES INTERFACE_LINK_LIBRARIES "${_target}")
        endif()
    endforeach()
endfunction()
