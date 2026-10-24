# C++ unit tests, following LinearAlgebra's Testing/Cxx layout.
if(SCALARS_ENABLE_GTEST)
    include("${SCALARS_SOURCE_DIR}/cmake/scalars_dependencies.cmake")
    scalars_setup_gtest()

    configure_file(
        "${SCALARS_SOURCE_DIR}/Testing/scalars_test.h.in"
        "${SCALARS_BINARY_DIR}/Testing/scalars_test.h" @ONLY
    )
    file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cxx"
    )
    add_executable(ScalarsCxxTests ${test_sources})
    target_include_directories(
        ScalarsCxxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/.."
    )
    target_compile_definitions(
        ScalarsCxxTests PRIVATE SCALARS_HAS_GTEST=1 SCALARS_GOOGLE_TEST USE_GTEST _VARIADIC_MAX=10
    )
    target_link_libraries(ScalarsCxxTests PRIVATE Gtest::gtest Gtest::gtest_main Scalars::Scalars)
    scalars_configure_target(ScalarsCxxTests)
    set_target_properties(
        ScalarsCxxTests PROPERTIES FOLDER "Tests" VS_DEBUGGER_WORKING_DIRECTORY
                                                  "$<TARGET_FILE_DIR:ScalarsCxxTests>"
    )
    add_test(NAME ScalarsCxxTests COMMAND ScalarsCxxTests
             WORKING_DIRECTORY "$<TARGET_FILE_DIR:ScalarsCxxTests>"
    )
    set_tests_properties(ScalarsCxxTests PROPERTIES TIMEOUT 120 LABELS "Scalars;Unit")
endif()
