find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME ScalarsSetupTests COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s
                                            "${CMAKE_CURRENT_SOURCE_DIR}" -p test_setup.py
             WORKING_DIRECTORY "${SCALARS_SOURCE_DIR}"
    )
    set_tests_properties(ScalarsSetupTests PROPERTIES TIMEOUT 30 LABELS "Scalars;CMake")
endif()
