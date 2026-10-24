cmake_minimum_required(VERSION 3.24)
project(ScalarsInstallConsumer LANGUAGES CXX)

find_package(Scalars CONFIG REQUIRED)

add_executable(install_consumer main.cpp)
target_link_libraries(install_consumer PRIVATE Scalars::Scalars)
add_executable(core_consumer core.cpp)
target_link_libraries(core_consumer PRIVATE Scalars::Core)

enable_testing()
add_test(NAME installed_package COMMAND install_consumer)
add_test(NAME installed_core COMMAND core_consumer)
