cmake_minimum_required(VERSION 3.20)

project(MandelbrotSet LANGUAGES CXX CUDA)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")


# C++ standard for host code
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C++ standard for CUDA host compiler
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(SFML CONFIG REQUIRED COMPONENTS Graphics Window System)

add_executable(sfml_test kernel.cu)

target_link_libraries(sfml_test
    PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
)

# This is CRITICAL
set_target_properties(sfml_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
