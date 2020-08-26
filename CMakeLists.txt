# Author: Arjun Ramaswami
cmake_minimum_required (VERSION 3.11)

project(FPGAExperiments)

set(CMAKE_C_STANDARD 11)

# Download build dependencies
add_subdirectory("${CMAKE_SOURCE_DIR}/extern" "${CMAKE_BINARY_DIR}/extern")

# Include hlslib in CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${hlslib_SOURCE_DIR}/cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Find hlslib Intel OpenCL kernels
find_package(IntelFPGAOpenCL REQUIRED)

# Add sub directories
add_subdirectory(api)
add_subdirectory(kernels)
add_subdirectory(expms)
