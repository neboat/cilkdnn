cmake_minimum_required(VERSION 3.5)

project(cilkdnn VERSION 0.1 LANGUAGES CXX)

# Find external library dependencies using default CMake macros
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost 1.63 REQUIRED COMPONENTS system python numpy)
include(FindPythonInterp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fcilkplus -ffast-math -Wall -mavx")

# Generate output files we want to export.
add_subdirectory(src)

# Test code
enable_testing()
add_subdirectory(test)
 
