cmake_minimum_required(VERSION 3.5)

project(cilkdnn VERSION 0.1 LANGUAGES CXX)

# Find external library dependencies using default CMake macros
#find_package(PythonLibs 2.7 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(Boost 1.63 REQUIRED COMPONENTS system python310 numpy310)
include(FindPythonInterp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopencilk -ffast-math -Wall -march=native -g -gdwarf-4")

# Generate output files we want to export.
add_subdirectory(src)

# Test code
enable_testing()
add_subdirectory(test)
 
