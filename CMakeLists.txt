cmake_minimum_required(VERSION 3.5)

project(cilkdnn VERSION 0.1 LANGUAGES CXX)

# Find external library dependencies using default CMake macros
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS system python)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fcilkplus -ffast-math -Wall -mavx")

add_library(cilkdnn SHARED src/cilkdnn.cpp)

target_include_directories(cilkdnn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS})

## TBD: Do we need these lines?  I would think so,
## but we don't have tests that require it yet.
#target_link_libraries(cilkdnn PUBLIC
#   ${Boost_LIBRARIES}
#   ${PYTHON_LIBRARIES})

enable_testing()
add_subdirectory(test)
 
