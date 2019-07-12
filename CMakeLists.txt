cmake_minimum_required(VERSION 3.5)

project(cilkdnn VERSION 0.1 LANGUAGES CXX)

# Find external library dependencies using default CMake macros
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS system python numpy)
include(FindPythonInterp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fcilkplus -ffast-math -Wall -mavx")

add_library(cilknn SHARED src/cilkdnn.cpp)

target_include_directories(cilknn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS})

target_link_libraries(cilknn PUBLIC
   ${Boost_LIBRARIES}
   ${PYTHON_LIBRARIES})

# Don't append 'lib' prefix to the .so for Python 
set_target_properties( cilknn PROPERTIES PREFIX "")

enable_testing()
add_subdirectory(test)
 
