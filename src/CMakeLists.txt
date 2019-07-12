# CMake hackery to generate a LLVM bc file from a cpp file.
# This functionality could probably be done in a better way, but this seems to work
# for now...
function (generate_bc test_name)
  # Trick CMake into running a compile-like command on the input file.
  # The output file will end in a .o because CMake thinks compilation should produce object files.
  # But it really has some other contents.
  add_library(${test_name}_bcobj OBJECT ${test_name}_bcgen.cpp)
  target_include_directories(${test_name}_bcobj PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(${test_name}_bcobj PRIVATE -O1 -emit-llvm -ftapir=none -mllvm -enable-tapir-loop-stripmine=false -DNDEBUG)

  # Add custom command / target to copy the object file to our output .bc file.
  add_custom_command(OUTPUT ${test_name}.bc
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${test_name}_bcobj> ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.bc)
  add_custom_target(${test_name}_bc ALL DEPENDS ${test_name}_bcobj ${test_name}.bc)
endfunction()  

generate_bc(matmul)
generate_bc(conv2d)

# Generate shared library
add_library(cilknn SHARED cilkdnn.cpp)

target_include_directories(cilknn PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS})

target_link_libraries(cilknn PUBLIC
   ${Boost_LIBRARIES}
   ${PYTHON_LIBRARIES})

# Don't append 'lib' prefix to the .so for Python 
set_target_properties( cilknn PROPERTIES PREFIX "")
