# CMake hackery to generate a LLVM bc file from a cpp file.
# This functionality could probably be done in a better way,
# but this seems to work for now...
function (generate_bc test_name)
  # Add custom command to generate bc file.
  configure_file(${test_name}_bcgen.cpp ${test_name}_bcgen.cpp COPYONLY)
  set(orig_cflags ${CMAKE_CXX_FLAGS})
  separate_arguments(orig_cflags)
  set(bcgen_cflags -O1 -emit-llvm -ftapir=none -mllvm -enable-tapir-loop-stripmine=false -DNDEBUG ${orig_cflags})
  set(bcgen_args -c ${test_name}_bcgen.cpp ${bcgen_cflags} -I ${PROJECT_SOURCE_DIR}/include -o ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.bc)
  add_custom_command(OUTPUT ${test_name}.bc
    COMMAND VERBATIM ${CMAKE_CXX_COMPILER} ${bcgen_args})
  add_custom_target(${test_name}_bc ALL DEPENDS ${test_name}_bcgen.cpp ${test_name}.bc)
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
