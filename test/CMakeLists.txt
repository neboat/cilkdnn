function (setup_test_exe test_name)
    add_executable(test-${test_name} test-${test_name}.cpp)
    target_include_directories(test-${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(test-${test_name} PRIVATE "-O3")
endfunction()  


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

setup_test_exe(matmul)
add_test(run_test-matmul test-matmul COMMAND test-matmul 100 100 100)
generate_bc(matmul)

setup_test_exe(conv2d)
add_test(run_test-conv2d test-conv2d)
generate_bc(conv2d)
