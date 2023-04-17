function (setup_test_exe test_name)
    add_executable(test-${test_name} test-${test_name}.cpp)
    target_include_directories(test-${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(test-${test_name} PRIVATE "-O3")
    target_link_libraries(test-${test_name} tcmalloc)
endfunction()  

setup_test_exe(matmul)
add_test(run_test-matmul test-matmul COMMAND test-matmul -c 100 100 100)

setup_test_exe(conv2d)
add_test(run_test-conv2d test-conv2d)

# Run Python script as part of build.
configure_file(cilkdnn_pytest.py cilkdnn_pytest.py COPYONLY)
add_test(NAME run_cilkdnn_pytest
         COMMAND ${PYTHON_EXECUTABLE} cilkdnn_pytest.py
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
# TBD: there may be a more robust way to locate cilknn.so in the build
# directory, but this works for now.       
set(cilknn_dir ${PROJECT_BINARY_DIR}/src)
set_tests_properties(run_cilkdnn_pytest PROPERTIES ENVIRONMENT "PYTHONPATH=${cilknn_dir}")
