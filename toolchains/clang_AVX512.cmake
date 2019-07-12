set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Hacking to make sure flags set in the toolchain file
# make it through to the main project.
unset(CMAKE_CXX_FLAGS CACHE)
set(CMAKE_CXX_FLAGS "-mfma -mavx2 -DUSE_AVX2 -mavx512f -mavx512cd -DUSE_AVX512" CACHE STRING "" FORCE)
