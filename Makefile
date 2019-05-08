TARGETS = matmul.bc test

CXX=~/tapir/src/build/bin/clang++
CFLAGS=-std=c++11 -fcilkplus -ffast-math -mavx -mfma -mavx2 -Wall
AVX512FLAGS=-mavx512f -mavx512cd -DUSE_AVX512

.PHONY : clean all

all: $(TARGETS)

%.bc : %.cpp
	$(CXX) -c $^ -emit-llvm -ftapir=none $(CFLAGS) -O1 -mllvm -enable-tapir-loop-stripmine=false -DNDEBUG # $(AVX512FLAGS)

test : test.cpp matmul.cpp
	$(CXX) $(CFLAGS) -g $^ -o $@ -DNOINLINEATTR -O3 # -march=native

clean:
	rm -rf $(TARGETS)
