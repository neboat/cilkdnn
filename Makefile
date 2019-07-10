TARGETS = matmul.bc test test-conv2d cilkdnn.so

CXX=~/tapir/src/build/bin/clang++
CFLAGS=-std=c++11 -fcilkplus -ffast-math -Wall -mavx -mfma -mavx2
AVX512FLAGS=-mavx512f -mavx512cd -DUSE_AVX512

.PHONY : clean all

all: $(TARGETS)

%.bc : %.cpp
	$(CXX) -c $^ -emit-llvm -ftapir=none $(CFLAGS) -O1 -mllvm -enable-tapir-loop-stripmine=false -DNDEBUG # $(AVX512FLAGS)

test : test.cpp matmul.cpp
	$(CXX) $(CFLAGS) -g $^ -o $@ -DNOINLINEATTR -O3 # -march=native

test-conv2d : test-conv2d.cpp matmul.cpp
	$(CXX) $(CFLAGS) -g $^ -o $@ -DNOINLINEATTR -O3 # -march=native

cilkdnn.so : matmul.cpp cilkdnn.cpp
	$(CXX) $(CFLAGS) -g -fPIC -shared -I/usr/include/python2.7/ -lboost_python -lboost_numpy -lpython2.7 $^ -o $@ -DNOINLINEATTR -O3

clean:
	rm -rf $(TARGETS)
