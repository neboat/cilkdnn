TARGETS = matmul_bcgen.bc test-matmul test-conv2d cilkdnn.so

CXX=clang++
CFLAGS=-std=c++11 -fcilkplus -ffast-math -Wall -mavx 
AVX2_FLAGS=-mfma -mavx2 -DUSE_AVX2
AVX512FLAGS=-mavx512f -mavx512cd -DUSE_AVX512

INCLUDE_DIR=./include

.PHONY : clean all

all: $(TARGETS)

%_bcgen.bc : %_bcgen.cpp
	$(CXX) -c -I$(INCLUDE_DIR) $^ -emit-llvm -ftapir=none $(CFLAGS) -O1 -mllvm -enable-tapir-loop-stripmine=false -DNDEBUG # $(AVX512FLAGS)
%.ll : %_bcgen.bc
	llvm-dis $^ 

test-matmul : test-matmul.cpp 
	$(CXX) -I$(INCLUDE_DIR) $(CFLAGS) -g $^ -o $@ -DNOINLINEATTR -O3 # -march=native

test-conv2d : test-conv2d.cpp 
	$(CXX) -I$(INCLUDE_DIR) $(CFLAGS) -g $^ -o $@ -DNOINLINEATTR -O3 # -march=native

cilkdnn.so : cilkdnn.cpp
	$(CXX) -I$(INCLUDE_DIR) $(CFLAGS) -g -fPIC -shared -I/usr/include/python2.7/ -lboost_python -lboost_numpy -lpython2.7 $^ -o $@ -DNOINLINEATTR -O3

clean:
	rm -rf $(TARGETS) *.bc *.ll
