// ~/tapir/src/build/bin/clang++ matmul.cpp test.cpp -o test -fcilkplus -std=c++11 -ffast-math  -mavx -mfma -mavx2 -O3 -Wall -g

#include <cassert>
#include <cstdlib>
#include <iostream>

#include <cilkdnn/matmul.hpp>

// template <typename F, bool transpose_lhs, bool transpose_rhs>
// void matmul(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
//             int64_t m, int64_t n, int64_t k);

// template <typename F, bool transpose_lhs, bool transpose_rhs>
// void matmul_ploops(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
//                    int64_t m, int64_t n, int64_t k);

// template <typename F>
// void matmul(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
//             int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs);

// template <typename F>
// void matmul_ploops(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
//                    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs);

int main(int argc, char *argv[]) {
  int m = 10;
  int n = 128;
  int k = 784;
  int32_t transpose_lhs = 0, transpose_rhs = 0;
  if (argc > 3) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  }
  if (argc > 5) {
    transpose_lhs = std::atoi(argv[4]);
    transpose_rhs = std::atoi(argv[5]);
  }

  float *C = new float[n * m];
  float *Ctmp = new float[n * m];
  // Create and initialize input matrices
  float *A, *B;
  if (transpose_lhs) {
    A = new float[m * k];
    for (int64_t i = 0; i < m; ++i)
      for (int64_t l = 0; l < k; ++l)
	A[i * k + l] = (float)rand() / (float)RAND_MAX;
  } else {
    A = new float[k * m];
    for (int64_t l = 0; l < k; ++l)
      for (int64_t i = 0; i < m; ++i)
	A[l * m + i] = (float)rand() / (float)RAND_MAX;
  }
  if (transpose_rhs) {
    B = new float[k * n];
    for (int64_t l = 0; l < k; ++l)
      for (int64_t j = 0; j < n; ++j)
	B[l * n + j] = (float)rand() / (float)RAND_MAX;
  } else {
    B = new float[n * k];
    for (int64_t j = 0; j < n; ++j)
      for (int64_t l = 0; l < k; ++l)
	B[j * k + l] = (float)rand() / (float)RAND_MAX;
  }
  // for (int64_t i = 0; i < m; ++i)
  //   for (int64_t l = 0; l < k; ++l)
  //     A[l * m + i] = (float)rand() / (float)RAND_MAX;
  // for (int64_t l = 0; l < k; ++l)
  //   for (int64_t j = 0; j < n; ++j)
  //     B[j * k + l] = (float)rand() / (float)RAND_MAX;

  matmul<float>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
  matmul_ploops<float>(Ctmp, A, B, m, n, k, transpose_lhs, transpose_rhs);

  // matmul<float, false, true>(C, A, B, m, n, k);
  // matmul_ploops<float, false, true>(Ctmp, A, B, m, n, k);

  // Verify result
  int64_t diff_count = 0;
  for (int64_t j = 0; j < n; ++j) {
    for (int64_t i = 0; i < m; ++i) {
      float diff = C[j * m + i] - Ctmp[j * m + i];
      if (diff < 0)
        diff = -diff;
      if (!((diff < 0.01) || (diff/C[j * m + i] < 0.00001))) {
        std::cerr << "Outputs differ: "
                  << "C[" << i << ", " << j << "] = " << C[j*m + i] << " "
                  << "Ctmp[" << i << ", " << j << "] = " << Ctmp[j*m + i]
                  << "\n";
        diff_count++;
      }
    }
  }
  assert(diff_count == 0 && "Verify failed");

  for (int trial = 0; trial < 50; ++trial)
    // matmul<float, false, true>(C, A, B, m, n, k);
    matmul<float>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    
  delete[] C;
  delete[] Ctmp;
  delete[] A;
  delete[] B;
  return 0;
}
