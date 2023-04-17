// Compile string, using Opencilk clang++:
// clang++ test-matmul.cpp -o test -fopencilk -std=c++17 -ffast-math  -mavx -mfma -mavx2 -O3 -Wall -g

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cilkdnn/matmul.hpp>

#include <chrono>
#include <vector>
#include <cmath>

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

using EL_T = double;
static const int NUM_TRIALS = 51;

int main(int argc, char *argv[]) {
  int m = 10;
  int n = 128;
  int k = 784;
  int32_t transpose_lhs = 0, transpose_rhs = 0;
  bool verify = false;
  int arg_start = 0;
  if (argc > 1) {
    if (0 == std::strcmp("-c", argv[1])) {
      verify = true;
      ++arg_start;
      --argc;
    }
  }
  if (argc > 3) {
    m = std::atoi(argv[arg_start + 1]);
    n = std::atoi(argv[arg_start + 2]);
    k = std::atoi(argv[arg_start + 3]);
  }
  if (argc > 5) {
    transpose_lhs = std::atoi(argv[arg_start + 4]);
    transpose_rhs = std::atoi(argv[arg_start + 5]);
  }

  EL_T *C = new EL_T[n * m];
  EL_T *Ctmp = new EL_T[n * m];
  // Create and initialize input matrices
  EL_T *A, *B;
  if (transpose_lhs) {
    A = new EL_T[m * k];
    for (int64_t i = 0; i < m; ++i)
      for (int64_t l = 0; l < k; ++l)
        A[i * k + l] = (EL_T)rand() / (EL_T)RAND_MAX;
  } else {
    A = new EL_T[k * m];
    for (int64_t l = 0; l < k; ++l)
      for (int64_t i = 0; i < m; ++i)
        A[l * m + i] = (EL_T)rand() / (EL_T)RAND_MAX;
  }
  if (transpose_rhs) {
    B = new EL_T[k * n];
    for (int64_t l = 0; l < k; ++l)
      for (int64_t j = 0; j < n; ++j)
        B[l * n + j] = (EL_T)rand() / (EL_T)RAND_MAX;
  } else {
    B = new EL_T[n * k];
    for (int64_t j = 0; j < n; ++j)
      for (int64_t l = 0; l < k; ++l)
        B[j * k + l] = (EL_T)rand() / (EL_T)RAND_MAX;
  }
  // for (int64_t i = 0; i < m; ++i)
  //   for (int64_t l = 0; l < k; ++l)
  //     A[l * m + i] = (EL_T)rand() / (EL_T)RAND_MAX;
  // for (int64_t l = 0; l < k; ++l)
  //   for (int64_t j = 0; j < n; ++j)
  //     B[j * k + l] = (EL_T)rand() / (EL_T)RAND_MAX;

  if (verify) {
    std::cerr << "Checking matmul algorithm.\n";
    matmul<EL_T>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    matmul<EL_T>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    matmul_ploops<EL_T>(Ctmp, A, B, m, n, k, transpose_lhs, transpose_rhs);

    // matmul<EL_T, false, true>(C, A, B, m, n, k);
    // matmul_ploops<EL_T, false, true>(Ctmp, A, B, m, n, k);

    // Verify result
    int64_t diff_count = 0;
    cilk_for (int64_t j = 0; j < n; ++j) {
      cilk_for (int64_t i = 0; i < m; ++i) {
        EL_T diff = C[j * m + i] - Ctmp[j * m + i];
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
    if (diff_count > 0) {
      std::cerr << "Check failed.\n";
      delete[] C;
      delete[] Ctmp;
      delete[] A;
      delete[] B;
      return -1;
    }
    std::cerr << "Check passed.\n";
  }

  assert(NUM_TRIALS > 0);

  std::vector<double> trials(NUM_TRIALS);
  auto init = std::chrono::steady_clock::now();

  for (int trial = 0; trial < NUM_TRIALS; ++trial) {
    auto start = std::chrono::steady_clock::now();
    // matmul<EL_T, false, true>(C, A, B, m, n, k);
    matmul<EL_T>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration<double>(end-start).count();
    auto stime = std::chrono::duration<double>(start-init).count();
    std::cout << stime << " " << time << "\n";
    trials[trial] = time;
  }

  std::sort(trials.begin(), trials.end());
  double median_trial = (NUM_TRIALS & 0x1) ? trials[NUM_TRIALS/2] : (trials[NUM_TRIALS/2] + trials[NUM_TRIALS/2 + 1]) / 2;
  std::cout << "median matmul time: " << median_trial << "s\n";

  delete[] C;
  delete[] Ctmp;
  delete[] A;
  delete[] B;
  return 0;
}
