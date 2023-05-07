// Compile string, using Opencilk clang++:
// clang++ test-matmul.cpp -o test -fopencilk -std=c++17 -ffast-math  -mavx -mfma -mavx2 -O3 -Wall -g

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cilkdnn/matmul.hpp>
#include <cilk/cilkscale.h>

#include <chrono>
#include <vector>
#include <cmath>

#include <unistd.h>

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

// __attribute__((weak)) int fib(int n) {
//   if (n < 2)
//     return n;
//   int x = cilk_spawn fib(n - 1);
//   int y = fib(n - 2);
//   cilk_sync;
//   return x + y;
// }

// const int64_t big_array_size = 1UL << 27;
// int64_t big_array[big_array_size] = {0};

// __attribute__((weak)) void clear_cache(int64_t x) {
//   cilk_for(int64_t i = 0; i < big_array_size; ++i)
//     ++big_array[i];
//   fprintf(stderr, "big_array[%ld] %ld\n", x, big_array[x]);
// }

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
    zero_init(C, m, n, m, n);
    zero_init(Ctmp, m, n, m, n);
    std::cerr << "Checking matmul algorithm.\n";
    matmul<EL_T>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    // zero_init(C, m, n, m, n);
    matmul<EL_T, /* initOutput */ true>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    matmul_ploops<EL_T>(Ctmp, A, B, m, n, k, transpose_lhs, transpose_rhs);

    // matmul<EL_T, false, true>(C, A, B, m, n, k);
    // matmul_ploops<EL_T, false, true>(Ctmp, A, B, m, n, k);

    // Verify result
    int64_t diff_count = 0;
    cilk_for(int64_t j = 0; j < n; ++j) {
      cilk_for(int64_t i = 0; i < m; ++i) {
        EL_T diff = C[j * m + i] - Ctmp[j * m + i];
        if (diff < 0)
          diff = -diff;
        if (!((diff < 0.01) || (diff / C[j * m + i] < 0.00001))) {
          std::cerr << "Outputs differ: "
                    << "C[" << i << ", " << j << "] = " << C[j * m + i] << " "
                    << "Ctmp[" << i << ", " << j << "] = " << Ctmp[j * m + i]
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
  // double prod = 1.0;
  // double sum = 0.0;
  auto init = std::chrono::steady_clock::now();

  cilk_scope {
  for (int trial = 0; trial < NUM_TRIALS; ++trial) {
    // fib(40);
    // Initialize output to zero.
    zero_init(C, m, n, m, n);

    auto start = std::chrono::steady_clock::now();
    // matmul<EL_T, false, true>(C, A, B, m, n, k);
    wsp_t start_wsp = wsp_getworkspan();
    // matmul<EL_T, true>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    matmul<EL_T, false>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    auto end = std::chrono::steady_clock::now();
    wsp_t end_wsp = wsp_getworkspan();
    auto time = std::chrono::duration<double>(end - start).count();
    auto stime = std::chrono::duration<double>(start - init).count();
    trials[trial] = time;
    std::cout << stime << " " << time << "\n";
    wsp_dump(end_wsp - start_wsp, "matmul");
    usleep(10000);
    // clear_cache(trial);
    // prod *= time;
    // sum += time;
  }
  }

  // auto end = std::chrono::steady_clock::now();
  // std::cout << "avg. matmul time: " << std::chrono::duration<double>(end-start).count() / NUM_TRIALS << "s\n";
  std::sort(trials.begin(), trials.end());
  double median_trial =
      (NUM_TRIALS & 0x1)
          ? trials[NUM_TRIALS / 2]
          : (trials[NUM_TRIALS / 2] + trials[NUM_TRIALS / 2 + 1]) / 2;
  std::cout << "median matmul time: " << median_trial << "s\n";
  // double geomean = pow(prod, 1.0/(double)NUM_TRIALS);
  // std::cout << "geomean " << geomean << ", avg " << sum / (double)NUM_TRIALS
  // << "\n";

  delete[] C;
  delete[] Ctmp;
  delete[] A;
  delete[] B;
  return 0;
}
