// Compile string:
// ~/tapir/src/build/bin/clang++ -c matmul.cpp -emit-llvm -fcilkplus -ftapir=none -std=c++11 -ffast-math  -mavx -mfma -mavx2 -mavx512f -mavx512cd -O3

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cilk/cilk.h>
// #include <iostream>

#ifdef NOINLINEATTR
#define INLINEATTR __attribute__((noinline))
#else
#define INLINEATTR __attribute__((always_inline))
#endif

// const int64_t BASE = 32768/3;
const int64_t BASE = 32768;

template <typename F>
__attribute__((always_inline))
static void buffer_init(F *__restrict__ dst, const F *__restrict__ src,
                        int64_t m, int64_t n, int64_t mstride, int64_t nstride,
                        int32_t transpose) {
  if (!transpose) {
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i)
        dst[j * m + i] = src[j * mstride + i];
  } else {
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < n; ++j)
        dst[i * n + j] = src[i * nstride + j];
  }    
}

#define ARG_INDEX(arg, ii, m, jj, n, transpose)         \
  ((transpose) ? arg[((jj) * m) + (ii)] : arg[((ii) * n) + (jj)])

template <typename F, int64_t INum, int64_t JNum, bool transpose_lhs, bool transpose_rhs>
__attribute__((always_inline))
void matmul_vec
(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
 int64_t i, int64_t j, int64_t l,
 int64_t mstride, int64_t nstride, int64_t kstride) noexcept {
  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F)*INum)));
  vF outv[JNum];

#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum)
    outv[vnum] = (vF){ 0.0 };

  vF lhsv;
#pragma clang loop unroll(full)
  for (int64_t vidx = 0; vidx < INum; ++vidx)
    lhsv[vidx] = ARG_INDEX(lhs, l, kstride, i+vidx, mstride, transpose_lhs);

  vF rhsv[JNum];
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
    F rhs_val = ARG_INDEX(rhs, j+vnum, nstride, l, kstride, transpose_rhs);
    rhsv[vnum] = (vF) { rhs_val, rhs_val, rhs_val, rhs_val,
                        rhs_val, rhs_val, rhs_val, rhs_val };
  }
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum)
    outv[vnum] = lhsv * rhsv[vnum];

#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < INum; ++vidx) {
      out[(j+vnum) * mstride + (i+vidx)] += outv[vnum][vidx];
    }
  }
}

template <typename F, bool transpose_lhs, bool transpose_rhs>
void matmul_base(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
                 int64_t m, int64_t n, int64_t k,
                 int64_t mstride, int64_t nstride, int64_t kstride) noexcept {
  // std::cout << "m = " << m << ", n = " << n << ", k = " << k << "\n";

  // Zero-initialize the temporary buffer for out.
  F outTmp[n*m];
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i < m; ++i)
      outTmp[j * m + i] = 0.0;
  F lhsTmp[((transpose_lhs) ? (m*k) : (k*m))];
  F rhsTmp[((transpose_rhs) ? (k*n) : (n*k))];
  buffer_init(lhsTmp, lhs, m, k, mstride, kstride, transpose_lhs);
  buffer_init(rhsTmp, rhs, k, n, kstride, nstride, transpose_rhs);

  // for (int j = 0; j < n; ++j)
  //   for (int i = 0; i < m; ++i)
  //     for (int l = 0; l < k; ++l)
  //       outTmp[j * m + i] += lhsTmp[l * m + i] * rhsTmp[j * k + l];

  // // Add the result of this base-case multiplication back into out.
  // for (int j = 0; j < n; ++j)
  //   for (int i = 0; i < m; ++i)
  //     out[j * mstride + i] += outTmp[j * m + i];

//   // Perform the base-case multiplication.
//   if (!transpose_lhs && !transpose_rhs) {
//     for (int64_t j = 0; j < n; ++j)
//       for (int64_t l = 0; l < k; ++l)
// #pragma clang loop unroll_count(8)
//         for (int64_t i = 0; i < m; ++i)
//           out[j * mstride + i] += lhs[l * mstride + i] * rhs[j * kstride + l];
          
//   } else if (!transpose_lhs && transpose_rhs) {
//     for (int64_t l = 0; l < k; ++l)
//       for (int64_t j = 0; j < n; ++j)
// #pragma clang loop unroll_count(8)
//         for (int64_t i = 0; i < m; ++i)
//           out[j * mstride + i] += lhs[l * mstride + i] * rhs[l * nstride + j];

//   } else if (transpose_lhs && !transpose_rhs) {
//     for (int64_t j = 0; j < n; ++j)
//       for (int64_t i = 0; i < m; ++i)
// #pragma clang loop unroll_count(8)
//         for (int64_t l = 0; l < k; ++l)
//           out[j * mstride + i] += lhs[i * kstride + l] * rhs[j * kstride + l];

//   } else {
//     for (int64_t i = 0; i < m; ++i)
//       for (int64_t j = 0; j < n; ++j)
// #pragma clang loop unroll_count(8)
//         for (int64_t l = 0; l < k; ++l)
//           out[j * mstride + i] += lhs[i * kstride + l] * rhs[l * nstride + j];
//   }

//   for (int64_t l = 0; l < k; ++l)
//     for (int64_t j = 0; j < n; ++j)
// #pragma clang loop unroll_count(8)
//       for (int64_t i = 0; i < m; ++i)
//         out[j * mstride + i] +=
//           ARG_INDEX(lhs, l, kstride, i, mstride, transpose_lhs) *
//           ARG_INDEX(rhs, j, nstride, l, kstride, transpose_rhs);

  const int64_t nVec = 4; // (!transpose_lhs) ? 8 : 4;
#define AVX512COARSEN 8

//   if (m > 16*4 * AVX512COARSEN) {
//     for (int64_t l = 0; l < k; ++l) {
//       for (int64_t jj = 0; jj < n/nVec; ++jj) {
//         for (int64_t ii = 0; ii < m/16; ++ii)
//           matmul_vec<F, 16, nVec, transpose_lhs, transpose_rhs>
//             (out, lhs, rhs, 16 * ii, nVec * jj, l, mstride, nstride, kstride);
//         for (int64_t j = nVec * jj; j < nVec * (jj+1); ++j)
// #pragma clang loop vectorize(disable)
//           for (int64_t i = 16 * (m/16); i < m; ++i)
//             out[j * mstride + i] +=
//               ARG_INDEX(lhs, l, kstride, i, mstride, transpose_lhs) *
//               ARG_INDEX(rhs, j, nstride, l, kstride, transpose_rhs);
//       }
//       for (int64_t j = nVec * (n/nVec); j < n; ++j) {
//         // for (int64_t ii = 0; ii < m/8; ++ii)
//         //   matmul_vec<F, 8, 1, transpose_lhs, transpose_rhs>
//         //     (out, lhs, rhs, 8 * ii, j, l, mstride, nstride, kstride);
// #pragma clang loop vectorize(disable)
//         // for (int64_t i = 8 * (m/8); i < m; ++i)
//         for (int64_t i = 0; i < m; ++i)
//           out[j * mstride + i] +=
//             ARG_INDEX(lhs, l, kstride, i, mstride, transpose_lhs) *
//             ARG_INDEX(rhs, j, nstride, l, kstride, transpose_rhs);
//       }
//     }
//   } else {

    for (int64_t l = 0; l < k; ++l) {
      for (int64_t jj = 0; jj < n/nVec; ++jj) {
        for (int64_t ii = 0; ii < m/8; ++ii)
          // matmul_vec<F, 8, nVec, transpose_lhs, transpose_rhs>
          //   (out, lhs, rhs, 8 * ii, nVec * jj, l, mstride, nstride, kstride);
          matmul_vec<F, 8, nVec, transpose_lhs, transpose_rhs>
            (outTmp, lhsTmp, rhsTmp, 8 * ii, nVec * jj, l, m, n, k);
        for (int64_t j = nVec * jj; j < nVec * (jj+1); ++j)
#pragma clang loop vectorize(disable)
          for (int64_t i = 8 * (m/8); i < m; ++i)
            // out[j * mstride + i] +=
            //   ARG_INDEX(lhs, l, kstride, i, mstride, transpose_lhs) *
            //   ARG_INDEX(rhs, j, nstride, l, kstride, transpose_rhs);
            outTmp[j * m + i] +=
              ARG_INDEX(lhsTmp, l, k, i, m, transpose_lhs) *
              ARG_INDEX(rhsTmp, j, n, l, k, transpose_rhs);
      }
      // for (int64_t jj = 0; jj < n/2; ++jj) {
      //   for (int64_t j = 2 * jj; j < 2 * (jj+1); ++j)
      //     for (int64_t i = 8 * (m/8); i < m; ++i)
      //       out[j * mstride + i] +=
      //         ARG_INDEX(lhs, l, kstride, i, mstride, transpose_lhs) *
      //         ARG_INDEX(rhs, j, nstride, l, kstride, transpose_rhs);
      // }
      // for (int64_t jj = (4 * (n/4)) / 2; jj < n/2; ++jj) {
      //   for (int64_t ii = 0; ii < m/8; ++ii)
      //     matmul_vec<F, 8, 2, transpose_lhs, transpose_rhs>
      //       (out, lhs, rhs, 8 * ii, 2 * jj, l, mstride, nstride, kstride);
      // }
      // for (int64_t jj = (4 * (n/4)) / 2; jj < n/2; ++jj) {
      //   for (int64_t j = 2 * jj; j < 2 * (jj+1); ++j)
      //     for (int64_t i = 8 * (m/8); i < m; ++i)
      //       out[j * mstride + i] +=
      //         ARG_INDEX(lhs, l, kstride, i, mstride, transpose_lhs) *
      //         ARG_INDEX(rhs, j, nstride, l, kstride, transpose_rhs);
      // }

      for (int64_t j = nVec * (n/nVec); j < n; ++j) {
//         for (int64_t ii = 0; ii < m/8; ++ii)
//           matmul_vec<F, 8, 1, transpose_lhs, transpose_rhs>
//             (out, lhs, rhs, 8 * ii, j, l, mstride, nstride, kstride);
#pragma clang loop vectorize(disable)
        // for (int64_t i = 8 * (m/8); i < m; ++i)
        for (int64_t i = 0; i < m; ++i)
          // out[j * mstride + i] +=
          //   ARG_INDEX(lhs, l, kstride, i, mstride, transpose_lhs) *
          //   ARG_INDEX(rhs, j, nstride, l, kstride, transpose_rhs);
          outTmp[j * m + i] +=
            ARG_INDEX(lhsTmp, l, k, i, m, transpose_lhs) *
            ARG_INDEX(rhsTmp, j, n, l, k, transpose_rhs);
      }
    }

  // } else {
  //   for (int64_t j = 0; j < n; ++j) {
  //     for (int64_t l = 0; l < k; ++l) {
  //       // for (int64_t ii = 0; ii < m/8; ++ii)
  //       //   matmul_vec<F, 8, 1, transpose_lhs, transpose_rhs>
  //       //     (out, lhs, rhs, 8 * ii, 1 * j, l, mstride, nstride, kstride);
  //       // for (int64_t i = 8 * (m/8); i < m; ++i)
  //       for (int64_t i = 0; i < m; ++i)
  //         out[j * mstride + i] +=
  //           ARG_INDEX(lhs, l, kstride, i, mstride, transpose_lhs) *
  //           ARG_INDEX(rhs, j, nstride, l, kstride, transpose_rhs);
  //     }
  //   }
  // }

  // Add the result of this base-case multiplication back into out.
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      out[j * mstride + i] += outTmp[j * m + i];
}

__attribute__((always_inline))
static int64_t split_dim(int64_t n) {
  // Special case: n is a power of 2.
  if ((n & -n) == n)
    return n/2;
  const int64_t split = 1 << (64 - __builtin_clzl(n - 1));
  return split / 2;
  // const int64_t split = (n > 1) ? (1 << (64 - __builtin_clzl(n - 1))) : n;
  // return split / 2;
}

// template <typename F, bool transpose_lhs, bool transpose_rhs>
template <typename F>
void matmul_dac(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
                int64_t m, int64_t n, int64_t k,
                int64_t mstride, int64_t nstride, int64_t kstride,
                bool transpose_lhs, bool transpose_rhs
                ) noexcept {
  if (m == 0 || n == 0 || k == 0)
    return;

  // if ((m * n) + (m * k) + (n * k) <= BASE / sizeof(F)) {
  if (m * n * k <= BASE) {
    // matmul_base<F, transpose_lhs, transpose_rhs>
    //   (out, lhs, rhs, m, n, k, mstride, nstride, kstride);
    if (transpose_lhs && transpose_rhs) {
      matmul_base<F, true, true>
        (out, lhs, rhs, m, n, k, mstride, nstride, kstride);
    } else if (transpose_lhs && !transpose_rhs) {
      matmul_base<F, true, false>
        (out, lhs, rhs, m, n, k, mstride, nstride, kstride);
    } else if (!transpose_lhs && transpose_rhs) {
      matmul_base<F, false, true>
        (out, lhs, rhs, m, n, k, mstride, nstride, kstride);
    } else {
      matmul_base<F, false, false>
        (out, lhs, rhs, m, n, k, mstride, nstride, kstride);
    }
    return;
  }

  // Split the maximum dimension
  const int64_t max_dim = std::max(std::max(m, n), k);
  // We prefer to spawn higher in the recursion tree than lower.
  // Because the base case vectorizes over dimension m, which is the
  // fastest moving dimension of the output matrix, we prefer to split
  // n instead of m.
  if (max_dim == n) {
    const int64_t split = split_dim(n);
    cilk_spawn matmul_dac<F// , transpose_lhs, transpose_rhs
                          >
      (out,
       &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs),
       m, split, k, mstride, nstride, kstride,
       transpose_lhs, transpose_rhs
       );
    matmul_dac<F// , transpose_lhs, transpose_rhs
               >
      (out + (split * mstride),
       &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
       &ARG_INDEX(rhs, split, nstride, 0, kstride, transpose_rhs),
       m, (n - split), k, mstride, nstride, kstride,
       transpose_lhs, transpose_rhs
       );
    cilk_sync;
  } else if (max_dim == m) {
    const int64_t split = split_dim(m);
    cilk_spawn matmul_dac<F// , transpose_lhs, transpose_rhs
                          >
      (out,
       &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs),
       split, n, k, mstride, nstride, kstride,
       transpose_lhs, transpose_rhs
       );
    matmul_dac<F// , transpose_lhs, transpose_rhs
               >
      (out + split,
       &ARG_INDEX(lhs, 0, kstride, split, mstride, transpose_lhs),
       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs),
       (m - split), n, k, mstride, nstride, kstride,
       transpose_lhs, transpose_rhs
       );
    cilk_sync;
  } else { // max_dim == k
    const int64_t split = split_dim(k);
    matmul_dac<F// , transpose_lhs, transpose_rhs
               >
      (out,
       &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs),
       m, n, split, mstride, nstride, kstride,
       transpose_lhs, transpose_rhs
       );
    matmul_dac<F// , transpose_lhs, transpose_rhs
               >
      (out,
       &ARG_INDEX(lhs, split, kstride, 0, mstride, transpose_lhs),
       &ARG_INDEX(rhs, 0, nstride, split, kstride, transpose_rhs),
       m, n, (k - split), mstride, nstride, kstride,
       transpose_lhs, transpose_rhs
       );
  }

  // int64_t msplit = m, nsplit = n;
  // if (max_dim == n || max_dim == m) {
  //   if (max_dim == n)
  //     nsplit = split_dim(n);
  //   else if (max_dim == m)
  //     msplit = split_dim(m);

  //   cilk_spawn matmul_dac<F>
  //     (out,
  //      &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //      &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs),
  //      msplit, nsplit, k, mstride, nstride, kstride,
  //      transpose_lhs, transpose_rhs
  //      );

  //   if (max_dim == n) {
  //     matmul_dac<F>
  //       (out + (nsplit * mstride),
  //        &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //        &ARG_INDEX(rhs, nsplit, nstride, 0, kstride, transpose_rhs),
  //        m, (n - nsplit), k, mstride, nstride, kstride,
  //        transpose_lhs, transpose_rhs
  //        );
  //   } else if (max_dim == m) {
  //     matmul_dac<F>
  //       (out + msplit,
  //        &ARG_INDEX(lhs, 0, kstride, msplit, mstride, transpose_lhs),
  //        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs),
  //        (m - msplit), n, k, mstride, nstride, kstride,
  //        transpose_lhs, transpose_rhs
  //        );
  //   }
  //   cilk_sync;
  // } else {
  //   const int64_t split = split_dim(k);
  //   matmul_dac<F// , transpose_lhs, transpose_rhs
  //              >
  //     (out,
  //      &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //      &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs),
  //      m, n, split, mstride, nstride, kstride,
  //      transpose_lhs, transpose_rhs
  //      );
  //   matmul_dac<F// , transpose_lhs, transpose_rhs
  //              >
  //     (out,
  //      &ARG_INDEX(lhs, split, kstride, 0, mstride, transpose_lhs),
  //      &ARG_INDEX(rhs, 0, nstride, split, kstride, transpose_rhs),
  //      m, n, (k - split), mstride, nstride, kstride,
  //      transpose_lhs, transpose_rhs
  //      );
  // }
}

// template <typename F, bool transpose_lhs, bool transpose_rhs>
template <typename F>
INLINEATTR
void matmul(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
            int64_t m, int64_t n, int64_t k,
            int32_t transpose_lhs, int32_t transpose_rhs) noexcept {
  // Initialize output to zero.
  cilk_for (int64_t i = 0; i < m; ++i) {
    cilk_for (int64_t j = 0; j < n; ++j) {
      out[j * m + i] = 0.0;
    }
  }
  // matmul_dac<F, transpose_lhs, transpose_rhs>
  //   (out, lhs, rhs, m, n, k, m, n, k);
  matmul_dac<F>
    (out, lhs, rhs, m, n, k, m, n, k, transpose_lhs, transpose_rhs);
}

template <typename F>
INLINEATTR
void matmul_ploops(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
                   int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
  if (n > m) {
    cilk_for (int64_t i = 0; i < m; ++i) {
      cilk_for (int64_t j = 0; j < n; ++j) {
        out[j * m + i] = 0.0;
        for (int64_t l = 0; l < k; ++l)
          out[j * m + i] +=
            ARG_INDEX(lhs, l, k, i, m, transpose_lhs) *
            ARG_INDEX(rhs, j, n, l, k, transpose_rhs);
      }
    }
  } else {
    cilk_for (int64_t j = 0; j < n; ++j) {
      cilk_for (int64_t i = 0; i < m; ++i) {
        out[j * m + i] = 0.0;
        for (int64_t l = 0; l < k; ++l)
          out[j * m + i] +=
            ARG_INDEX(lhs, l, k, i, m, transpose_lhs) *
            ARG_INDEX(rhs, j, n, l, k, transpose_rhs);
      }
    }
  }
}

template void
matmul_ploops<float>(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
                                   int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs);

template void
matmul_ploops<double>(double *__restrict__ out, const double *__restrict__ lhs, const double *__restrict__ rhs,
                                    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs);

extern "C" {
INLINEATTR
void matmul_f32(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
                int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
#ifndef NDEBUG
  matmul<float>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
#else
  matmul_ploops<float>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
#endif
}

INLINEATTR
void matmul_f64(double *__restrict__ out, const double *__restrict__ lhs, const double *__restrict__ rhs,
                int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
#ifndef NDEBUG
  matmul<double>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
#else
  matmul<double>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
#endif
}

// INLINEATTR
// void matmul_f32_0_0(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
//                     int64_t m, int64_t n, int64_t k) {
// #ifndef NDEBUG
//   matmul<float, false, false>(out, lhs, rhs, m, n, k);
// #else
//   matmul<float, false, false>(out, lhs, rhs, m, n, k);
// #endif
// }

// INLINEATTR
// void matmul_f32_0_1(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
//                     int64_t m, int64_t n, int64_t k) {
// #ifndef NDEBUG
//   matmul<float, false, true>(out, lhs, rhs, m, n, k);
// #else
//   matmul<float, false, true>(out, lhs, rhs, m, n, k);
// #endif
// }

// INLINEATTR
// void matmul_f32_1_0(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
//                     int64_t m, int64_t n, int64_t k) {
// #ifndef NDEBUG
//   matmul<float, true, false>(out, lhs, rhs, m, n, k);
// #else
//   matmul<float, true, false>(out, lhs, rhs, m, n, k);
// #endif
// }

// INLINEATTR
// void matmul_f32_1_1(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
//                     int64_t m, int64_t n, int64_t k) {
// #ifndef NDEBUG
//   matmul<float, true, true>(out, lhs, rhs, m, n, k);
// #else
//   matmul<float, true, true>(out, lhs, rhs, m, n, k);
// #endif
// }

// INLINEATTR
// void matmul_f64_0_0(double *__restrict__ out, const double *__restrict__ lhs, const double *__restrict__ rhs,
//                     int64_t m, int64_t n, int64_t k) {
// #ifndef NDEBUG
//   matmul<double, false, false>(out, lhs, rhs, m, n, k);
// #else
//   matmul<double, false, false>(out, lhs, rhs, m, n, k);
// #endif
// }

// INLINEATTR
// void matmul_f64_0_1(double *__restrict__ out, const double *__restrict__ lhs, const double *__restrict__ rhs,
//                     int64_t m, int64_t n, int64_t k) {
// #ifndef NDEBUG
//   matmul<double, false, true>(out, lhs, rhs, m, n, k);
// #else
//   matmul<double, false, true>(out, lhs, rhs, m, n, k);
// #endif
// }

// INLINEATTR
// void matmul_f64_1_0(double *__restrict__ out, const double *__restrict__ lhs, const double *__restrict__ rhs,
//                     int64_t m, int64_t n, int64_t k) {
// #ifndef NDEBUG
//   matmul<double, true, false>(out, lhs, rhs, m, n, k);
// #else
//   matmul<double, true, false>(out, lhs, rhs, m, n, k);
// #endif
// }

// INLINEATTR
// void matmul_f64_1_1(double *__restrict__ out, const double *__restrict__ lhs, const double *__restrict__ rhs,
//                     int64_t m, int64_t n, int64_t k) {
// #ifndef NDEBUG
//   matmul<double, true, true>(out, lhs, rhs, m, n, k);
// #else
//   matmul<double, true, true>(out, lhs, rhs, m, n, k);
// #endif
// }

}
