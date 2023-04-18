// Compile string, using Opencilk clang++:
// clang++ -c matmul.cpp -emit-llvm -fopencilk -ftapir=none -std=c++17 -ffast-math  -mavx -mfma -mavx2 # -mavx512f -mavx512cd -O3

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cilk/cilk.h>
#include <cstdio>

#include "common_sysdep.hpp"

const int64_t BASE = 32768 * 24;

#define INDEX(ii, m, jj, n, rhs)                                               \
  ((rhs) ? ((((ii) / 4) * 4 * n) + ((jj)*4) + ((ii)&3))                        \
         : ((((jj) / 8) * 8 * m) + ((ii)*8) + ((jj)&7)))

#define BUF_INDEX(arg, ii, m, jj, n, rhs)       \
  (arg[INDEX(ii, m, jj, n, rhs)])

#define ARG_INDEX(arg, ii, m, jj, n, transpose)                         \
  ((transpose) ? arg[((jj) * m) + (ii)] : arg[((ii) * n) + (jj)])

__attribute__((always_inline))
static int64_t split_dim(int64_t n) {
  // Special case: n is a power of 2.
  if ((n & -n) == n)
    return n/2;
  const int64_t split = 1 << ((8 * sizeof(int64_t)) - __builtin_clzl(n - 1));
  return split / 2;
}

template <typename F>
__attribute__((always_inline)) static void
buffer_transpose(F *__restrict__ dst, const F *__restrict__ src, int64_t x,
                 int64_t y, int64_t stride) {
  const int64_t TBASE = 4;
  for (int64_t jj = 0; jj < y / TBASE; ++jj) {
    for (int64_t i = 0; i < x; ++i) {
      for (int64_t j = (jj * TBASE); j < ((jj + 1) * TBASE); ++j) {
	BUF_INDEX(dst, j, y, i, x, true) = src[j * stride + i];
      }
    }
  }
  for (int64_t j = (y / TBASE) * TBASE; j < y; ++j) {
    for (int64_t i = 0; i < x; ++i) {
      BUF_INDEX(dst, j, y, i, x, true) = src[j * stride + i];
    }
  }
}

template <typename F, bool transpose>
__attribute__((always_inline)) static void
buffer_copy(F *__restrict__ dst, const F *__restrict__ src, int64_t x,
            int64_t y, int64_t stride) {
  const int64_t CBASE = 8;
  for (int64_t jj = 0; jj < y / CBASE; ++jj) {
    for (int64_t ii = 0; ii < x / CBASE; ++ii) {
      for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
        for (int64_t i = (ii * CBASE); i < ((ii + 1) * CBASE); ++i) {
          BUF_INDEX(dst, j, y, i, x, transpose) = src[j * stride + i];
        }
      }
    }
    for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
      for (int64_t i = (x / CBASE) * CBASE; i < x; ++i) {
        BUF_INDEX(dst, j, y, i, x, transpose) = src[j * stride + i];
      }
    }
  }
  for (int64_t ii = 0; ii < x / CBASE; ++ii) {
    for (int64_t j = (y / CBASE) * CBASE; j < y; ++j) {
      for (int64_t i = (ii * CBASE); i < ((ii + 1) * CBASE); ++i) {
        BUF_INDEX(dst, j, y, i, x, transpose) = src[j * stride + i];
      }
    }
  }
  for (int64_t j = (y / CBASE) * CBASE; j < y; ++j) {
    for (int64_t i = (x / CBASE) * CBASE; i < x; ++i) {
      BUF_INDEX(dst, j, y, i, x, transpose) = src[j * stride + i];
    }
  }
}

template <typename F, bool transposed, bool want_transpose>
__attribute__((always_inline)) static void
buffer_init(F *__restrict__ dst, const F *__restrict__ src, int64_t m,
            int64_t n, int64_t mstride, int64_t nstride) {
  if (!want_transpose) {
    if (!transposed) {
      buffer_copy<F, false>(dst, src, m, n, mstride);
    } else {
      buffer_copy<F, false>(dst, src, n, m, nstride);
    }
  } else {
    if (!transposed) {
      // buffer_copy<F, true>(dst, src, m, n, mstride);
      buffer_transpose(dst, src, m, n, mstride);
    } else {
      // buffer_copy<F, true>(dst, src, n, m, nstride);
      buffer_transpose(dst, src, n, m, nstride);
    }
  }
}

// A simple and general vectorized base case for matrix multiply.
// This base case computes a INum x JNum submatrix in column-major
// order from a INum subcolumn of A and a JNum subrow of B.
template <typename F, int64_t INum, int64_t JNum, bool transpose_lhs,
          bool transpose_rhs>
__attribute__((always_inline)) void
matmul_vec(F *__restrict__ out, const F *__restrict__ lhs,
           const F *__restrict__ rhs, int64_t i, int64_t j, int64_t l,
           int64_t mstride, int64_t nstride, int64_t kstride) noexcept {
  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F) * INum)));
  vF outv[JNum];

  // Zero-initialize output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum)
    outv[vnum] = (vF){0.0};

  // Get INum values from a column of lhs.
  vF lhsv;
#pragma clang loop unroll(full)
  for (int64_t vidx = 0; vidx < INum; ++vidx) {
    lhsv[vidx] = BUF_INDEX(lhs, l, kstride, i + vidx, mstride, transpose_lhs);
  }

  // Fill each rhs vector with a value from one of INum rows of rhs.
  vF rhsv[JNum];
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
    // Read the value from a row of rhs.
    F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, l, kstride, transpose_rhs);
    // Broadcast that value through one of the rhsv.
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < INum; ++vidx) {
      rhsv[vnum][vidx] = rhs_val;
    }
  }

  // Each output vector gets the element-wise product of lhsv and one
  // of the rhsv.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum)
    outv[vnum] = lhsv * rhsv[vnum];

  // Add the output vectors to the output matrix.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < INum; ++vidx) {
      out[(j + vnum) * mstride + (i + vidx)] += outv[vnum][vidx];
    }
  }
}

// A simple and general vectorized base case for matrix multiply.
// This base case computes a INum x JNum submatrix in column-major
// order from a INum subcolumn of A and a JNum subrow of B.
template <typename F, int64_t INum, int64_t JNumMax>
__attribute__((always_inline)) void
matmul_vec_flex(F *__restrict__ out, const F *__restrict__ lhs,
                const F *__restrict__ rhs, int64_t i, int64_t j, int64_t l,
                int64_t JNum, int64_t KNum, int64_t mstride, int64_t nstride,
                int64_t kstride, int64_t outstride) noexcept {
  __builtin_assume(JNum < JNumMax);
  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F) * INum)));
  vF outv[JNum];

  // Zero-initialize output vectors.
  for (int64_t vnum = 0; vnum < JNum; ++vnum)
    outv[vnum] = (vF){0.0};

  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
    // Get INum values from a column of lhs.
    vF lhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < INum; ++vidx) {
      lhsv[vidx] = BUF_INDEX(lhs, my_l, kstride, i + vidx, mstride, false);
    }

    // Fill each rhs vector with a value from one of INum rows of rhs.
    vF rhsv[JNum];
    for (int64_t vnum = 0; vnum < JNum; ++vnum) {
      // Read the value from a row of rhs.
      F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, true);
      // Broadcast that value through one of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vidx = 0; vidx < INum; ++vidx) {
        rhsv[vnum][vidx] = rhs_val;
      }
    }

    // Each output vector gets the element-wise product of lhsv and one
    // of the rhsv.
    for (int64_t vnum = 0; vnum < JNum; ++vnum)
      outv[vnum] += lhsv * rhsv[vnum];
  }

  // Add the output vectors to the output matrix.
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < INum; ++vidx) {
      out[(j + vnum) * outstride + (i + vidx)] += outv[vnum][vidx];
    }
  }
}

// A simple and general vectorized base case for matrix multiply.
// This base case computes a INum x JNum submatrix in column-major
// order from a INum subcolumn of A and a JNum subrow of B.
template <typename F, int64_t JNum, int64_t INumMax>
__attribute__((always_inline)) void
matmul_vec_flex_col(F *__restrict__ out, const F *__restrict__ lhs,
                    const F *__restrict__ rhs, int64_t i, int64_t j, int64_t l,
                    int64_t INum, int64_t KNum, int64_t mstride,
                    int64_t nstride, int64_t kstride,
                    int64_t outstride) noexcept {
  __builtin_assume(INum < INumMax);
  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F) * JNum)));
  vF outv[INum];

  // Zero-initialize output vectors.
  for (int64_t vnum = 0; vnum < INum; ++vnum)
    outv[vnum] = (vF){0.0};

  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
    // Fill each lhs vector with a value from one of JNum rows of lhs.
    vF lhsv[INum];
    for (int64_t vnum = 0; vnum < INum; ++vnum) {
      F lhs_val = BUF_INDEX(lhs, my_l, kstride, i + vnum, mstride, false);
#pragma clang loop unroll(full)
      for (int64_t vidx = 0; vidx < JNum; ++vidx) {
        lhsv[vnum][vidx] = lhs_val;
      }
    }

    // Get JNum values from a row of rhs.
    vF rhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < JNum; ++vidx) {
      rhsv[vidx] = BUF_INDEX(rhs, j + vidx, nstride, my_l, kstride, true);
    }

    // Each output vector gets the element-wise product of lhsv and one
    // of the rhsv.
    for (int64_t vnum = 0; vnum < INum; ++vnum)
      outv[vnum] += lhsv[vnum] * rhsv;
  }

  // Add the output vectors to the output matrix.
  for (int64_t vnum = 0; vnum < INum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < JNum; ++vidx) {
      out[(j + vidx) * outstride + (i + vnum)] += outv[vnum][vidx];
    }
  }
}

// A specialized base case that computes the outer product of
// subcolumns of A and subrows of B.  Unlike the more general
// vectorized base case, this version uses fewer memory accesses by
// storing the outer-product result in vector registers.
template <typename F>
__attribute__((always_inline)) void
matmul_vec_op_halfvec(F *__restrict__ out, const F *__restrict__ lhs,
                      const F *__restrict__ rhs, int64_t i, int64_t j,
                      int64_t l, int64_t KNum, int64_t mstride, int64_t nstride,
                      int64_t kstride, int64_t outstride) noexcept {
  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F) * 4)));

  // Vectors storing output submatrix.
  vF outv[4];

  // Zero-initialize the output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < 4; ++vnum)
    outv[vnum] = (vF){0.0};

  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
    // In the following comments, A denotes the rhs, and B denotes the lhs.

    // Store a subcolumn of lhs into lhsv.
    // lhsv = A0 A1 A2 A3
    vF lhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 4; ++vidx) {
      lhsv[vidx] = BUF_INDEX(lhs, my_l, kstride, i + vidx, mstride, false);
    }

    // Store a subrow of rhs into rhsv, replicated twice.
    // rhsv = B0 B1 B2 B3
    vF rhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 4; ++vidx) {
      rhsv[vidx] = BUF_INDEX(rhs, j + vidx, nstride, my_l, kstride, true);
    }

    // Perform the multiplications using two vector shuffles --- one
    // for lhs and one for rhs --- and four vector multiplies among
    // the inputs and their shuffles variations.
    // outv[0] = A0B0 A1B1 A2B2 A3B3
    outv[0] += lhsv * rhsv;
    // rhsv_p0 = B1 B0 B3 B2
    vF rhsv_p0 = __builtin_shufflevector(rhsv, rhsv, 1, 0, 3, 2);
    // outv[1] = A0B1 A1B0 A2B3 A3B2
    outv[1] += lhsv * rhsv_p0;
    // lhsv_p0 = A2 A3 A0 A1
    vF lhsv_p0 = __builtin_shufflevector(lhsv, lhsv, 2, 3, 0, 1);
    // outv[2] = A2B0 A3B1 A0B2 A1B3
    outv[2] += lhsv_p0 * rhsv;
    // outv[3] = A2B1 A3B0 A0B3 A1B2
    outv[3] += lhsv_p0 * rhsv_p0;
  }

  // Shuffle the output vectors to support simple vector-add
  // operations to store the result back into the output matrix.
  //
  vF st[8];
  // A0B0, A1B0, A2B2, A3B2
  st[0] = __builtin_shufflevector(outv[0], outv[1], 0, 5, 2, 7);
  // A0B1, A1B1, A2B3, A3B3
  st[1] = __builtin_shufflevector(outv[1], outv[0], 0, 5, 2, 7);
  // A0B2, A1B2, A2B0, A3B0
  st[2] = __builtin_shufflevector(outv[2], outv[3], 2, 7, 0, 5);
  // A0B3, A1B3, A2B1, A3B1
  st[3] = __builtin_shufflevector(outv[3], outv[2], 2, 7, 0, 5);

  // A0B0, A1B0, A2B0, A3B0
  st[4] = __builtin_shufflevector(st[0], st[2], 0, 1, 6, 7);
  // A0B1, A1B1, A2B1, A3B1
  st[5] = __builtin_shufflevector(st[1], st[3], 0, 1, 6, 7);
  // A0B2, A1B2, A2B2, A3B2
  st[6] = __builtin_shufflevector(st[2], st[0], 0, 1, 6, 7);
  // A0B3, A1B3, A2B3, A3B3
  st[7] = __builtin_shufflevector(st[3], st[1], 0, 1, 6, 7);

  // Add the output vectors to the output matrix.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < 4; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 4; ++vidx) {
      out[(j + vnum) * outstride + (i + vidx)] += st[4 + vnum][vidx];
    }
  }
}

// A specialized base case that computes the outer product of
// subcolumns of A and subrows of B.  Unlike the more general
// vectorized base case, this version uses fewer memory accesses by
// storing the outer-product result in vector registers.
template <typename F>
__attribute__((always_inline)) void
matmul_vec_op(F *__restrict__ out, const F *__restrict__ lhs,
              const F *__restrict__ rhs, int64_t i, int64_t j, int64_t l,
              int64_t KNum, int64_t mstride, int64_t nstride, int64_t kstride,
              int64_t outstride) noexcept {
  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F) * 8)));

  // Vectors storing output submatrix.
  vF outv[4];

  // Zero-initialize the output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < 4; ++vnum)
    outv[vnum] = (vF){0.0};

  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
    // Store a subcolumn of lhs into lhsv.
    // lhsv = A0 A1 A2 A3 A4 A5 A6 A7
    vF lhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 8; ++vidx)
      lhsv[vidx] = BUF_INDEX(lhs, my_l, kstride, i + vidx, mstride, false);

    // Store a subrow of rhs into rhsv, replicated twice.
    // rhsv = B0 B1 B2 B3 B0 B1 B2 B3
    vF rhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 4; ++vidx) {
      rhsv[vidx] = BUF_INDEX(rhs, j + vidx, nstride, my_l, kstride, true);
      rhsv[vidx + 4] = rhsv[vidx];
    }

    // Perform the multiplications using two vector shuffles --- one
    // for lhs and one for rhs --- and four vector multiplies among
    // the inputs and their shuffles variations.
    // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B0 A5B1 A6B2 A7B3
    outv[0] += lhsv * rhsv;
    // rhsv_p0 = B1 B0 B3 B2 B1 B0 B3 B2
    vF rhsv_p0 = __builtin_shufflevector(rhsv, rhsv, 1, 0, 3, 2, 5, 4, 7, 6);
    // outv[1] = A1B0 A0B1 A3B2 A2B3 A5B0 A4B1 A7B2 A6B3
    outv[1] += lhsv * rhsv_p0;
    // lhsv_p0 = A2 A3 A0 A1 A6 A7 A4 A5
    vF lhsv_p0 = __builtin_shufflevector(lhsv, lhsv, 2, 3, 0, 1, 6, 7, 4, 5);
    // outv[2] = A2B0 A3B1 A0B2 A1B3 A6B0 A7B1 A4B2 A5B3
    outv[2] += lhsv_p0 * rhsv;
    // outv[2] = A2B1 A3B0 A0B3 A1B2 A6B1 A7B0 A4B3 A5B2
    outv[3] += lhsv_p0 * rhsv_p0;
  }

  // Shuffle the output vectors to support simple vector-add
  // operations to store the result back into the output matrix.
  //
  // Below, A denotes the rhs, and B denotes the lhs.
  vF st[8];
  // A0B0, A1B0, A2B2, A3B2, A4B0, A5B0, A6B2, A7B2
  st[0] = __builtin_shufflevector(outv[0], outv[1], 0, 9, 2, 11, 4, 13, 6, 15);
  // A0B1, A1B1, A2B3, A3B3, A4B1, A5B1, A6B3, A7B3
  st[1] = __builtin_shufflevector(outv[1], outv[0], 0, 9, 2, 11, 4, 13, 6, 15);
  // A0B2, A1B2, A2B0, A3B0, A4B2, A5B2, A6B0, A7B0
  st[2] = __builtin_shufflevector(outv[2], outv[3], 2, 11, 0, 9, 6, 15, 4, 13);
  // A0B3, A1B3, A2B1, A3B1, A4B3, A5B3, A6B1, A7B1
  st[3] = __builtin_shufflevector(outv[3], outv[2], 2, 11, 0, 9, 6, 15, 4, 13);

  // A0B0, A1B0, A2B0, A3B0, A4B0, A5B0, A6B0, A7B0
  st[4] = __builtin_shufflevector(st[0], st[2], 0, 1, 10, 11, 4, 5, 14, 15);
  // A0B1, A1B1, A2B1, A3B1, A4B1, A5B1, A6B1, A7B1
  st[5] = __builtin_shufflevector(st[1], st[3], 0, 1, 10, 11, 4, 5, 14, 15);
  // A0B2, A1B2, A2B2, A3B2, A4B2, A5B2, A6B2, A7B2
  st[6] = __builtin_shufflevector(st[2], st[0], 0, 1, 10, 11, 4, 5, 14, 15);
  // A0B3, A1B3, A2B3, A3B3, A4B3, A5B3, A6B3, A7B3
  st[7] = __builtin_shufflevector(st[3], st[1], 0, 1, 10, 11, 4, 5, 14, 15);

  // Add the output vectors to the output matrix.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < 4; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 8; ++vidx) {
      out[(j + vnum) * outstride + (i + vidx)] += st[4 + vnum][vidx];
    }
  }
}

#ifdef USE_AVX512
// TODO: Fix use of AVX512
const int64_t nVec = 8;
const int64_t mVec = 16;
#else
const int64_t nVec = 4;
const int64_t mVec = 8;
#endif

// Base-case for the divide-and-conquer matmul.
template <typename F, bool transpose_lhs, bool transpose_rhs>
void matmul_base(F *__restrict__ out, const F *__restrict__ lhs,
                 const F *__restrict__ rhs, int64_t m, int64_t n, int64_t k,
                 int64_t mstride, int64_t nstride, int64_t kstride) noexcept {
  // The stride of the output is mstride.
  const int64_t outstride = mstride;

  // Initialize the lhs and rhs buffers from the inputs, transposing
  // the inputs as necessary.
  thread_local F lhsTmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  thread_local F rhsTmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  // thread_local F *lhsTmp = nullptr;
  // if (!lhsTmp) lhsTmp = new (std::align_val_t(64)) F[BASE / sizeof(F)];
  // thread_local F *rhsTmp = nullptr;
  // if (!rhsTmp) rhsTmp = new (std::align_val_t(64)) F[BASE / sizeof(F)];

  buffer_init<F, transpose_lhs, transpose_lhs>(lhsTmp, lhs, m, k, mstride,
                                               kstride);
  buffer_init<F, transpose_rhs, !transpose_rhs>(rhsTmp, rhs, k, n, kstride,
                                                nstride);

  // Handle a vectorizable hyperrectangle whose sides are multiples of
  // nVec and mVec.  This hyperrectangle can be handled fully with
  // vector operations using matmul_vec_op.
  for (int64_t jj = 0; jj < n / nVec; ++jj) {
    for (int64_t ii = 0; ii < m / mVec; ++ii) {
      matmul_vec_op<F>(out, lhsTmp, rhsTmp, mVec * ii, nVec * jj, 0, k, m, n, k,
                       outstride);
    }
  }
  if (mVec * (m / mVec) < m) {
    // Handle extra entries in the m dimension.
    if (2 * (m / mVec) < (m / (mVec / 2))) {
      for (int64_t jj = 0; jj < n / nVec; ++jj) {
        matmul_vec_op_halfvec<F>(out, lhsTmp, rhsTmp, mVec * (m / mVec),
                                 nVec * jj, 0, k, m, n, k, outstride);
      }
    }
    for (int64_t jj = 0; jj < n / nVec; ++jj) {
      matmul_vec_flex_col<F, nVec, mVec / 2>(
          out, lhsTmp, rhsTmp, (mVec / 2) * (m / (mVec / 2)), jj * nVec, 0,
          m - (mVec / 2) * (m / (mVec / 2)), k, m, n, k, outstride);
    }
  }
  if (nVec * (n / nVec) < n) {
    // Handle extra entries in the n dimension.
    for (int64_t ii = 0; ii < m / mVec; ++ii) {
      matmul_vec_flex<F, mVec, nVec>(
          out, lhsTmp, rhsTmp, mVec * ii, nVec * (n / nVec), 0,
          n - (nVec * (n / nVec)), k, m, n, k, outstride);
    }
    // We permute the order of loops here to exploit spatial locality
    // in out and lhs.
    for (int64_t j = nVec * (n / nVec); j < n; ++j) {
      for (int64_t l = 0; l < k; ++l) {
        for (int64_t i = mVec * (m / mVec); i < m; ++i) {
          out[j * outstride + i] += BUF_INDEX(lhsTmp, l, k, i, m, false) *
                                    BUF_INDEX(rhsTmp, j, n, l, k, true);
        }
      }
    }
  }
}

template <typename F, bool transpose_lhs, bool transpose_rhs>
void matmul_dac(F *__restrict__ out, const F *__restrict__ lhs,
                const F *__restrict__ rhs, int64_t m, int64_t n, int64_t k,
                int64_t mstride, int64_t nstride, int64_t kstride) noexcept {
  // if (m == 0 || n == 0 || k == 0)
  //   return;

  // Check that the total size of the submatrices fits within BASE
  // bytes.
  if ((m * n) + (m * k) + (n * k) <= BASE / sizeof(F)) {
    matmul_base<F, transpose_lhs, transpose_rhs>(out, lhs, rhs, m, n, k,
                                                 mstride, nstride, kstride);
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
    cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, split, k,
        mstride, nstride, kstride);
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out + (split * mstride),
        &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, split, nstride, 0, kstride, transpose_rhs), m,
        (n - split), k, mstride, nstride, kstride);
    cilk_sync;
  } else if (max_dim == m) {
    const int64_t split = split_dim(m);
    cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n, k,
        mstride, nstride, kstride);
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out + split, &ARG_INDEX(lhs, 0, kstride, split, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), (m - split), n,
        k, mstride, nstride, kstride);
    cilk_sync;
  } else { // max_dim == k
    const int64_t split = split_dim(k);
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, n, split,
        mstride, nstride, kstride);
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, split, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, split, kstride, transpose_rhs), m, n,
        (k - split), mstride, nstride, kstride);
  }
}

template <typename F>
INLINEATTR void zero_init(F *__restrict__ out, int64_t m, int64_t n,
                          int64_t mstride, int64_t nstride) {
  const int64_t ZI_BASE = 16;
  if ((m * n) <= ZI_BASE * ZI_BASE) {
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i)
        out[j * mstride + i] = 0.0;
    return;
  }

  if (m > n) {
    const int64_t split = split_dim(m);
    cilk_scope {
      cilk_spawn zero_init(out, split, n, mstride, nstride);
      zero_init(out + split, (m - split), n, mstride, nstride);
    }
  } else {
    const int64_t split = split_dim(n);
    cilk_scope {
      cilk_spawn zero_init(out, m, split, mstride, nstride);
      zero_init(out + (split * mstride), m, (n - split), mstride, nstride);
    }
  }
}

template <typename F>
INLINEATTR void matmul(F *__restrict__ out, const F *__restrict__ lhs,
                       const F *__restrict__ rhs, int64_t m, int64_t n,
                       int64_t k, int32_t transpose_lhs,
                       int32_t transpose_rhs) noexcept {
  // Initialize output to zero.
  zero_init(out, m, n, m, n);

  if (transpose_lhs && transpose_rhs) {
    matmul_dac<F, true, true>(out, lhs, rhs, m, n, k, m, n, k);
  } else if (transpose_lhs && !transpose_rhs) {
    matmul_dac<F, true, false>(out, lhs, rhs, m, n, k, m, n, k);
  } else if (!transpose_lhs && transpose_rhs) {
    matmul_dac<F, false, true>(out, lhs, rhs, m, n, k, m, n, k);
  } else {
    matmul_dac<F, false, false>(out, lhs, rhs, m, n, k, m, n, k);
  }
}

template <typename F>
INLINEATTR void matmul_ploops(F *__restrict__ out, const F *__restrict__ lhs,
                              const F *__restrict__ rhs, int64_t m, int64_t n,
                              int64_t k, int32_t transpose_lhs,
                              int32_t transpose_rhs) {
  if (n > m) {
    cilk_for(int64_t i = 0; i < m; ++i) {
      cilk_for(int64_t j = 0; j < n; ++j) {
        out[j * m + i] = 0.0;
        for (int64_t l = 0; l < k; ++l)
          out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, transpose_lhs) *
                            ARG_INDEX(rhs, j, n, l, k, transpose_rhs);
      }
    }
  } else {
    cilk_for(int64_t j = 0; j < n; ++j) {
      cilk_for(int64_t i = 0; i < m; ++i) {
        out[j * m + i] = 0.0;
        for (int64_t l = 0; l < k; ++l)
          out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, transpose_lhs) *
                            ARG_INDEX(rhs, j, n, l, k, transpose_rhs);
      }
    }
  }
}

template void matmul_ploops<float>(float *__restrict__ out,
                                   const float *__restrict__ lhs,
                                   const float *__restrict__ rhs, int64_t m,
                                   int64_t n, int64_t k, int32_t transpose_lhs,
                                   int32_t transpose_rhs);

template void matmul_ploops<double>(double *__restrict__ out,
                                    const double *__restrict__ lhs,
                                    const double *__restrict__ rhs, int64_t m,
                                    int64_t n, int64_t k, int32_t transpose_lhs,
                                    int32_t transpose_rhs);

extern "C" {
INLINEATTR
void matmul_f32(float *__restrict__ out, const float *__restrict__ lhs,
                const float *__restrict__ rhs, int64_t m, int64_t n, int64_t k,
                int32_t transpose_lhs, int32_t transpose_rhs) {
#ifndef NDEBUG
  matmul<float>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
#else
  matmul<float>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
#endif
}

INLINEATTR
void matmul_f64(double *__restrict__ out, const double *__restrict__ lhs,
                const double *__restrict__ rhs, int64_t m, int64_t n, int64_t k,
                int32_t transpose_lhs, int32_t transpose_rhs) {
#ifndef NDEBUG
  matmul<double>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
#else
  matmul<double>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
#endif
}
}
