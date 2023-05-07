// Compile string, using Opencilk clang++:
// clang++ -c matmul.cpp -emit-llvm -fopencilk -ftapir=none -std=c++17 -ffast-math  -mavx -mfma -mavx2 # -mavx512f -mavx512cd -O3

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cilk/cilk.h>
// #include <cilk/cilk_stub.h>
#include <cstdio>

#include "common_sysdep.hpp"
#include "matmul_common.hpp"

__attribute__((always_inline)) static int64_t split_dim(int64_t n) {
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
  // const int64_t TBASE = 4;
  const int64_t TBASE = nVec;
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

template <typename F>
__attribute__((always_inline)) static void
buffer_transpose_v(F *__restrict__ dst, const F *__restrict__ src, int64_t x,
                 int64_t y, int64_t stride) {
  typedef F vF __attribute__((vector_size(sizeof(F) * nVec)));
  const int64_t TBASE = nVec;
  vF in[TBASE], out[3 * TBASE];
  for (int64_t jj = 0; jj < y / TBASE; ++jj) {
    // __builtin_prefetch(&src[((jj + 1) * TBASE) * stride], 0, 2);

#pragma clang loop unroll_count(2)
    for (int64_t ii = 0; ii < x / TBASE; ++ii) {
      // // __builtin_prefetch(&src[(((jj + 1) * TBASE) * stride + (ii * TBASE))], 0, 1);
      // __builtin_prefetch(&BUF_INDEX(dst, (jj * TBASE), y, ((ii + 1) * TBASE), x, true), 1, 3);

      // Load TBASE == nVec columns of input from src using vector
      // loads.  These vector loads may be unaligned according to the
      // vector width.
      for (int64_t vnum = 0; vnum < TBASE; ++vnum) {
	int64_t j = (jj * TBASE) + vnum;
	// in[vnum] = *reinterpret_cast<const vF *>(src + (j * stride) + (ii * TBASE));
	// __builtin_prefetch(&src[j * stride + ((ii + 1) * TBASE)]);
	// __builtin_prefetch(&src[(j + TBASE) * stride + (ii * TBASE)], 0, 1);
	// __builtin_prefetch(&BUF_INDEX(dst, (jj * TBASE), y, (ii * TBASE) + vnum, x, true), 1, 3);
#pragma clang loop unroll(full)
	for (int64_t vidx = 0; vidx < nVec; ++vidx) {
	  in[vnum][vidx] = src[j * stride + (ii * TBASE) + vidx];
	}
      }

      // The input vectors store a square submatrix of src in
      // column-major order: each vector stores a column of the
      // submatrix.  Transpose the submatrix to generate output
      // vectors storing rows of the submatrix.

      // out[0] = __builtin_shufflevector(in[0], in[1], 0, 8, 2, 10, 4, 12, 6, 14);
      // out[2] = __builtin_shufflevector(in[2], in[3], 0, 8, 2, 10, 4, 12, 6, 14);
      // out[4] = __builtin_shufflevector(in[4], in[5], 0, 8, 2, 10, 4, 12, 6, 14);
      // out[6] = __builtin_shufflevector(in[6], in[7], 0, 8, 2, 10, 4, 12, 6, 14);
      // out[1] = __builtin_shufflevector(in[0], in[1], 1, 9, 3, 11, 5, 13, 7, 15);
      // out[3] = __builtin_shufflevector(in[2], in[3], 1, 9, 3, 11, 5, 13, 7, 15);
      // out[5] = __builtin_shufflevector(in[4], in[5], 1, 9, 3, 11, 5, 13, 7, 15);
      // out[7] = __builtin_shufflevector(in[6], in[7], 1, 9, 3, 11, 5, 13, 7, 15);

      // out[8]  = __builtin_shufflevector(out[0], out[2], 0, 1, 8, 9, 4, 5, 12, 13);
      // out[9]  = __builtin_shufflevector(out[1], out[3], 0, 1, 8, 9, 4, 5, 12, 13);
      // out[12] = __builtin_shufflevector(out[4], out[6], 0, 1, 8, 9, 4, 5, 12, 13);
      // out[13] = __builtin_shufflevector(out[5], out[7], 0, 1, 8, 9, 4, 5, 12, 13);
      // out[10] = __builtin_shufflevector(out[0], out[2], 2, 3, 10, 11, 6, 7, 14, 15);
      // out[11] = __builtin_shufflevector(out[1], out[3], 2, 3, 10, 11, 6, 7, 14, 15);
      // out[14] = __builtin_shufflevector(out[4], out[6], 2, 3, 10, 11, 6, 7, 14, 15);
      // out[15] = __builtin_shufflevector(out[5], out[7], 2, 3, 10, 11, 6, 7, 14, 15);

      // out[16] = __builtin_shufflevector(out[8],  out[12], 0, 1, 2, 3, 8, 9, 10, 11);
      // out[17] = __builtin_shufflevector(out[9],  out[13], 0, 1, 2, 3, 8, 9, 10, 11);
      // out[18] = __builtin_shufflevector(out[10], out[14], 0, 1, 2, 3, 8, 9, 10, 11);
      // out[19] = __builtin_shufflevector(out[11], out[15], 0, 1, 2, 3, 8, 9, 10, 11);
      // out[20] = __builtin_shufflevector(out[8],  out[12], 4, 5, 6, 7, 12, 13, 14, 15);
      // out[21] = __builtin_shufflevector(out[9],  out[13], 4, 5, 6, 7, 12, 13, 14, 15);
      // out[22] = __builtin_shufflevector(out[10], out[14], 4, 5, 6, 7, 12, 13, 14, 15);
      // out[23] = __builtin_shufflevector(out[11], out[15], 4, 5, 6, 7, 12, 13, 14, 15);

      out[0] = __builtin_shufflevector(in[0], in[4], 0, 1, 2, 3, 8, 9, 10, 11);
      out[1] = __builtin_shufflevector(in[1], in[5], 0, 1, 2, 3, 8, 9, 10, 11);
      out[2] = __builtin_shufflevector(in[2], in[6], 0, 1, 2, 3, 8, 9, 10, 11);
      out[3] = __builtin_shufflevector(in[3], in[7], 0, 1, 2, 3, 8, 9, 10, 11);
      out[4] = __builtin_shufflevector(in[0], in[4], 4, 5, 6, 7, 12, 13, 14, 15);
      out[5] = __builtin_shufflevector(in[1], in[5], 4, 5, 6, 7, 12, 13, 14, 15);
      out[6] = __builtin_shufflevector(in[2], in[6], 4, 5, 6, 7, 12, 13, 14, 15);
      out[7] = __builtin_shufflevector(in[3], in[7], 4, 5, 6, 7, 12, 13, 14, 15);

      // __builtin_prefetch(
      //     &src[((jj * TBASE) + 0) * stride + ((ii + 1) * TBASE)]);
      out[8]  = __builtin_shufflevector(out[0], out[2], 0, 1, 8, 9, 4, 5, 12, 13);
      // __builtin_prefetch(
      //     &src[((jj * TBASE) + 1) * stride + ((ii + 1) * TBASE)]);
      out[9]  = __builtin_shufflevector(out[1], out[3], 0, 1, 8, 9, 4, 5, 12, 13);
      // __builtin_prefetch(
      //     &src[((jj * TBASE) + 2) * stride + ((ii + 1) * TBASE)]);
      out[12] = __builtin_shufflevector(out[4], out[6], 0, 1, 8, 9, 4, 5, 12, 13);
      // __builtin_prefetch(
      //     &src[((jj * TBASE) + 3) * stride + ((ii + 1) * TBASE)]);
      out[13] = __builtin_shufflevector(out[5], out[7], 0, 1, 8, 9, 4, 5, 12, 13);
      // __builtin_prefetch(
      //     &src[((jj * TBASE) + 4) * stride + ((ii + 1) * TBASE)]);
      out[10] = __builtin_shufflevector(out[0], out[2], 2, 3, 10, 11, 6, 7, 14, 15);
      // __builtin_prefetch(
      //     &src[((jj * TBASE) + 5) * stride + ((ii + 1) * TBASE)]);
      out[11] = __builtin_shufflevector(out[1], out[3], 2, 3, 10, 11, 6, 7, 14, 15);
      // __builtin_prefetch(
      //     &src[((jj * TBASE) + 6) * stride + ((ii + 1) * TBASE)]);
      out[14] = __builtin_shufflevector(out[4], out[6], 2, 3, 10, 11, 6, 7, 14, 15);
      // __builtin_prefetch(
      //     &src[((jj * TBASE) + 7) * stride + ((ii + 1) * TBASE)]);
      out[15] = __builtin_shufflevector(out[5], out[7], 2, 3, 10, 11, 6, 7, 14, 15);

      out[16] = __builtin_shufflevector(out[8],  out[9],  0, 8, 2, 10, 4, 12, 6, 14);
      out[18] = __builtin_shufflevector(out[10], out[11], 0, 8, 2, 10, 4, 12, 6, 14);
      out[20] = __builtin_shufflevector(out[12], out[13], 0, 8, 2, 10, 4, 12, 6, 14);
      out[22] = __builtin_shufflevector(out[14], out[15], 0, 8, 2, 10, 4, 12, 6, 14);
      out[17] = __builtin_shufflevector(out[8],  out[9],  1, 9, 3, 11, 5, 13, 7, 15);
      out[19] = __builtin_shufflevector(out[10], out[11], 1, 9, 3, 11, 5, 13, 7, 15);
      out[21] = __builtin_shufflevector(out[12], out[13], 1, 9, 3, 11, 5, 13, 7, 15);
      out[23] = __builtin_shufflevector(out[14], out[15], 1, 9, 3, 11, 5, 13, 7, 15);

      // Store the submatrix rows into dst using vectorized stores.
      // The alignment of dst ensures that these stores are aligned
      // according to the vector width.
      for (int64_t vnum = 0; vnum < TBASE; ++vnum) {
        *reinterpret_cast<vF *>(
            &BUF_INDEX(dst, (jj * TBASE), y, (ii * TBASE) + vnum, x, true)) =
            out[(2 * TBASE) + vnum];
      }
    }
  }
  if (x > (x / TBASE) * TBASE) {
    for (int64_t jj = 0; jj < y / TBASE; ++jj) {
      for (int64_t i = (x / TBASE) * TBASE; i < x; ++i) {
	for (int64_t j = (jj * TBASE); j < ((jj + 1) * TBASE); ++j) {
	  BUF_INDEX(dst, j, y, i, x, true) = src[j * stride + i];
	}
      }
    }
  }
  for (int64_t j = (y / TBASE) * TBASE; j < y; ++j) {
    for (int64_t i = 0; i < x; ++i) {
      BUF_INDEX(dst, j, y, i, x, true) = src[j * stride + i];
    }
  }
}

template <typename F>
__attribute__((always_inline)) static void
buffer_transpose_v4(F *__restrict__ dst, const F *__restrict__ src, int64_t x,
		    int64_t y, int64_t stride) {
  const int64_t nVec = 4;
  typedef F vF __attribute__((vector_size(sizeof(F) * nVec)));
  const int64_t TBASE = nVec;
  vF in[TBASE], out[2 * TBASE];
  for (int64_t jj = 0; jj < y / TBASE; ++jj) {
#pragma clang loop unroll_count(4)
    for (int64_t ii = 0; ii < x / TBASE; ++ii) {

      // Load TBASE == nVec columns of input from src using vector
      // loads.  These vector loads may be unaligned according to the
      // vector width.
      for (int64_t vnum = 0; vnum < TBASE; ++vnum) {
	int64_t j = (jj * TBASE) + vnum;
#pragma clang loop unroll(full)
	for (int64_t vidx = 0; vidx < nVec; ++vidx) {
	  in[vnum][vidx] = src[j * stride + (ii * TBASE) + vidx];
	}
      }

      // The input vectors store a square submatrix of src in
      // column-major order: each vector stores a column of the
      // submatrix.  Transpose the submatrix to generate output
      // vectors storing rows of the submatrix.

      out[0] = __builtin_shufflevector(in[0], in[2], 0, 1, 4, 5);
      out[1] = __builtin_shufflevector(in[1], in[3], 0, 1, 4, 5);
      out[2] = __builtin_shufflevector(in[0], in[2], 2, 3, 6, 7);
      out[3] = __builtin_shufflevector(in[1], in[3], 2, 3, 6, 7);

      out[4] = __builtin_shufflevector(out[0], out[1], 0, 4, 2, 6);
      out[5] = __builtin_shufflevector(out[0], out[1], 1, 5, 3, 7);
      out[6] = __builtin_shufflevector(out[2], out[3], 0, 4, 2, 6);
      out[7] = __builtin_shufflevector(out[2], out[3], 1, 5, 3, 7);

      // Store the submatrix rows into dst using vectorized stores.
      // The alignment of dst ensures that these stores are aligned
      // according to the vector width.
      for (int64_t vnum = 0; vnum < TBASE; ++vnum) {
        *reinterpret_cast<vF *>(
            &BUF_INDEX(dst, (jj * TBASE), y, (ii * TBASE) + vnum, x, true)) =
            out[TBASE + vnum];
        // #pragma clang loop unroll(full)
        // 	for (int64_t vidx = 0; vidx < nVec; ++vidx) {
        //           BUF_INDEX(dst, (jj * TBASE) + vidx, y, (ii * TBASE) + vnum,
        //           x, true) =
        //               out[TBASE + vnum][vidx];
        //         }
      }

    }
  }
  if (x > (x / TBASE) * TBASE) {
    for (int64_t jj = 0; jj < y / TBASE; ++jj) {
      for (int64_t i = (x / TBASE) * TBASE; i < x; ++i) {
	for (int64_t j = (jj * TBASE); j < ((jj + 1) * TBASE); ++j) {
	  BUF_INDEX(dst, j, y, i, x, true) = src[j * stride + i];
	}
      }
    }
  }
  for (int64_t j = (y / TBASE) * TBASE; j < y; ++j) {
    for (int64_t i = 0; i < x; ++i) {
      BUF_INDEX(dst, j, y, i, x, true) = src[j * stride + i];
    }
  }
}

template <typename F, typename vF, int64_t vWidth, int64_t numVec, bool transpose>
__attribute__((always_inline)) static void
buffer_copy_vec(F *__restrict__ dst, const F *__restrict__ src, int64_t i,
		int64_t j, int64_t x, int64_t y, int64_t stride) {
  vF tmp[numVec];
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < numVec; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < vWidth; ++vidx) {
      tmp[vnum][vidx] =
          src[j * stride + i + (vnum * vWidth) + vidx];
    }
  }
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < numVec; ++vnum) {
    *reinterpret_cast<vF *>(&BUF_INDEX(
        dst, j, y, i + (vnum * vWidth), x, transpose)) = tmp[vnum];
  }
}

template <typename F, bool transpose, int64_t x>
__attribute__((always_inline)) static void
buffer_copy_block(F *__restrict__ dst, const F *__restrict__ src, int64_t y,
                  int64_t stride) {
  const int64_t vWidth = mVec;
  typedef F vF __attribute__((vector_size(sizeof(F) * vWidth)));

  const int64_t CBASE_x = 1 * mVBlk; // 3 * mVec;
  const int64_t numVec = CBASE_x / vWidth;
  const int64_t JUnroll = 4; // 16;

#pragma clang loop unroll_count(JUnroll)
  for (int64_t j = 0; j < y; ++j) {
    for (int64_t ii = 0; ii < x / CBASE_x; ++ii) {
      buffer_copy_vec<F, vF, vWidth, numVec, transpose>(dst, src, ii * CBASE_x,
                                                        j, x, y, stride);
    }
  }
}

template <typename F, int64_t xBase, int64_t yBase, bool transpose>
__attribute__((always_inline)) static void
buffer_copy(F *__restrict__ dst, const F *__restrict__ src, int64_t x,
            int64_t y, int64_t stride) {
  const int64_t vWidth = mVec;
  typedef F vF __attribute__((vector_size(sizeof(F) * vWidth)));

  const int64_t CBASE = yBase; // kBlk; // 128; // kBlk; // 2 * 8; // 2 * mVec;
  const int64_t CBASE_x = xBase; // 2 * mVBlk;
  const int64_t numVec = CBASE_x / vWidth;
  const int64_t JUnroll = 8; // 4;

  for (int64_t jj = 0; jj < y / CBASE; ++jj) {
    for (int64_t ii = 0; ii < x / CBASE_x; ++ii) {
#pragma clang loop unroll_count(JUnroll)
      for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
        buffer_copy_vec<F, vF, vWidth, numVec, transpose>(dst, src, (ii * CBASE_x), j, x, y, stride);
      }
    }

    if (CBASE_x * (x / CBASE_x) < x) {
      int64_t i = CBASE_x * (x / CBASE_x);
      int64_t xRem = x - i;
      const int64_t xBlk5 = 5 * vWidth;
      const int64_t xBlk4 = 4 * vWidth;
      const int64_t xBlk3 = 3 * vWidth;
      const int64_t xBlk2 = 2 * vWidth;
      const int64_t xBlk1 = 1 * vWidth;
      switch (xRem / xBlk5) {
      case 5: {
#pragma clang loop unroll_count(JUnroll)
	for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
	  buffer_copy_vec<F, vF, vWidth, (xBlk5 / vWidth), transpose>(dst, src, i, j, x, y, stride);
	}
	i += xBlk5;
	xRem -= xBlk5;
	break;
      }
      case 4: {
#pragma clang loop unroll_count(JUnroll)
	for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
	  buffer_copy_vec<F, vF, vWidth, (xBlk4 / vWidth), transpose>(dst, src, i, j, x, y, stride);
	}
	i += xBlk4;
	xRem -= xBlk4;
	break;
      }
      case 3: {
#pragma clang loop unroll_count(JUnroll)
	for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
	  buffer_copy_vec<F, vF, vWidth, (xBlk3 / vWidth), transpose>(dst, src, i, j, x, y, stride);
	}
	i += xBlk3;
	xRem -= xBlk3;
	break;
      }
      case 2: {
#pragma clang loop unroll_count(JUnroll)
	for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
	  buffer_copy_vec<F, vF, vWidth, (xBlk2 / vWidth), transpose>(dst, src, i, j, x, y, stride);
	}
	i += xBlk2;
	xRem -= xBlk2;
	break;
      }
      case 1: {
#pragma clang loop unroll_count(JUnroll)
	for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
	  buffer_copy_vec<F, vF, vWidth, (xBlk1 / vWidth), transpose>(dst, src, i, j, x, y, stride);
	}
	i += xBlk1;
	xRem -= xBlk1;
	break;
      }
      }

      if (xRem) {
#pragma clang loop unroll_count(JUnroll)
	for (int64_t j = (jj * CBASE); j < ((jj + 1) * CBASE); ++j) {
	  for (int64_t iRem = i; iRem < x; ++iRem) {
	    BUF_INDEX(dst, j, y, iRem, x, transpose) = src[j * stride + iRem];
	  }
	}
      }
    }
  }

  if ((y / CBASE) * CBASE == y)
    return;

  for (int64_t ii = 0; ii < x / CBASE_x; ++ii) {
    for (int64_t j = (y / CBASE) * CBASE; j < y; ++j) {
      buffer_copy_vec<F, vF, vWidth, numVec, transpose>(dst, src, (ii * CBASE_x), j, x, y, stride);
    }
  }
  for (int64_t j = (y / CBASE) * CBASE; j < y; ++j) {
    for (int64_t i = (x / CBASE_x) * CBASE_x; i < x; ++i) {
      BUF_INDEX(dst, j, y, i, x, transpose) = src[j * stride + i];
    }
  }
}

template <typename F, int64_t m, int64_t n, bool transposed, bool want_transpose>
static void buffer_init_block(F *__restrict__ dst, const F *__restrict__ src,
			      int64_t mstride, int64_t nstride) {
  if (!want_transpose) {
    if (!transposed) {
      buffer_copy_block<F, false, m>(dst, src, n, mstride);
    } else {
      buffer_copy_block<F, false, n>(dst, src, m, nstride);
    }
  } else {
    if (!transposed) {
      // buffer_transpose(dst, src, m, n, mstride);
      // buffer_transpose_v4(dst, src, m, n, mstride);
      buffer_transpose_v(dst, src, m, n, mstride);
    } else {
      buffer_transpose(dst, src, n, m, nstride);
    }
  }
}

template <typename F, int64_t mBase, int64_t nBase, bool transposed,
          bool want_transpose>
static void buffer_init(F *__restrict__ dst, const F *__restrict__ src,
                        int64_t m, int64_t n, int64_t mstride,
                        int64_t nstride) {
  if (!want_transpose) {
#if PRINT
    fprintf(stderr, "  buffer_copy from %p\n", src);
#endif
    if (!transposed) {
      buffer_copy<F, mBase, nBase, false>(dst, src, m, n, mstride);
    } else {
      buffer_copy<F, nBase, mBase, false>(dst, src, n, m, nstride);
    }
  } else {
#if PRINT
    fprintf(stderr, "  buffer_transpose from %p\n", src);
#endif
    if (!transposed) {
      // buffer_transpose(dst, src, m, n, mstride);
      // buffer_transpose_v4(dst, src, m, n, mstride);
      buffer_transpose_v(dst, src, m, n, mstride);
    } else {
      buffer_transpose(dst, src, n, m, nstride);
    }
  }
}

// template <typename F, bool transposed, bool want_transpose>
// // __attribute__((always_inline))
// void
// buffer_init(F *__restrict__ dst, const F *__restrict__ src, int64_t m,
//             int64_t n, int64_t mstride, int64_t nstride);

// A simple and general vectorized base case for matrix multiply.
// This base case computes a INum x JNum submatrix in column-major
// order from a INum subcolumn of A and a JNum subrow of B.
template <typename F, int64_t INum, int64_t JNum, bool transpose_lhs,
          bool transpose_rhs>
__attribute__((always_inline)) void
matmul_vec(F *__restrict__ out, const F *__restrict__ lhs,
           const F *__restrict__ rhs, int64_t i, int64_t j, int64_t l,
           int64_t KNum, int64_t mstride, int64_t nstride,
           int64_t kstride, int64_t outstride) noexcept {
  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F) * INum)));
  vF outv[JNum];

  // Zero-initialize output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum)
    outv[vnum] = (vF){0.0};

#pragma clang loop unroll_count(8)
  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {

    // Get INum values from a column of lhs.
    vF lhsv;
    lhsv = *reinterpret_cast<const vF *>(
        &BUF_INDEX(lhs, my_l, kstride, i, mstride, transpose_lhs));

    // Each output vector gets the element-wise product of lhsv and one
    // of the rhsv.
#pragma clang loop unroll(full)
    for (int64_t vnum = 0; vnum < JNum; ++vnum)
      outv[vnum] += lhsv * BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride,
                                     transpose_rhs);
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
template <typename F, int64_t NumMVec, int64_t JNum, bool transpose_lhs,
          bool transpose_rhs>
__attribute__((always_inline)) void
matmul_vec_x8(F *__restrict__ out, const F *__restrict__ lhs,
              const F *__restrict__ rhs, int64_t i, int64_t j, int64_t l,
              int64_t KNum, int64_t mstride, int64_t nstride, int64_t kstride,
              int64_t outstride) noexcept {
  // Vector type
  const int64_t vWidth = mVec;
  typedef F vF __attribute__((vector_size(sizeof(F) * vWidth)));
  vF outv[NumMVec][JNum];

  // Zero-initialize output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      outv[mvnum][vnum] = (vF){0.0};
    }
  }

  const int64_t kUnroll = 4;
  const int64_t kRem = NumMVec * JNum; // 32;
  // const F *lhsprefetch =
  //     &BUF_INDEX(lhs, l, kstride, i + mVBlk, mstride, transpose_lhs);
  for (int64_t ll = 0; ll < (KNum / kBlk); ++ll) {
    for (int64_t my_l = l + (ll * kBlk); my_l < l + ((ll + 1) * kBlk) - kRem;
         my_l += kUnroll) {
      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll, kstride, transpose_rhs));
      // __builtin_prefetch(
      //     &BUF_INDEX(lhs, my_l + kRem, kstride, i, mstride, transpose_lhs), 0,
      //     3);
      // __builtin_prefetch(
      //     &BUF_INDEX(lhs, my_l + kUnroll, kstride, i, mstride, transpose_lhs),
      //     0, 1);

      // Get values from a column of lhs.
      vF lhsv[NumMVec];
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, transpose_rhs);
#pragma clang loop unroll(full)
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(&BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 1, kstride,
				    transpose_rhs));
      // __builtin_prefetch(&BUF_INDEX(lhs, my_l + kRem, kstride, i + (1 * vWidth),
      //                               mstride, transpose_lhs),
      //                    0, 3);

      // Get values from a column of lhs.
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 1, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 1, kstride, transpose_rhs);
#pragma clang loop unroll(full)
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 2, kstride, transpose_rhs));
      // __builtin_prefetch(lhsprefetch, 0, 1);
      // lhsprefetch += vWidth;

      // __builtin_prefetch(&BUF_INDEX(lhs, my_l + kRem, kstride, i + (2 * vWidth),
      //                               mstride, transpose_lhs),
      //                    0, 3);
      // __builtin_prefetch(&BUF_INDEX(lhs, my_l + (kUnroll * 6), kstride, i,
      //                               mstride, transpose_lhs),
      //                    0, 3);
      // __builtin_prefetch(&BUF_INDEX(lhs, my_l + (4 * kUnroll), kstride, i,
      //                               mstride, transpose_lhs),
      //                    0, 3);
      // __builtin_prefetch(&BUF_INDEX(lhs, my_l, kstride, i + mVBlk,
      //                               mstride, transpose_lhs),
      //                    0, 1);

      // Get values from a column of lhs.
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 2, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 2, kstride, transpose_rhs);
#pragma clang loop unroll(full)
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 3, kstride, transpose_rhs));

      // Get values from a column of lhs.
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 3, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 3, kstride, transpose_rhs);
#pragma clang loop unroll(full)
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }
    }

    // Handle the final KRem iterations while prefetching the output
    // locations.
    int64_t out_prefetch_num = 0;
    for (int64_t my_l = l + ((ll + 1) * kBlk) - kRem;
         my_l < l + ((ll + 1) * kBlk); my_l += kUnroll) {
      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll, kstride, transpose_rhs));
      // __builtin_prefetch(
      //     &BUF_INDEX(lhs, my_l + kUnroll, kstride, i, mstride, transpose_lhs),
      //     0, 1);

      // Get values from a column of lhs.
      vF lhsv[NumMVec];
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      __builtin_prefetch(&out[(j + (out_prefetch_num / NumMVec)) * outstride +
                              (i + ((out_prefetch_num % NumMVec) * vWidth))],
                         1, 3);
      out_prefetch_num++;

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, transpose_rhs);
#pragma clang loop unroll(full)
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	  // __builtin_prefetch(&out[(j + vnum) * outstride + (i + (mvnum * vWidth))], 1, 3);
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(&BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 1, kstride,
				    transpose_rhs));

      // Get values from a column of lhs.
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 1, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      __builtin_prefetch(&out[(j + (out_prefetch_num / NumMVec)) * outstride +
                              (i + ((out_prefetch_num % NumMVec) * vWidth))],
                         1, 3);
      out_prefetch_num++;

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 1, kstride, transpose_rhs);
#pragma clang loop unroll(full)
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 2, kstride, transpose_rhs));
      // __builtin_prefetch(&BUF_INDEX(lhs, my_l + (4 * kUnroll), kstride, i,
      //                               mstride, transpose_lhs),
      //                    0, 3);
      // __builtin_prefetch(&BUF_INDEX(lhs, my_l, kstride, i + mVBlk,
      //                               mstride, transpose_lhs),
      //                    0, 1);

      // Get values from a column of lhs.
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 2, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      __builtin_prefetch(&out[(j + (out_prefetch_num / NumMVec)) * outstride +
                              (i + ((out_prefetch_num % NumMVec) * vWidth))],
                         1, 3);
      out_prefetch_num++;

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 2, kstride, transpose_rhs);
#pragma clang loop unroll(full)
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 3, kstride, transpose_rhs));

      // Get values from a column of lhs.
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 3, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      __builtin_prefetch(&out[(j + (out_prefetch_num / NumMVec)) * outstride +
                              (i + ((out_prefetch_num % NumMVec) * vWidth))],
                         1, 3);
      out_prefetch_num++;

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 3, kstride, transpose_rhs);
#pragma clang loop unroll(full)
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }
    }
  }

  // Handle any remaining elements in a non-full k-block.
  for (int64_t my_l = l + ((KNum / kBlk) * kBlk); my_l < l + KNum; ++my_l) {
    // Get values from a column of lhs.
    vF lhsv[NumMVec];
#pragma clang loop unroll(full)
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
          lhs, my_l, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
    }

    // Each output vector gets the element-wise product of lhsv and
    // one of the rhsv.
#pragma clang loop unroll(full)
    for (int64_t vnum = 0; vnum < JNum; ++vnum) {
      F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, transpose_rhs);
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
      }
    }
  }

  // Add the output vectors to the output matrix.
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
#pragma clang loop unroll(full)
      for (int64_t vidx = 0; vidx < vWidth; ++vidx) {
        out[(j + vnum) * outstride + (i + (mvnum * vWidth) + vidx)] +=
            outv[mvnum][vnum][vidx];
      }
    }
  }
}

// A simple and general vectorized base case for matrix multiply.
// This base case computes a INum x JNum submatrix in column-major
// order from a INum subcolumn of A and a JNum subrow of B.
template <typename F, int64_t NumMVec, int64_t JNum, int64_t KNum,
          bool transpose_lhs, bool transpose_rhs>
__attribute__((always_inline))
void matmul_vec_x8_block(F *__restrict__ out, const F *__restrict__ lhs,
                         const F *__restrict__ rhs, int64_t i, int64_t j,
                         int64_t mstride, int64_t nstride,
                         int64_t kstride, int64_t outstride) noexcept {
  // Vector type
  const int64_t vWidth = mVec;
  typedef F vF __attribute__((vector_size(sizeof(F) * vWidth)));
  vF outv[NumMVec][JNum];

  // Zero-initialize output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
    // __builtin_prefetch(&out[(j + vnum) * outstride + i], 1, 2);
#pragma clang loop unroll(full)
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      // __builtin_prefetch(&out[(j + vnum) * outstride + (i + (mvnum * vWidth))],
      //                    1, 2);
      outv[mvnum][vnum] = (vF){0.0};
    }
  }

  const int64_t kUnroll = 4;
  const int64_t kRem = kUnroll * JNum; // 32;
  for (int64_t ll = 0; ll < (KNum / kBlk); ++ll) {
    // __builtin_prefetch(
    //     &BUF_INDEX(lhs, l + ((ll + 1) * kBlk), kstride, i, mstride, false), 0,
    //     1);
    for (int64_t my_l = (ll * kBlk); my_l < ((ll + 1) * kBlk) - kRem; my_l += kUnroll) {
      __builtin_prefetch(&BUF_INDEX(rhs, j, nstride, my_l + kUnroll, kstride,
				    transpose_rhs));
      // __builtin_prefetch(
      //     &BUF_INDEX(lhs, my_l + kUnroll, kstride, i, mstride, transpose_lhs),
      //     0, 1);

      // Get values from a column of lhs.
      vF lhsv[NumMVec];
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, transpose_rhs);
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(&BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 1, kstride,
				    transpose_rhs));

      // Get values from a column of lhs.
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 1, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 1, kstride, transpose_rhs);
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 2, kstride, transpose_rhs));
      __builtin_prefetch(&BUF_INDEX(lhs, my_l + (4 * kUnroll), kstride, i,
                                    mstride, transpose_lhs),
                         0, 3);
      // __builtin_prefetch(&BUF_INDEX(lhs, my_l, kstride, i + mVBlk,
      //                               mstride, transpose_lhs),
      //                    0, 1);

      // Get values from a column of lhs.
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 2, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 2, kstride, transpose_rhs);
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 3, kstride, transpose_rhs));

      // Get values from a column of lhs.
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 3, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 3, kstride, transpose_rhs);
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

    }

    // Handle the final KRem iterations while prefetching the output
    // locations.
    int64_t out_prefetch_num = 0;
    for (int64_t my_l = ((ll + 1) * kBlk) - kRem; my_l < ((ll + 1) * kBlk);
         my_l += kUnroll) {
      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll, kstride, transpose_rhs));
      // __builtin_prefetch(
      //     &BUF_INDEX(lhs, my_l + kUnroll, kstride, i, mstride, transpose_lhs),
      //     0, 1);

      // Get values from a column of lhs.
      vF lhsv[NumMVec];
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      __builtin_prefetch(&out[(j + (out_prefetch_num / NumMVec)) * outstride +
                              (i + ((out_prefetch_num % NumMVec) * vWidth))],
                         1, 3);
      out_prefetch_num++;

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, transpose_rhs);
        // __builtin_prefetch(&out[(j + vnum) * outstride + (i)], 1, 2);
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	  // __builtin_prefetch(&out[(j + vnum) * outstride + (i + (mvnum * vWidth))], 1, 3);
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(&BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 1, kstride,
				    transpose_rhs));

      // Get values from a column of lhs.
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 1, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 1, kstride, transpose_rhs);
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 2, kstride, transpose_rhs));
      __builtin_prefetch(&BUF_INDEX(lhs, my_l + (4 * kUnroll), kstride, i,
                                    mstride, transpose_lhs),
                         0, 3);
      // __builtin_prefetch(&BUF_INDEX(lhs, my_l, kstride, i + mVBlk,
      //                               mstride, transpose_lhs),
      //                    0, 1);

      // Get values from a column of lhs.
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 2, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 2, kstride, transpose_rhs);
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }

      __builtin_prefetch(
          &BUF_INDEX(rhs, j, nstride, my_l + kUnroll + 3, kstride, transpose_rhs));

      // Get values from a column of lhs.
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l + 3, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
      }

      // Each output vector gets the element-wise product of lhsv and one
      // of the rhsv.
#pragma clang loop unroll(full)
      for (int64_t vnum = 0; vnum < JNum; ++vnum) {
	F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l + 3, kstride, transpose_rhs);
        for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
          outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
        }
      }
    }
  }

//   // Handle any remaining elements in a non-full k-block.
//   for (int64_t my_l = l + ((KNum / kBlk) * kBlk); my_l < l + KNum; ++my_l) {
//     // Get values from a column of lhs.
//     vF lhsv[NumMVec];
//     for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
//       lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
//           lhs, my_l, kstride, i + (mvnum * vWidth), mstride, transpose_lhs));
//     }

//     // Each output vector gets the element-wise product of lhsv and
//     // one of the rhsv.
// #pragma clang loop unroll(full)
//     for (int64_t vnum = 0; vnum < JNum; ++vnum) {
//       F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, transpose_rhs);
//       for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
// 	outv[mvnum][vnum] += lhsv[mvnum] * rhs_val;
//       }
//     }
//   }

  // Add the output vectors to the output matrix.
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
#pragma clang loop unroll(full)
      for (int64_t vidx = 0; vidx < vWidth; ++vidx) {
        out[(j + vnum) * outstride + (i + (mvnum * vWidth) + vidx)] +=
            outv[mvnum][vnum][vidx];
      }
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

#pragma clang loop unroll_count(16)
  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
    // Get INum values from a column of lhs.
    vF lhsv;
// #pragma clang loop unroll(full)
//     for (int64_t vidx = 0; vidx < INum; ++vidx) {
//       lhsv[vidx] = BUF_INDEX(lhs, my_l, kstride, i + vidx, mstride, false);
//     }
    lhsv = *reinterpret_cast<const vF *>(
        &BUF_INDEX(lhs, my_l, kstride, i, mstride, false));

//     // Fill each rhs vector with a value from one of INum rows of rhs.
//     vF rhsv[JNum];
//     for (int64_t vnum = 0; vnum < JNum; ++vnum) {
//       // Read the value from a row of rhs.
//       F rhs_val = BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, true);
//       // Broadcast that value through one of the rhsv.
// #pragma clang loop unroll(full)
//       for (int64_t vidx = 0; vidx < INum; ++vidx) {
//         rhsv[vnum][vidx] = rhs_val;
//       }
//     }

    // Each output vector gets the element-wise product of lhsv and one
    // of the rhsv.
    for (int64_t vnum = 0; vnum < JNum; ++vnum)
      // outv[vnum] += lhsv * rhsv[vnum];
      outv[vnum] += lhsv * BUF_INDEX(rhs, j + vnum, nstride, my_l, kstride, true);
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

  // #pragma clang loop unroll_count(4)
  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
//     // Fill each lhs vector with a value from one of JNum rows of lhs.
//     vF lhsv[INum];
//     for (int64_t vnum = 0; vnum < INum; ++vnum) {
//       F lhs_val = BUF_INDEX(lhs, my_l, kstride, i + vnum, mstride, false);
// #pragma clang loop unroll(full)
//       for (int64_t vidx = 0; vidx < JNum; ++vidx) {
//         lhsv[vnum][vidx] = lhs_val;
//       }
//     }

    // Get JNum values from a row of rhs.
    vF rhsv;
// #pragma clang loop unroll(full)
//     for (int64_t vidx = 0; vidx < JNum; ++vidx) {
//       rhsv[vidx] = BUF_INDEX(rhs, j + vidx, nstride, my_l, kstride, true);
//     }
    rhsv = *reinterpret_cast<const vF *>(
        &BUF_INDEX(rhs, j, nstride, my_l, kstride, true));

    // Each output vector gets the element-wise product of lhsv and one
    // of the rhsv.
    for (int64_t vnum = 0; vnum < INum; ++vnum)
      // outv[vnum] += lhsv[vnum] * rhsv;
      outv[vnum] += rhsv * BUF_INDEX(lhs, my_l, kstride, i + vnum, mstride, false);
  }

  // Add the output vectors to the output matrix.
  for (int64_t vnum = 0; vnum < INum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < JNum; ++vidx) {
      out[(j + vidx) * outstride + (i + vnum)] += outv[vnum][vidx];
    }
  }
}

// // A specialized base case that computes the outer product of
// // subcolumns of A and subrows of B.  Unlike the more general
// // vectorized base case, this version uses fewer memory accesses by
// // storing the outer-product result in vector registers.
// template <typename F>
// __attribute__((always_inline)) void
// matmul_vec_op_halfvec(F *__restrict__ out, const F *__restrict__ lhs,
//                       const F *__restrict__ rhs, int64_t i, int64_t j,
//                       int64_t l, int64_t KNum, int64_t mstride, int64_t nstride,
//                       int64_t kstride, int64_t outstride) noexcept {
//   // Vector type
//   typedef F vF __attribute__((vector_size(sizeof(F) * (mVec / 2))));

//   // Vectors storing output submatrix.
//   vF outv[nVec];

//   // Zero-initialize the output vectors.
// #pragma clang loop unroll(full)
//   for (int64_t vnum = 0; vnum < nVec; ++vnum)
//     outv[vnum] = (vF){0.0};

//   for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
//     // In the following comments, A denotes the rhs, and B denotes the lhs.

//     // Store a subcolumn of lhs into lhsv.
//     // lhsv = A0 A1 A2 A3
//     vF lhsv;
// #pragma clang loop unroll(full)
//     for (int64_t vidx = 0; vidx < mVec / 2; ++vidx) {
//       lhsv[vidx] = BUF_INDEX(lhs, my_l, kstride, i + vidx, mstride, false);
//     }

//     // Store a subrow of rhs into rhsv, replicated twice.
//     // rhsv = B0 B1 B2 B3
//     vF rhsv;
// #pragma clang loop unroll(full)
//     for (int64_t vidx = 0; vidx < nVec; ++vidx) {
//       rhsv[vidx] = BUF_INDEX(rhs, j + vidx, nstride, my_l, kstride, true);
//     }

//     // Perform the multiplications using two vector shuffles --- one
//     // for lhs and one for rhs --- and four vector multiplies among
//     // the inputs and their shuffles variations.
//     // outv[0] = A0B0 A1B1 A2B2 A3B3
//     outv[0] += lhsv * rhsv;
//     // rhsv_p = B1 B0 B3 B2
// #ifdef USE_AVX512
// #else
// #define RHS_PERM 1, 0, 3, 2
// #define LHS_PERM 2, 3, 0, 1
// #endif // USE_AVX512
//     vF rhsv_p = __builtin_shufflevector(rhsv, rhsv, RHS_PERM);
//     // outv[1] = A0B1 A1B0 A2B3 A3B2
//     outv[1] += lhsv * rhsv_p;
//     // lhsv_p = A2 A3 A0 A1
//     vF lhsv_p = __builtin_shufflevector(lhsv, lhsv, LHS_PERM);
//     // outv[2] = A2B0 A3B1 A0B2 A1B3
//     outv[2] += lhsv_p * rhsv;
//     // outv[3] = A2B1 A3B0 A0B3 A1B2
//     outv[3] += lhsv_p * rhsv_p;
//   }

//   // Shuffle the output vectors to support simple vector-add
//   // operations to store the result back into the output matrix.
//   //
//   vF st[2 * nVec];
// #ifdef USE_AVX512
// #else
// #define ST01_PERM 0, 5, 2, 7
// #define ST23_PERM 2, 7, 0, 5
// #define ST4567_PERM 0, 1, 6, 7
// #endif // USE_AVX512
//   // A0B0, A1B0, A2B2, A3B2
//   st[0] = __builtin_shufflevector(outv[0], outv[1], ST01_PERM);
//   // A0B1, A1B1, A2B3, A3B3
//   st[1] = __builtin_shufflevector(outv[1], outv[0], ST01_PERM);
//   // A0B2, A1B2, A2B0, A3B0
//   st[2] = __builtin_shufflevector(outv[2], outv[3], ST23_PERM);
//   // A0B3, A1B3, A2B1, A3B1
//   st[3] = __builtin_shufflevector(outv[3], outv[2], ST23_PERM);

//   // A0B0, A1B0, A2B0, A3B0
//   st[4] = __builtin_shufflevector(st[0], st[2], ST4567_PERM);
//   // A0B1, A1B1, A2B1, A3B1
//   st[5] = __builtin_shufflevector(st[1], st[3], ST4567_PERM);
//   // A0B2, A1B2, A2B2, A3B2
//   st[6] = __builtin_shufflevector(st[2], st[0], ST4567_PERM);
//   // A0B3, A1B3, A2B3, A3B3
//   st[7] = __builtin_shufflevector(st[3], st[1], ST4567_PERM);

//   // Add the output vectors to the output matrix.
// #pragma clang loop unroll(full)
//   for (int64_t vnum = 0; vnum < nVec; ++vnum) {
// #pragma clang loop unroll(full)
//     for (int64_t vidx = 0; vidx < mVec / 2; ++vidx) {
//       out[(j + vnum) * outstride + (i + vidx)] += st[nVec + vnum][vidx];
//     }
//   }
// #undef RHS_PERM
// #undef LHS_PERM
// #undef ST01_PERM
// #undef ST23_PERM
// #undef ST4567_PERM
// }

// template <typename F, int64_t mVec, int64_t nVec, typename vF = F __attribute__((vector_size(sizeof(F) * mVec)))>
// __attribute__((always_inline)) static void
// matmul_vec_vF_muladd(F *__restrict__ out, vF outv[nVec], vF lhsv, vF rhsv,
//                      int64_t outstride);

// __attribute__((always_inline)) static void
// matmul_vec_vF_muladd<double, 8, 8>(double *__restrict__ out, vF outv[8], vF lhsv, vF rhsv,
//                      int64_t outstride);

// A specialized base case that computes the outer product of
// subcolumns of A and subrows of B.  Unlike the more general
// vectorized base case, this version uses fewer memory accesses by
// storing the outer-product result in vector registers.
template <typename F, int64_t mVec, int64_t nVec>
__attribute__((always_inline)) void
matmul_vec_op(F *__restrict__ out, const F *__restrict__ lhs,
              const F *__restrict__ rhs, int64_t i, int64_t j, int64_t l,
              int64_t KNum, int64_t mstride, int64_t nstride, int64_t kstride,
              int64_t outstride) noexcept {
  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F) * mVec)));

  // Vectors storing output submatrix.
  vF outv[nVec];

  // Zero-initialize the output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < nVec; ++vnum)
    outv[vnum] = (vF){0.0};

#pragma clang loop unroll_count(8)
  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
    // Store a subcolumn of lhs into lhsv.
    // lhsv = A0 A1 A2 A3 A4 A5 A6 A7
    vF lhsv;
// #pragma clang loop unroll(full)
//     for (int64_t vidx = 0; vidx < mVec; ++vidx)
//       lhsv[vidx] = BUF_INDEX(lhs, my_l, kstride, i + vidx, mstride, false);
    lhsv = *reinterpret_cast<const vF *>(
        &BUF_INDEX(lhs, my_l, kstride, i, mstride, false));

    // Store a subrow of rhs into rhsv, replicated twice.
    // rhsv = B0 B1 B2 B3 B0 B1 B2 B3
    vF rhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < nVec; ++vidx) {
      rhsv[vidx] = BUF_INDEX(rhs, j + vidx, nstride, my_l, kstride, true);
    }
    for (int64_t copy = 1; copy < mVec / nVec; ++copy) {
      // rhsv = B0 B1 B2 B3 B0 B1 B2 B3
      for (int64_t vidx = 0; vidx < nVec; ++vidx) {
        rhsv[vidx + (copy * nVec)] = rhsv[vidx];
      }
    }

    if (nVec == 4) {
      // // Perform the multiplications using two vector shuffles --- one
      // // for lhs and one for rhs --- and four vector multiplies among
      // // the inputs and their permutations.

      // // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B0 A5B1 A6B2 A7B3
      // outv[0] += lhsv * rhsv;
      // // rhsv_p = B1 B0 B3 B2 B1 B0 B3 B2
      // vF rhsv_p = __builtin_shufflevector(rhsv, rhsv, 1, 0, 3, 2, 5, 4, 7, 6);
      // // outv[1] = A0B1 A1B0 A2B3 A3B2 A4B1 A5B0 A6B3 A7B2
      // outv[1] += lhsv * rhsv_p;
      // // lhsv_p = A2 A3 A0 A1 A6 A7 A4 A5
      // vF lhsv_p = __builtin_shufflevector(lhsv, lhsv, 2, 3, 0, 1, 6, 7, 4, 5);
      // // outv[2] = A2B0 A3B1 A0B2 A1B3 A6B0 A7B1 A4B2 A5B3
      // outv[2] += lhsv_p * rhsv;
      // // outv[3] = A2B1 A3B0 A0B3 A1B2 A6B1 A7B0 A4B3 A5B2
      // outv[3] += lhsv_p * rhsv_p;

    } else { // nVec == 8
      // Perform the multiplications using four vector shuffles and
      // eight vector multiplies among the inputs and their
      // permutations.

      // // Perm 0
      // // // rhsv_p = B5 B4 B7 B6 B1 B0 B3 B2
      // // vF rhsv_p = __builtin_shufflevector(rhsv, rhsv, 5, 4, 7, 6, 1, 0, 3, 2);
      // // // lhsv_p0 = A2 A3 A0 A1 A6 A7 A4 A5
      // // vF lhsv_p0 = __builtin_shufflevector(lhsv, lhsv, 2, 3, 0, 1, 6, 7, 4, 5);
      // // // lhsv_p1 = A4 A5 A6 A7 A0 A1 A2 A3
      // // vF lhsv_p1 = __builtin_shufflevector(lhsv, lhsv, 4, 5, 6, 7, 0, 1, 2, 3);
      // // // lhsv_p2 = A3 A2 A1 A0 A7 A6 A5 A4
      // // vF lhsv_p2 = __builtin_shufflevector(lhsv, lhsv, 3, 2, 1, 0, 7, 6, 5, 4);

      // // // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B4 A5B5 A6B6 A7B7
      // // outv[0] += lhsv * rhsv;
      // // // outv[1] = A0B5 A1B4 A2B7 A3B6 A4B1 A5B0 A6B3 A7B2
      // // outv[1] += lhsv * rhsv_p;
      // // // outv[2] = A2B0 A3B1 A0B2 A1B3 A6B4 A7B5 A4B6 A5B7
      // // outv[2] += lhsv_p0 * rhsv;
      // // // outv[3] = A2B5 A3B4 A0B7 A1B6 A6B1 A7B0 A4B3 A5B2
      // // outv[3] += lhsv_p0 * rhsv_p;
      // // // outv[4] = A4B0 A5B1 A6B2 A7B3 A0B4 A1B5 A2B6 A3B7
      // // outv[4] += lhsv_p1 * rhsv;
      // // // outv[5] = A4B5 A5B4 A6B7 A7B6 A0B1 A1B0 A2B3 A3B2
      // // outv[5] += lhsv_p1 * rhsv_p;
      // // // outv[6] = A3B0 A2B1 A1B2 A0B3 A7B4 A6B5 A5B6 A4B7
      // // outv[6] += lhsv_p2 * rhsv;
      // // // outv[7] = A3B5 A2B4 A1B7 A0B6 A7B1 A6B0 A5B3 A4B2
      // // outv[7] += lhsv_p2 * rhsv_p;

      // // Perm 1
      // // // rhsv_p = B7 B6 B5 B4 B3 B2 B1 B0
      // // vF rhsv_p = __builtin_shufflevector(rhsv, rhsv, 7, 6, 5, 4, 3, 2, 1, 0);
      // // // lhsv_p0 = A2 A3 A0 A1 A6 A7 A4 A5
      // // vF lhsv_p0 = __builtin_shufflevector(lhsv, lhsv, 2, 3, 0, 1, 6, 7, 4, 5);
      // // // lhsv_p1 = A4 A5 A6 A7 A0 A1 A2 A3
      // // vF lhsv_p1 = __builtin_shufflevector(lhsv, lhsv, 4, 5, 6, 7, 0, 1, 2, 3);
      // // // lhsv_p2 = A1 A0 A3 A2 A5 A4 A7 A6
      // // vF lhsv_p2 = __builtin_shufflevector(lhsv, lhsv, 1, 0, 3, 2, 5, 4, 7, 6);

      // // // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B4 A5B5 A6B6 A7B7
      // // outv[0] += lhsv * rhsv;
      // // // outv[1] = A0B7 A1B6 A2B5 A3B4 A4B3 A5B2 A6B1 A7B0
      // // outv[1] += lhsv * rhsv_p;
      // // // outv[2] = A2B0 A3B1 A0B2 A1B3 A6B4 A7B5 A4B6 A5B7
      // // outv[2] += lhsv_p0 * rhsv;
      // // // outv[3] = A2B7 A3B6 A0B5 A1B4 A6B3 A7B2 A4B1 A5B0
      // // outv[3] += lhsv_p0 * rhsv_p;
      // // // outv[4] = A4B0 A5B1 A6B2 A7B3 A0B4 A1B5 A2B6 A3B7
      // // outv[4] += lhsv_p1 * rhsv;
      // // // outv[5] = A4B7 A5B6 A6B5 A7B4 A0B3 A1B2 A2B1 A3B0
      // // outv[5] += lhsv_p1 * rhsv_p;
      // // // outv[6] = A1B0 A0B1 A3B2 A2B3 A5B4 A4B5 A7B6 A6B7
      // // outv[6] += lhsv_p2 * rhsv;
      // // // outv[7] = A1B7 A0B6 A3B5 A2B4 A5B3 A4B2 A7B1 A6B0
      // // outv[7] += lhsv_p2 * rhsv_p;

      // // Perm 2
      // // rhsv_p0 = B2 B3 B0 B1 B6 B7 B4 B5
      // vF rhsv_p0 = __builtin_shufflevector(rhsv, rhsv, 2, 3, 0, 1, 6, 7, 4, 5);
      // // rhsv_p1 = B4 B5 B6 B7 B0 B1 B2 B3
      // vF rhsv_p1 = __builtin_shufflevector(rhsv, rhsv, 4, 5, 6, 7, 0, 1, 2, 3);
      // // rhsv_p2 = B6 B7 B4 B5 B2 B3 B0 B1
      // vF rhsv_p2 = __builtin_shufflevector(rhsv, rhsv, 6, 7, 4, 5, 2, 3, 0, 1);
      // // vF rhsv_p2 = __builtin_shufflevector(rhsv_p0, rhsv_p0, 4, 5, 6, 7, 0, 1, 2, 3);
      // // lhsv_p = A1 A0 A3 A2 A5 A4 A7 A6
      // vF lhsv_p = __builtin_shufflevector(lhsv, lhsv, 1, 0, 3, 2, 5, 4, 7, 6);

      // // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B4 A5B5 A6B6 A7B7
      // outv[0] += lhsv * rhsv;
      // // outv[1] = A0B2 A1B3 A2B0 A3B1 A4B6 A5B7 A6B4 A7B5
      // outv[1] += lhsv * rhsv_p0;
      // // outv[2] = A0B4 A1B5 A2B6 A3B7 A4B0 A5B1 A6B2 A7B3
      // outv[2] += lhsv * rhsv_p1;
      // // outv[3] = A0B6 A1B7 A2B4 A3B5 A4B2 A5B3 A6B0 A7B1
      // outv[3] += lhsv * rhsv_p2;
      // // outv[4] = A1B0 A0B1 A3B2 A2B3 A5B4 A4B5 A7B6 A6B7
      // outv[4] += lhsv_p * rhsv;
      // // outv[5] = A1B2 A0B3 A3B0 A2B1 A5B6 A4B7 A7B4 A6B5
      // outv[5] += lhsv_p * rhsv_p0;
      // // outv[6] = A1B4 A0B5 A3B6 A2B7 A5B0 A4B1 A7B2 A6B3
      // outv[6] += lhsv_p * rhsv_p1;
      // // outv[7] = A1B6 A0B7 A3B4 A2B5 A5B2 A4B3 A7B0 A6B1
      // outv[7] += lhsv_p * rhsv_p2;

      
      // rhsv_p0 = B2 B3 B0 B1 B6 B7 B4 B5
      vF rhsv_p0 = __builtin_shufflevector(rhsv, rhsv, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
      // rhsv_p1 = B4 B5 B6 B7 B0 B1 B2 B3
      vF rhsv_p1 = __builtin_shufflevector(rhsv, rhsv, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
      // rhsv_p2 = B6 B7 B4 B5 B2 B3 B0 B1
      vF rhsv_p2 = __builtin_shufflevector(rhsv, rhsv, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9);
      // vF rhsv_p2 = __builtin_shufflevector(rhsv_p0, rhsv_p0, 4, 5, 6, 7, 0, 1, 2, 3);
      // lhsv_p = A1 A0 A3 A2 A5 A4 A7 A6
      vF lhsv_p = __builtin_shufflevector(lhsv, lhsv, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

      // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B4 A5B5 A6B6 A7B7 A8B0 A9B1 ...
      outv[0] += lhsv * rhsv;
      // outv[1] = A0B2 A1B3 A2B0 A3B1 A4B6 A5B7 A6B4 A7B5 A8B2 A9B3 ...
      outv[1] += lhsv * rhsv_p0;
      // outv[2] = A0B4 A1B5 A2B6 A3B7 A4B0 A5B1 A6B2 A7B3 A8B4 A9B5 ...
      outv[2] += lhsv * rhsv_p1;
      // outv[3] = A0B6 A1B7 A2B4 A3B5 A4B2 A5B3 A6B0 A7B1 A8B6 A9B7 ...
      outv[3] += lhsv * rhsv_p2;
      // outv[4] = A1B0 A0B1 A3B2 A2B3 A5B4 A4B5 A7B6 A6B7 A9B0 A8B1 ...
      outv[4] += lhsv_p * rhsv;
      // outv[5] = A1B2 A0B3 A3B0 A2B1 A5B6 A4B7 A7B4 A6B5 A9B2 A8B3 ...
      outv[5] += lhsv_p * rhsv_p0;
      // outv[6] = A1B4 A0B5 A3B6 A2B7 A5B0 A4B1 A7B2 A6B3 A9B4 A8B5 ...
      outv[6] += lhsv_p * rhsv_p1;
      // outv[7] = A1B6 A0B7 A3B4 A2B5 A5B2 A4B3 A7B0 A6B1 A9B6 A8B7 ...
      outv[7] += lhsv_p * rhsv_p2;
    }
  }

  // Shuffle the output vectors to support simple vector-add
  // operations to store the result back into the output matrix.
  //
  // Below, A denotes the rhs, and B denotes the lhs.
  if (nVec == 4) {
//     vF st[2 * nVec];
//     // A0B0, A1B0, A2B2, A3B2, A4B0, A5B0, A6B2, A7B2
//     st[0] = __builtin_shufflevector(outv[0], outv[1], 0, 9, 2, 11, 4, 13, 6, 15);
//     // A0B1, A1B1, A2B3, A3B3, A4B1, A5B1, A6B3, A7B3
//     st[1] = __builtin_shufflevector(outv[1], outv[0], 0, 9, 2, 11, 4, 13, 6, 15);
//     // A0B2, A1B2, A2B0, A3B0, A4B2, A5B2, A6B0, A7B0
//     st[2] = __builtin_shufflevector(outv[2], outv[3], 2, 11, 0, 9, 6, 15, 4, 13);
//     // A0B3, A1B3, A2B1, A3B1, A4B3, A5B3, A6B1, A7B1
//     st[3] = __builtin_shufflevector(outv[3], outv[2], 2, 11, 0, 9, 6, 15, 4, 13);

//     // A0B0, A1B0, A2B0, A3B0, A4B0, A5B0, A6B0, A7B0
//     st[4] = __builtin_shufflevector(st[0], st[2], 0, 1, 10, 11, 4, 5, 14, 15);
//     // A0B1, A1B1, A2B1, A3B1, A4B1, A5B1, A6B1, A7B1
//     st[5] = __builtin_shufflevector(st[1], st[3], 0, 1, 10, 11, 4, 5, 14, 15);
//     // A0B2, A1B2, A2B2, A3B2, A4B2, A5B2, A6B2, A7B2
//     st[6] = __builtin_shufflevector(st[2], st[0], 0, 1, 10, 11, 4, 5, 14, 15);
//     // A0B3, A1B3, A2B3, A3B3, A4B3, A5B3, A6B3, A7B3
//     st[7] = __builtin_shufflevector(st[3], st[1], 0, 1, 10, 11, 4, 5, 14, 15);

//     // Add the output vectors to the output matrix.
// #pragma clang loop unroll(full)
//     for (int64_t vnum = 0; vnum < nVec; ++vnum) {
// #pragma clang loop unroll(full)
//       for (int64_t vidx = 0; vidx < mVec; ++vidx) {
// 	out[(j + vnum) * outstride + (i + vidx)] += st[nVec + vnum][vidx];
//       }
//     }
  } else { // nVec == 8
    vF st[3 * nVec];

    // // Perm 2
    // // A0B0 A1B0 A2B2 A3B2 A4B4 A5B4 A6B6 A7B6
    // st[0] = __builtin_shufflevector(outv[0], outv[4], 0, 8, 2, 10, 4, 12, 6, 14);
    // // A0B1 A1B1 A2B3 A3B3 A4B5 A5B5 A6B7 A7B7
    // st[1] = __builtin_shufflevector(outv[4], outv[0], 1, 9, 3, 11, 5, 13, 7, 15);
    // // A0B2 A1B2 A2B0 A3B0 A4B6 A5B6 A6B4 A7B4
    // st[2] = __builtin_shufflevector(outv[1], outv[5], 0, 8, 2, 10, 4, 12, 6, 14);
    // // A0B3 A1B3 A2B1 A3B1 A4B7 A5B7 A6B5 A7B5
    // st[3] = __builtin_shufflevector(outv[5], outv[1], 1, 9, 3, 11, 5, 13, 7, 15);
    // // A0B4 A1B4 A2B6 A3B6 A4B0 A5B0 A6B2 A7B2
    // st[4] = __builtin_shufflevector(outv[2], outv[6], 0, 8, 2, 10, 4, 12, 6, 14);
    // // A0B5 A1B5 A2B7 A3B7 A4B1 A5B1 A6B3 A7B3
    // st[5] = __builtin_shufflevector(outv[6], outv[2], 1, 9, 3, 11, 5, 13, 7, 15);
    // // A0B6 A1B6 A2B4 A3B4 A4B2 A5B2 A6B0 A7B0
    // st[6] = __builtin_shufflevector(outv[3], outv[7], 0, 8, 2, 10, 4, 12, 6, 14);
    // // A0B7 A1B7 A2B5 A3B5 A4B3 A5B3 A6B1 A7B1
    // st[7] = __builtin_shufflevector(outv[7], outv[3], 1, 9, 3, 11, 5, 13, 7, 15);

    // // A0B0 A1B0 A2B0 A3B0 A4B4 A5B4 A6B4 A7B4
    // st[8]  = __builtin_shufflevector(st[0], st[2], 0, 1, 10, 11, 4, 5, 14, 15);
    // // A0B1 A1B1 A2B1 A3B1 A4B5 A5B5 A6B5 A7B5
    // st[9]  = __builtin_shufflevector(st[1], st[3], 0, 1, 10, 11, 4, 5, 14, 15);
    // // A0B2 A1B2 A2B2 A3B2 A4B6 A5B6 A6B6 A7B6
    // st[10] = __builtin_shufflevector(st[2], st[0], 0, 1, 10, 11, 4, 5, 14, 15);
    // // A0B3 A1B3 A2B3 A3B3 A4B7 A5B7 A6B7 A7B7
    // st[11] = __builtin_shufflevector(st[3], st[1], 0, 1, 10, 11, 4, 5, 14, 15);
    // // A0B4 A1B4 A2B4 A3B4 A4B0 A5B0 A6B0 A7B0
    // st[12] = __builtin_shufflevector(st[4], st[6], 0, 1, 10, 11, 4, 5, 14, 15);
    // // A0B5 A1B5 A2B5 A3B5 A4B1 A5B1 A6B1 A7B1
    // st[13] = __builtin_shufflevector(st[5], st[7], 0, 1, 10, 11, 4, 5, 14, 15);
    // // A0B6 A1B6 A2B6 A3B6 A4B2 A5B2 A6B2 A7B2
    // st[14] = __builtin_shufflevector(st[6], st[4], 0, 1, 10, 11, 4, 5, 14, 15);
    // // A0B7 A1B7 A2B7 A3B7 A4B3 A5B3 A6B3 A7B3
    // st[15] = __builtin_shufflevector(st[7], st[5], 0, 1, 10, 11, 4, 5, 14, 15);

    // // A0B0 A1B0 A2B0 A3B0 A4B0 A5B0 A6B0 A7B0
    // st[16] = __builtin_shufflevector(st[8], st[12], 0, 1, 2, 3, 12, 13, 14, 15);
    // // A0B1 A1B1 A2B1 A3B1 A4B1 A5B1 A6B1 A7B1
    // st[17] = __builtin_shufflevector(st[9], st[13], 0, 1, 2, 3, 12, 13, 14, 15);
    // // A0B2 A1B2 A2B2 A3B2 A4B2 A5B2 A6B2 A7B2
    // st[18] = __builtin_shufflevector(st[10], st[14], 0, 1, 2, 3, 12, 13, 14, 15);
    // // A0B3 A1B3 A2B3 A3B3 A4B3 A5B3 A6B3 A7B3
    // st[19] = __builtin_shufflevector(st[11], st[15], 0, 1, 2, 3, 12, 13, 14, 15);
    // // A0B4 A1B4 A2B4 A3B4 A4B4 A5B4 A6B4 A7B4
    // st[20] = __builtin_shufflevector(st[12], st[8], 0, 1, 2, 3, 12, 13, 14, 15);
    // // A0B5 A1B5 A2B5 A3B5 A4B5 A5B5 A6B5 A7B5
    // st[21] = __builtin_shufflevector(st[13], st[9], 0, 1, 2, 3, 12, 13, 14, 15);
    // // A0B6 A1B6 A2B6 A3B6 A4B6 A5B6 A6B6 A7B6
    // st[22] = __builtin_shufflevector(st[14], st[10], 0, 1, 2, 3, 12, 13, 14, 15);
    // // A0B7 A1B7 A2B7 A3B7 A4B7 A5B7 A6B7 A7B7
    // st[23] = __builtin_shufflevector(st[15], st[11], 0, 1, 2, 3, 12, 13, 14, 15);

    // // Perm 0
    // // // A0B0 A1B1 A2B0 A3B1 A4B4 A5B5 A6B4 A7B5
    // // st[0] = __builtin_shufflevector(outv[0], outv[2], 0, 1, 8, 9, 4, 5, 12, 13);
    // // // A0B2 A1B3 A2B2 A3B3 A4B6 A5B7 A6B6 A7B7
    // // st[1] = __builtin_shufflevector(outv[2], outv[0], 2, 3, 10, 11, 6, 7, 14, 15);
    // // // A0B1 A1B0 A2B1 A3B0 A4B5 A5B4 A6B5 A7B4
    // // st[2] = __builtin_shufflevector(outv[5], outv[6], 4, 5, 9, 8, 0, 1, 13, 12);
    // // // A0B3 A1B2 A2B3 A3B2 A4B7 A5B6 A6B7 A7B6
    // // st[3] = __builtin_shufflevector(outv[6], outv[5], 3, 2, 14, 15, 7, 6, 10, 11);
    // // // A0B4 A1B5 A2B4 A3B5 A4B0 A5B1 A6B0 A7B1
    // // st[4] = __builtin_shufflevector(outv[4], outv[7], 4, 5, 9, 8, 0, 1, 13, 12);
    // // // A0B6 A1B7 A2B6 A3B7 A4B2 A5B3 A6B2 A7B3
    // // st[5] = __builtin_shufflevector(outv[7], outv[4], 3, 2, 14, 15, 7, 6, 10, 11);
    // // // A0B5 A1B4 A2B5 A3B4 A4B1 A5B0 A6B1 A7B0
    // // st[6] = __builtin_shufflevector(outv[1], outv[3], 0, 1, 8, 9, 4, 5, 12, 13);
    // // // A0B7 A1B6 A2B7 A3B6 A4B3 A5B2 A6B3 A7B2
    // // st[7] = __builtin_shufflevector(outv[3], outv[1], 2, 3, 10, 11, 6, 7, 14, 15);

    // // // A0B0 A1B1 A2B0 A3B1 A4B0 A5B1 A6B0 A7B1
    // // st[8]  = __builtin_shufflevector(st[0], st[4], 0, 1, 2, 3, 12, 13, 14, 15);
    // // // A0B4 A1B5 A2B4 A3B5 A4B4 A5B5 A6B6 A7B7
    // // st[9]  = __builtin_shufflevector(st[4], st[0], 0, 1, 2, 3, 12, 13, 14, 15);
    // // // A0B2 A1B3 A2B2 A3B3 A4B2 A5B3 A6B2 A7B3
    // // st[10] = __builtin_shufflevector(st[1], st[5], 0, 1, 2, 3, 12, 13, 14, 15);
    // // // A0B6 A1B7 A2B6 A3B7 A4B6 A5B7 A6B6 A7B7
    // // st[11] = __builtin_shufflevector(st[5], st[1], 0, 1, 2, 3, 12, 13, 14, 15);
    // // // A0B1 A1B0 A2B1 A3B0 A4B1 A5B0 A6B1 A7B0
    // // st[12] = __builtin_shufflevector(st[2], st[6], 0, 1, 2, 3, 12, 13, 14, 15);
    // // // A0B5 A1B4 A2B5 A3B4 A4B5 A5B4 A6B5 A7B4
    // // st[13] = __builtin_shufflevector(st[6], st[2], 0, 1, 2, 3, 12, 13, 14, 15);
    // // // A0B3 A1B2 A2B3 A3B2 A4B3 A5B2 A6B3 A7B2
    // // st[14] = __builtin_shufflevector(st[3], st[7], 0, 1, 2, 3, 12, 13, 14, 15);
    // // // A0B7 A1B6 A2B7 A3B6 A4B7 A5B6 A6B7 A7B6
    // // st[15] = __builtin_shufflevector(st[7], st[3], 0, 1, 2, 3, 12, 13, 14, 15);

    // // // A0B0 A1B0 A2B0 A3B0 A4B0 A5B0 A6B0 A7B0
    // // st[16] = __builtin_shufflevector(st[8], st[12], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B1 A1B1 A2B1 A3B1 A4B1 A5B1 A6B1 A7B1
    // // st[17] = __builtin_shufflevector(st[12], st[8], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B2 A1B2 A2B2 A3B2 A4B2 A5B2 A6B2 A7B2
    // // st[18] = __builtin_shufflevector(st[10], st[14], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B3 A1B3 A2B3 A3B3 A4B3 A5B3 A6B3 A7B3
    // // st[19] = __builtin_shufflevector(st[14], st[10], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B4 A1B4 A2B4 A3B4 A4B4 A5B4 A6B4 A7B4
    // // st[20] = __builtin_shufflevector(st[9], st[13], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B5 A1B5 A2B5 A3B5 A4B5 A5B5 A6B5 A7B5
    // // st[21] = __builtin_shufflevector(st[13], st[9], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B6 A1B6 A2B6 A3B6 A4B6 A5B6 A6B6 A7B6
    // // st[22] = __builtin_shufflevector(st[11], st[15], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B7 A1B7 A2B7 A3B7 A4B7 A5B7 A6B7 A7B7
    // // st[23] = __builtin_shufflevector(st[15], st[11], 0, 9, 2, 11, 4, 13, 6, 15);

    // // Perm 1
    // // // A0B0 A1B1 A2B2 A3B3 A4B0 A5B1 A6B2 A7B3
    // // st[0] = __builtin_shufflevector(outv[0], outv[4], 0, 1, 2, 3, 8, 9, 10, 11);
    // // // A0B4 A1B5 A2B6 A3B7 A4B4 A5B5 A6B6 A7B7
    // // st[1] = __builtin_shufflevector(outv[4], outv[0], 4, 5, 6, 7, 12, 13, 14, 15);
    // // // A0B7 A1B6 A2B5 A3B4 A4B7 A5B6 A6B5 A7B4
    // // st[2] = __builtin_shufflevector(outv[1], outv[5], 0, 1, 2, 3, 8, 9, 10, 11);
    // // // A0B3 A1B2 A2B1 A3B0 A4B3 A5B2 A6B1 A7B0
    // // st[3] = __builtin_shufflevector(outv[5], outv[1], 4, 5, 6, 7, 12, 13, 14, 15);
    // // // A0B2 A1B3 A2B0 A3B1 A4B2 A5B3 A6B0 A7B1
    // // st[4] = __builtin_shufflevector(outv[2], outv[7], 2, 3, 0, 1, 13, 12, 15, 14);
    // // // A0B6 A1B7 A2B4 A3B5 A4B6 A5B7 A6B4 A7B5
    // // st[5] = __builtin_shufflevector(outv[7], outv[2], 1, 0, 3, 2, 14, 15, 12, 13);
    // // // A0B5 A1B4 A2B7 A3B6 A4B5 A5B4 A6B7 A7B6
    // // st[6] = __builtin_shufflevector(outv[3], outv[6], 2, 3, 0, 1, 13, 12, 15, 14);
    // // // A0B1 A1B0 A2B3 A3B2 A4B1 A5B0 A6B3 A7B2
    // // st[7] = __builtin_shufflevector(outv[6], outv[3], 1, 0, 3, 2, 14, 15, 12, 13);

    // // // A0B0 A1B1 A2B0 A3B1 A4B0 A5B1 A6B0 A7B1
    // // st[8]  = __builtin_shufflevector(st[0], st[4], 0, 1, 10, 11, 4, 5, 14, 15);
    // // // A0B2 A1B3 A2B2 A3B3 A4B2 A5B3 A6B2 A7B3
    // // st[9]  = __builtin_shufflevector(st[4], st[0], 0, 1, 10, 11, 4, 5, 14, 15);
    // // // A0B1 A1B0 A2B1 A3B0 A4B1 A5B0 A6B1 A7B0
    // // st[10] = __builtin_shufflevector(st[7], st[3], 0, 1, 10, 11, 4, 5, 14, 15);
    // // // A0B3 A1B2 A2B3 A3B2 A4B3 A5B2 A6B3 A7B2
    // // st[11] = __builtin_shufflevector(st[3], st[7], 0, 1, 10, 11, 4, 5, 14, 15);
    // // // A0B4 A1B5 A2B4 A3B5 A4B4 A5B5 A6B4 A7B5
    // // st[12] = __builtin_shufflevector(st[1], st[5], 0, 1, 10, 11, 4, 5, 14, 15);
    // // // A0B6 A1B7 A2B6 A3B7 A4B6 A5B7 A6B6 A7B7
    // // st[13] = __builtin_shufflevector(st[5], st[1], 0, 1, 10, 11, 4, 5, 14, 15);
    // // // A0B5 A1B4 A2B5 A3B4 A4B5 A5B4 A6B5 A7B4
    // // st[14] = __builtin_shufflevector(st[6], st[2], 0, 1, 10, 11, 4, 5, 14, 15);
    // // // A0B7 A1B6 A2B7 A3B6 A4B7 A5B6 A6B7 A7B6
    // // st[15] = __builtin_shufflevector(st[2], st[6], 0, 1, 10, 11, 4, 5, 14, 15);

    // // // A0B0 A1B0 A2B0 A3B0 A4B0 A5B0 A6B0 A7B0
    // // st[16] = __builtin_shufflevector(st[8], st[10], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B1 A1B1 A2B1 A3B1 A4B1 A5B1 A6B1 A7B1
    // // st[17] = __builtin_shufflevector(st[10], st[8], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B2 A1B2 A2B2 A3B2 A4B2 A5B2 A6B2 A7B2
    // // st[18] = __builtin_shufflevector(st[9], st[11], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B3 A1B3 A2B3 A3B3 A4B3 A5B3 A6B3 A7B3
    // // st[19] = __builtin_shufflevector(st[11], st[9], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B4 A1B4 A2B4 A3B4 A4B4 A5B4 A6B4 A7B4
    // // st[20] = __builtin_shufflevector(st[12], st[14], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B5 A1B5 A2B5 A3B5 A4B5 A5B5 A6B5 A7B5
    // // st[21] = __builtin_shufflevector(st[14], st[12], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B6 A1B6 A2B6 A3B6 A4B6 A5B6 A6B6 A7B6
    // // st[22] = __builtin_shufflevector(st[13], st[15], 0, 9, 2, 11, 4, 13, 6, 15);
    // // // A0B7 A1B7 A2B7 A3B7 A4B7 A5B7 A6B7 A7B7
    // // st[23] = __builtin_shufflevector(st[15], st[13], 0, 9, 2, 11, 4, 13, 6, 15);

    // Perm 2, mVec == 16
    // A0B0 A1B0 A2B2 A3B2 A4B4 A5B4 A6B6 A7B6 A8B0 A9B0 ...
    st[0] = __builtin_shufflevector(outv[0], outv[4], 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
    // A0B1 A1B1 A2B3 A3B3 A4B5 A5B5 A6B7 A7B7 A8B1 A9B1 ...
    st[1] = __builtin_shufflevector(outv[4], outv[0], 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);
    // A0B2 A1B2 A2B0 A3B0 A4B6 A5B6 A6B4 A7B4
    st[2] = __builtin_shufflevector(outv[1], outv[5], 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
    // A0B3 A1B3 A2B1 A3B1 A4B7 A5B7 A6B5 A7B5
    st[3] = __builtin_shufflevector(outv[5], outv[1], 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);
    // A0B4 A1B4 A2B6 A3B6 A4B0 A5B0 A6B2 A7B2
    st[4] = __builtin_shufflevector(outv[2], outv[6], 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
    // A0B5 A1B5 A2B7 A3B7 A4B1 A5B1 A6B3 A7B3
    st[5] = __builtin_shufflevector(outv[6], outv[2], 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);
    // A0B6 A1B6 A2B4 A3B4 A4B2 A5B2 A6B0 A7B0
    st[6] = __builtin_shufflevector(outv[3], outv[7], 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
    // A0B7 A1B7 A2B5 A3B5 A4B3 A5B3 A6B1 A7B1
    st[7] = __builtin_shufflevector(outv[7], outv[3], 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);

    // A0B0 A1B0 A2B0 A3B0 A4B4 A5B4 A6B4 A7B4
    st[8]  = __builtin_shufflevector(st[0], st[2], 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31);
    // A0B1 A1B1 A2B1 A3B1 A4B5 A5B5 A6B5 A7B5
    st[9]  = __builtin_shufflevector(st[1], st[3], 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31);
    // A0B2 A1B2 A2B2 A3B2 A4B6 A5B6 A6B6 A7B6
    st[10] = __builtin_shufflevector(st[2], st[0], 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31);
    // A0B3 A1B3 A2B3 A3B3 A4B7 A5B7 A6B7 A7B7
    st[11] = __builtin_shufflevector(st[3], st[1], 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31);
    // A0B4 A1B4 A2B4 A3B4 A4B0 A5B0 A6B0 A7B0
    st[12] = __builtin_shufflevector(st[4], st[6], 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31);
    // A0B5 A1B5 A2B5 A3B5 A4B1 A5B1 A6B1 A7B1
    st[13] = __builtin_shufflevector(st[5], st[7], 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31);
    // A0B6 A1B6 A2B6 A3B6 A4B2 A5B2 A6B2 A7B2
    st[14] = __builtin_shufflevector(st[6], st[4], 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31);
    // A0B7 A1B7 A2B7 A3B7 A4B3 A5B3 A6B3 A7B3
    st[15] = __builtin_shufflevector(st[7], st[5], 0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31);

    // A0B0 A1B0 A2B0 A3B0 A4B0 A5B0 A6B0 A7B0
    st[16] = __builtin_shufflevector(st[8], st[12], 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31);
    // A0B1 A1B1 A2B1 A3B1 A4B1 A5B1 A6B1 A7B1
    st[17] = __builtin_shufflevector(st[9], st[13], 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31);
    // A0B2 A1B2 A2B2 A3B2 A4B2 A5B2 A6B2 A7B2
    st[18] = __builtin_shufflevector(st[10], st[14], 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31);
    // A0B3 A1B3 A2B3 A3B3 A4B3 A5B3 A6B3 A7B3
    st[19] = __builtin_shufflevector(st[11], st[15], 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31);
    // A0B4 A1B4 A2B4 A3B4 A4B4 A5B4 A6B4 A7B4
    st[20] = __builtin_shufflevector(st[12], st[8], 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31);
    // A0B5 A1B5 A2B5 A3B5 A4B5 A5B5 A6B5 A7B5
    st[21] = __builtin_shufflevector(st[13], st[9], 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31);
    // A0B6 A1B6 A2B6 A3B6 A4B6 A5B6 A6B6 A7B6
    st[22] = __builtin_shufflevector(st[14], st[10], 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31);
    // A0B7 A1B7 A2B7 A3B7 A4B7 A5B7 A6B7 A7B7
    st[23] = __builtin_shufflevector(st[15], st[11], 0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31);

#pragma clang loop unroll(full)
    for (int64_t vnum = 0; vnum < nVec; ++vnum) {
#pragma clang loop unroll(full)
      for (int64_t vidx = 0; vidx < mVec; ++vidx) {
        out[(j + vnum) * outstride + (i + vidx)] += st[(2 * nVec) + vnum][vidx];
      }
    }
  }
}

// A specialized base case that computes the outer product of
// subcolumns of A and subrows of B.  Unlike the more general
// vectorized base case, this version uses fewer memory accesses by
// storing the outer-product result in vector registers.
template <typename F, int64_t NumMVec, int64_t nVec>
__attribute__((always_inline)) void
matmul_vec_op_ext(F *__restrict__ out, const F *__restrict__ lhs,
		  const F *__restrict__ rhs, int64_t i, int64_t j, int64_t l,
		  int64_t KNum, int64_t mstride, int64_t nstride, int64_t kstride,
		  int64_t outstride) noexcept {
  // Vector type
  const int64_t vWidth = mVec;
  typedef F vF __attribute__((vector_size(sizeof(F) * vWidth)));

  // Vectors storing output submatrix.
  vF outv[NumMVec][nVec];

  // Zero-initialize the output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < nVec; ++vnum) {
    // outv[vnum] = (vF){0.0};
#pragma clang loop unroll(full)
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      // __builtin_prefetch(&out[(j + vnum) * outstride + (i + (mvnum * vWidth))],
      //                    1, 3);
      outv[mvnum][vnum] = (vF){0.0};
    }
  }

  const int64_t kRem = NumMVec * nVec;
  for (int64_t ll = 0; ll < KNum / kBlk; ++ll) {
#pragma clang loop unroll_count(4)
    for (int64_t my_l = l + (ll * kBlk); my_l < l + ((ll + 1) * kBlk) - kRem;
         ++my_l) {
      // __builtin_prefetch(&BUF_INDEX(rhs, j, nstride, my_l + 4, kstride,
      // true));

      // Store a subcolumn of lhs into each lhsv.
      // lhsv = A0 A1 A2 A3 A4 A5 A6 A7
      vF lhsv[NumMVec];
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l, kstride, i + (mvnum * vWidth), mstride, false));
      }

      // Store a subrow of rhs into rhsv, replicated twice.
      // rhsv = B0 B1 B2 B3 B0 B1 B2 B3
      vF rhsv = *reinterpret_cast<const vF *>(
          &BUF_INDEX(rhs, j, nstride, my_l, kstride, true));

      // Perform the multiplications using four vector shuffles and
      // eight vector multiplies among the inputs and their
      // permutations.

      // lhsv_p = A1 A0 A3 A2 A5 A4 A7 A6
      vF lhsv_p[NumMVec];
      // rhsv_p0 = B2 B3 B0 B1 B6 B7 B4 B5
      vF rhsv_p0 = __builtin_shufflevector(rhsv, rhsv, 2, 3, 0, 1, 6, 7, 4, 5);
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][0] += lhsv[mvnum] * rhsv;
        lhsv_p[mvnum] = __builtin_shufflevector(lhsv[mvnum], lhsv[mvnum], 1, 0,
                                                3, 2, 5, 4, 7, 6);
      }

      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][4] += lhsv_p[mvnum] * rhsv;
      }

      // rhsv_p1 = B4 B5 B6 B7 B0 B1 B2 B3
      vF rhsv_p1 = __builtin_shufflevector(rhsv, rhsv, 4, 5, 6, 7, 0, 1, 2, 3);
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][1] += lhsv[mvnum] * rhsv_p0;
	outv[mvnum][5] += lhsv_p[mvnum] * rhsv_p0;
      }
    
      // rhsv_p2 = B6 B7 B4 B5 B2 B3 B0 B1
      vF rhsv_p2 = __builtin_shufflevector(rhsv_p0, rhsv_p0, 4, 5, 6, 7, 0, 1, 2, 3);
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][2] += lhsv[mvnum] * rhsv_p1;
	outv[mvnum][6] += lhsv_p[mvnum] * rhsv_p1;
      }

      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][3] += lhsv[mvnum] * rhsv_p2;
	outv[mvnum][7] += lhsv_p[mvnum] * rhsv_p2;
      }

      // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B4 A5B5 A6B6 A7B7
      // outv[1] = A0B2 A1B3 A2B0 A3B1 A4B6 A5B7 A6B4 A7B5
      // outv[2] = A0B4 A1B5 A2B6 A3B7 A4B0 A5B1 A6B2 A7B3
      // outv[3] = A0B6 A1B7 A2B4 A3B5 A4B2 A5B3 A6B0 A7B1
      // outv[4] = A1B0 A0B1 A3B2 A2B3 A5B4 A4B5 A7B6 A6B7
      // outv[5] = A1B2 A0B3 A3B0 A2B1 A5B6 A4B7 A7B4 A6B5
      // outv[6] = A1B4 A0B5 A3B6 A2B7 A5B0 A4B1 A7B2 A6B3
      // outv[7] = A1B6 A0B7 A3B4 A2B5 A5B2 A4B3 A7B0 A6B1
    }

    int64_t out_prefetch_num = 0;
#pragma clang loop unroll_count(4)
    for (int64_t my_l = l + ((ll + 1) * kBlk) - kRem;
         my_l < l + ((ll + 1) * kBlk); ++my_l) {
      // __builtin_prefetch(&BUF_INDEX(rhs, j, nstride, my_l + 4, kstride,
      // true));

      // Store a subcolumn of lhs into each lhsv.
      // lhsv = A0 A1 A2 A3 A4 A5 A6 A7
      vF lhsv[NumMVec];
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
        lhsv[mvnum] = *reinterpret_cast<const vF *>(&BUF_INDEX(
            lhs, my_l, kstride, i + (mvnum * vWidth), mstride, false));
      }

      // Store a subrow of rhs into rhsv, replicated twice.
      // rhsv = B0 B1 B2 B3 B0 B1 B2 B3
      vF rhsv = *reinterpret_cast<const vF *>(
          &BUF_INDEX(rhs, j, nstride, my_l, kstride, true));

      __builtin_prefetch(&out[(j + (out_prefetch_num / NumMVec)) * outstride +
                              (i + ((out_prefetch_num % NumMVec) * vWidth))],
                         1, 3);
      out_prefetch_num++;

      // Perform the multiplications using four vector shuffles and
      // eight vector multiplies among the inputs and their
      // permutations.

      // lhsv_p = A1 A0 A3 A2 A5 A4 A7 A6
      vF lhsv_p[NumMVec];
      // rhsv_p0 = B2 B3 B0 B1 B6 B7 B4 B5
      vF rhsv_p0 = __builtin_shufflevector(rhsv, rhsv, 2, 3, 0, 1, 6, 7, 4, 5);
#pragma clang loop unroll(full)
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][0] += lhsv[mvnum] * rhsv;
        lhsv_p[mvnum] = __builtin_shufflevector(lhsv[mvnum], lhsv[mvnum], 1, 0,
                                                3, 2, 5, 4, 7, 6);
      }

      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][4] += lhsv_p[mvnum] * rhsv;
      }

      // rhsv_p1 = B4 B5 B6 B7 B0 B1 B2 B3
      vF rhsv_p1 = __builtin_shufflevector(rhsv, rhsv, 4, 5, 6, 7, 0, 1, 2, 3);
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][1] += lhsv[mvnum] * rhsv_p0;
	outv[mvnum][5] += lhsv_p[mvnum] * rhsv_p0;
      }
    
      // rhsv_p2 = B6 B7 B4 B5 B2 B3 B0 B1
      vF rhsv_p2 = __builtin_shufflevector(rhsv_p0, rhsv_p0, 4, 5, 6, 7, 0, 1, 2, 3);
      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][2] += lhsv[mvnum] * rhsv_p1;
	outv[mvnum][6] += lhsv_p[mvnum] * rhsv_p1;
      }

      for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
	outv[mvnum][3] += lhsv[mvnum] * rhsv_p2;
	outv[mvnum][7] += lhsv_p[mvnum] * rhsv_p2;
      }

      // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B4 A5B5 A6B6 A7B7
      // outv[1] = A0B2 A1B3 A2B0 A3B1 A4B6 A5B7 A6B4 A7B5
      // outv[2] = A0B4 A1B5 A2B6 A3B7 A4B0 A5B1 A6B2 A7B3
      // outv[3] = A0B6 A1B7 A2B4 A3B5 A4B2 A5B3 A6B0 A7B1
      // outv[4] = A1B0 A0B1 A3B2 A2B3 A5B4 A4B5 A7B6 A6B7
      // outv[5] = A1B2 A0B3 A3B0 A2B1 A5B6 A4B7 A7B4 A6B5
      // outv[6] = A1B4 A0B5 A3B6 A2B7 A5B0 A4B1 A7B2 A6B3
      // outv[7] = A1B6 A0B7 A3B4 A2B5 A5B2 A4B3 A7B0 A6B1
    }
  }

  for (int64_t my_l = l + ((KNum / kBlk) * kBlk); my_l < l + KNum; ++my_l) {
    // Store a subcolumn of lhs into each lhsv.
    // lhsv = A0 A1 A2 A3 A4 A5 A6 A7
    vF lhsv[NumMVec];
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      lhsv[mvnum] = *reinterpret_cast<const vF *>(
          &BUF_INDEX(lhs, my_l, kstride, i + (mvnum * vWidth), mstride, false));
    }

    // Store a subrow of rhs into rhsv, replicated twice.
    // rhsv = B0 B1 B2 B3 B0 B1 B2 B3
    vF rhsv = *reinterpret_cast<const vF *>(
        &BUF_INDEX(rhs, j, nstride, my_l, kstride, true));

    // Perform the multiplications using four vector shuffles and
    // eight vector multiplies among the inputs and their
    // permutations.

    // lhsv_p = A1 A0 A3 A2 A5 A4 A7 A6
    vF lhsv_p[NumMVec];
    // rhsv_p0 = B2 B3 B0 B1 B6 B7 B4 B5
    vF rhsv_p0 = __builtin_shufflevector(rhsv, rhsv, 2, 3, 0, 1, 6, 7, 4, 5);
#pragma clang loop unroll(full)
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      outv[mvnum][0] += lhsv[mvnum] * rhsv;
      lhsv_p[mvnum] = __builtin_shufflevector(lhsv[mvnum], lhsv[mvnum], 1, 0, 3,
                                              2, 5, 4, 7, 6);
    }

    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      outv[mvnum][4] += lhsv_p[mvnum] * rhsv;
    }

    // rhsv_p1 = B4 B5 B6 B7 B0 B1 B2 B3
    vF rhsv_p1 = __builtin_shufflevector(rhsv, rhsv, 4, 5, 6, 7, 0, 1, 2, 3);
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      outv[mvnum][1] += lhsv[mvnum] * rhsv_p0;
      outv[mvnum][5] += lhsv_p[mvnum] * rhsv_p0;
    }

    // rhsv_p2 = B6 B7 B4 B5 B2 B3 B0 B1
    vF rhsv_p2 =
        __builtin_shufflevector(rhsv_p0, rhsv_p0, 4, 5, 6, 7, 0, 1, 2, 3);
    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      outv[mvnum][2] += lhsv[mvnum] * rhsv_p1;
      outv[mvnum][6] += lhsv_p[mvnum] * rhsv_p1;
    }

    for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
      outv[mvnum][3] += lhsv[mvnum] * rhsv_p2;
      outv[mvnum][7] += lhsv_p[mvnum] * rhsv_p2;
    }

    // outv[0] = A0B0 A1B1 A2B2 A3B3 A4B4 A5B5 A6B6 A7B7
    // outv[1] = A0B2 A1B3 A2B0 A3B1 A4B6 A5B7 A6B4 A7B5
    // outv[2] = A0B4 A1B5 A2B6 A3B7 A4B0 A5B1 A6B2 A7B3
    // outv[3] = A0B6 A1B7 A2B4 A3B5 A4B2 A5B3 A6B0 A7B1
    // outv[4] = A1B0 A0B1 A3B2 A2B3 A5B4 A4B5 A7B6 A6B7
    // outv[5] = A1B2 A0B3 A3B0 A2B1 A5B6 A4B7 A7B4 A6B5
    // outv[6] = A1B4 A0B5 A3B6 A2B7 A5B0 A4B1 A7B2 A6B3
    // outv[7] = A1B6 A0B7 A3B4 A2B5 A5B2 A4B3 A7B0 A6B1
  }

  // Shuffle the output vectors to support simple vector-add
  // operations to store the result back into the output matrix.
  //
  // Below, A denotes the rhs, and B denotes the lhs.

  vF st[NumMVec][3 * nVec];
  for (int64_t mvnum = 0; mvnum < NumMVec; ++mvnum) {
    // A0B0 A1B0 A2B2 A3B2 A4B4 A5B4 A6B6 A7B6
    st[mvnum][0] = __builtin_shufflevector(outv[mvnum][0], outv[mvnum][4], 0, 8, 2, 10, 4, 12, 6, 14);
    // A0B1 A1B1 A2B3 A3B3 A4B5 A5B5 A6B7 A7B7
    st[mvnum][1] = __builtin_shufflevector(outv[mvnum][4], outv[mvnum][0], 1, 9, 3, 11, 5, 13, 7, 15);
    // A0B2 A1B2 A2B0 A3B0 A4B6 A5B6 A6B4 A7B4
    st[mvnum][2] = __builtin_shufflevector(outv[mvnum][1], outv[mvnum][5], 0, 8, 2, 10, 4, 12, 6, 14);
    // A0B3 A1B3 A2B1 A3B1 A4B7 A5B7 A6B5 A7B5
    st[mvnum][3] = __builtin_shufflevector(outv[mvnum][5], outv[mvnum][1], 1, 9, 3, 11, 5, 13, 7, 15);
    // A0B4 A1B4 A2B6 A3B6 A4B0 A5B0 A6B2 A7B2
    st[mvnum][4] = __builtin_shufflevector(outv[mvnum][2], outv[mvnum][6], 0, 8, 2, 10, 4, 12, 6, 14);
    // A0B5 A1B5 A2B7 A3B7 A4B1 A5B1 A6B3 A7B3
    st[mvnum][5] = __builtin_shufflevector(outv[mvnum][6], outv[mvnum][2], 1, 9, 3, 11, 5, 13, 7, 15);
    // A0B6 A1B6 A2B4 A3B4 A4B2 A5B2 A6B0 A7B0
    st[mvnum][6] = __builtin_shufflevector(outv[mvnum][3], outv[mvnum][7], 0, 8, 2, 10, 4, 12, 6, 14);
    // A0B7 A1B7 A2B5 A3B5 A4B3 A5B3 A6B1 A7B1
    st[mvnum][7] = __builtin_shufflevector(outv[mvnum][7], outv[mvnum][3], 1, 9, 3, 11, 5, 13, 7, 15);

    // A0B0 A1B0 A2B0 A3B0 A4B4 A5B4 A6B4 A7B4
    st[mvnum][8]  = __builtin_shufflevector(st[mvnum][0], st[mvnum][2], 0, 1, 10, 11, 4, 5, 14, 15);
    // A0B1 A1B1 A2B1 A3B1 A4B5 A5B5 A6B5 A7B5
    st[mvnum][9]  = __builtin_shufflevector(st[mvnum][1], st[mvnum][3], 0, 1, 10, 11, 4, 5, 14, 15);
    // A0B2 A1B2 A2B2 A3B2 A4B6 A5B6 A6B6 A7B6
    st[mvnum][10] = __builtin_shufflevector(st[mvnum][2], st[mvnum][0], 0, 1, 10, 11, 4, 5, 14, 15);
    // A0B3 A1B3 A2B3 A3B3 A4B7 A5B7 A6B7 A7B7
    st[mvnum][11] = __builtin_shufflevector(st[mvnum][3], st[mvnum][1], 0, 1, 10, 11, 4, 5, 14, 15);
    // A0B4 A1B4 A2B4 A3B4 A4B0 A5B0 A6B0 A7B0
    st[mvnum][12] = __builtin_shufflevector(st[mvnum][4], st[mvnum][6], 0, 1, 10, 11, 4, 5, 14, 15);
    // A0B5 A1B5 A2B5 A3B5 A4B1 A5B1 A6B1 A7B1
    st[mvnum][13] = __builtin_shufflevector(st[mvnum][5], st[mvnum][7], 0, 1, 10, 11, 4, 5, 14, 15);
    // A0B6 A1B6 A2B6 A3B6 A4B2 A5B2 A6B2 A7B2
    st[mvnum][14] = __builtin_shufflevector(st[mvnum][6], st[mvnum][4], 0, 1, 10, 11, 4, 5, 14, 15);
    // A0B7 A1B7 A2B7 A3B7 A4B3 A5B3 A6B3 A7B3
    st[mvnum][15] = __builtin_shufflevector(st[mvnum][7], st[mvnum][5], 0, 1, 10, 11, 4, 5, 14, 15);

    // A0B0 A1B0 A2B0 A3B0 A4B0 A5B0 A6B0 A7B0
    st[mvnum][16] = __builtin_shufflevector(st[mvnum][8], st[mvnum][12], 0, 1, 2, 3, 12, 13, 14, 15);
    // A0B1 A1B1 A2B1 A3B1 A4B1 A5B1 A6B1 A7B1
    st[mvnum][17] = __builtin_shufflevector(st[mvnum][9], st[mvnum][13], 0, 1, 2, 3, 12, 13, 14, 15);
    // A0B2 A1B2 A2B2 A3B2 A4B2 A5B2 A6B2 A7B2
    st[mvnum][18] = __builtin_shufflevector(st[mvnum][10], st[mvnum][14], 0, 1, 2, 3, 12, 13, 14, 15);
    // A0B3 A1B3 A2B3 A3B3 A4B3 A5B3 A6B3 A7B3
    st[mvnum][19] = __builtin_shufflevector(st[mvnum][11], st[mvnum][15], 0, 1, 2, 3, 12, 13, 14, 15);
    // A0B4 A1B4 A2B4 A3B4 A4B4 A5B4 A6B4 A7B4
    st[mvnum][20] = __builtin_shufflevector(st[mvnum][12], st[mvnum][8], 0, 1, 2, 3, 12, 13, 14, 15);
    // A0B5 A1B5 A2B5 A3B5 A4B5 A5B5 A6B5 A7B5
    st[mvnum][21] = __builtin_shufflevector(st[mvnum][13], st[mvnum][9], 0, 1, 2, 3, 12, 13, 14, 15);
    // A0B6 A1B6 A2B6 A3B6 A4B6 A5B6 A6B6 A7B6
    st[mvnum][22] = __builtin_shufflevector(st[mvnum][14], st[mvnum][10], 0, 1, 2, 3, 12, 13, 14, 15);
    // A0B7 A1B7 A2B7 A3B7 A4B7 A5B7 A6B7 A7B7
    st[mvnum][23] = __builtin_shufflevector(st[mvnum][15], st[mvnum][11], 0, 1, 2, 3, 12, 13, 14, 15);

#pragma clang loop unroll(full)
    for (int64_t vnum = 0; vnum < nVec; ++vnum) {
#pragma clang loop unroll(full)
      for (int64_t vidx = 0; vidx < mVec; ++vidx) {
        out[(j + vnum) * outstride + (i + (mvnum * vWidth) + vidx)] +=
            st[mvnum][(2 * nVec) + vnum][vidx];
      }
    }
  }
}

// Base-case for the divide-and-conquer matmul.
template <typename F, int64_t k, bool transpose_lhs, bool transpose_rhs>
void matmul_base_block(F *__restrict__ out, const F *__restrict__ lhs,
                       const F *__restrict__ rhs, int64_t m, int64_t n,
                       int64_t mstride, int64_t nstride, int64_t kstride,
                       char order) noexcept {
#if PRINT
  fprintf(stderr, "matmul_base_block: lhs %p, rhs %p, m %ld, n %ld, k %ld, order %d\n", lhs, rhs, m, n, k, order);
#endif
  // The stride of the output is mstride.
  const int64_t outstride = mstride;

  // Initialize the lhs and rhs buffers from the inputs, transposing
  // the inputs as necessary.

  thread_local F lhsTmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  thread_local F rhsTmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  thread_local const F *last_lhs = nullptr;
  thread_local const F *last_rhs = nullptr;

  if (last_lhs != lhs) {
    buffer_init<F, transpose_lhs, transpose_lhs>(lhsTmp, lhs, m, k, mstride,
                                                 kstride);
    last_lhs = lhs;
  }
  if (last_rhs != rhs) {
    buffer_init<F, transpose_rhs, !transpose_rhs>(rhsTmp, rhs, k, n, kstride,
                                                  nstride);
    last_rhs = rhs;
  }

  const int64_t mBlk = mVBlk; // 3 * mVec;
  const int64_t mW = 1 * mBlk;
  const int64_t nBlk = nVec;
  const int64_t nW = 1 * nBlk;

  // Handle a vectorizable hyperrectangle whose sides are multiples of
  // nVec and mVec.  This hyperrectangle can be handled fully with
  // vector operations using matmul_vec_op.

  for (int64_t jj = 0; jj < n / nW; ++jj) {
    for (int64_t ii = 0; ii < m / mW; ++ii) {
      for (int64_t j = jj * (nW / nBlk); j < (jj + 1) * (nW / nBlk); ++j) {
	for (int64_t i = ii * (mW / mBlk); i < (ii + 1) * (mW / mBlk); ++i) {
          matmul_vec_x8_block<F, (mBlk / mVec), nBlk, k, false, true>(
              out, lhsTmp, rhsTmp, mBlk * i, nBlk * j, m, n, k,
              outstride);

          // matmul_vec_op_ext<F, (mBlk / mVec), nBlk>(out, lhsTmp, rhsTmp,
	  //                                           mBlk * i, nBlk * j, 0, k,
	  //                                           m, n, k, outstride);
	}
      }
    }
  }

  if (mW * (m / mW) < m) {
    // Handle extra entries in the m dimension.
    int64_t i = (mW * (m / mW));
    int64_t mRem = m - i;
    const int64_t mBlk2 = 2 * mVec;
    const int64_t mBlk1 = 1 * mVec;
    switch (mRem / mVec) {
    case 2: {
      for (int64_t jj = 0; jj < n / nW; ++jj) {
        for (int64_t j = jj * (nW / nBlk); j < (jj + 1) * (nW / nBlk); ++j) {
          matmul_vec_x8_block<F, 2, nBlk, k, false, true>(
              out, lhsTmp, rhsTmp, i, nBlk * j, m, n, k, outstride);
        }
      }
      i += mBlk2;
      mRem -= mBlk2;
      break;
    }
    case 1: {
      for (int64_t jj = 0; jj < n / nW; ++jj) {
        for (int64_t j = jj * (nW / nBlk); j < (jj + 1) * (nW / nBlk); ++j) {
          matmul_vec_x8_block<F, 1, nBlk, k, false, true>(
              out, lhsTmp, rhsTmp, i, nBlk * j, m, n, k, outstride);
        }
      }
      i += mBlk1;
      mRem -= mBlk1;
      break;
    }
    }

    if (mRem) {
      for (int64_t jj = 0; jj < n / nW; ++jj) {
	for (int64_t j = jj * (nW / nBlk); j < (jj + 1) * (nW / nBlk); ++j) {
	  matmul_vec_flex_col<F, nBlk, mW>(out, lhsTmp, rhsTmp, i, j * nBlk, 0,
					   mRem, k, m, n, k, outstride);
	}
      }
    }
  }

  if (nW * (n / nW) < n) {
    // Handle extra entries in the n dimension.
    for (int64_t ii = 0; ii < m / mVec; ++ii) {
      matmul_vec_flex<F, mVec, nW>(
          out, lhsTmp, rhsTmp, mVec * ii, nW * (n / nW), 0,
          n - (nW * (n / nW)), k, m, n, k, outstride);
    }
    // We permute the order of loops here to exploit spatial locality
    // in out and lhs.
    for (int64_t j = nW * (n / nW); j < n; ++j) {
      for (int64_t l = 0; l < k; ++l) {
        for (int64_t i = mVec * (m / mVec); i < m; ++i) {
          out[j * outstride + i] += BUF_INDEX(lhsTmp, l, k, i, m, false) *
                                    BUF_INDEX(rhsTmp, j, n, l, k, true);
        }
      }
    }
  }
}

// Base-case for the divide-and-conquer matmul.
template <typename F, int64_t mBase, int64_t nBase, int64_t kBase,
          bool transpose_lhs, bool transpose_rhs>
void matmul_base(F *__restrict__ out, const F *__restrict__ lhs,
                 const F *__restrict__ rhs, int64_t m, int64_t n, int64_t k,
                 int64_t mstride, int64_t nstride, int64_t kstride,
                 char order) noexcept {
#if PRINT
  fprintf(stderr, "matmul_base: lhs %p, rhs %p, m %ld, n %ld, k %ld, order %d\n", lhs, rhs, m, n, k, order);
#endif
  // The stride of the output is mstride.
  const int64_t outstride = mstride;

  // Initialize the lhs and rhs buffers from the inputs, transposing
  // the inputs as necessary.

  thread_local F lhsTmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  thread_local F rhsTmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  thread_local const F *last_lhs = nullptr;
  thread_local const F *last_rhs = nullptr;
  thread_local int64_t last_k = 0;

  if (last_lhs != lhs|| last_k != k) {
    buffer_init<F, mBase, kBase, transpose_lhs, transpose_lhs>(
        lhsTmp, lhs, m, k, mstride, kstride);

    last_lhs = lhs;
  }
  if (last_rhs != rhs || last_k != k) {
    buffer_init<F, kBase, nBase, transpose_rhs, !transpose_rhs>(
        rhsTmp, rhs, k, n, kstride, nstride);
    last_rhs = rhs;
  }
  last_k = k;

  // thread_local F lhs0Tmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  // thread_local F rhs0Tmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  // thread_local F lhs1Tmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  // thread_local F rhs1Tmp[BASE / sizeof(F)] __attribute__((aligned(64)));
  // thread_local const F *last_lhs0 = nullptr;
  // thread_local const F *last_rhs0 = nullptr;
  // thread_local const F *last_lhs1 = nullptr;
  // thread_local const F *last_rhs1 = nullptr;
  // thread_local int64_t last_m0 = 0;
  // thread_local int64_t last_n0 = 0;
  // thread_local int64_t last_m1 = 0;
  // thread_local int64_t last_n1 = 0;
  // thread_local int64_t last_lk0 = 0;
  // thread_local int64_t last_lk1 = 0;
  // thread_local int64_t last_rk0 = 0;
  // thread_local int64_t last_rk1 = 0;

  // F *lhsTmp = nullptr, *rhsTmp = nullptr;
  // switch (order) {
  // case 0b00: {
  //   lhsTmp = lhs0Tmp;
  //   rhsTmp = rhs0Tmp;
  //   if (last_lhs0 != lhs || last_m0 != m || last_lk0 != k) {
  //     buffer_init<F, transpose_lhs, transpose_lhs>(lhsTmp, lhs, m, k, mstride,
  //                                                  kstride);
  //     last_lhs0 = lhs;
  //     last_m0 = m;
  //     last_lk0 = k;
  //   }
  //   if (last_rhs0 != rhs || last_n0 != n || last_rk0 != k) {
  //     buffer_init<F, transpose_rhs, !transpose_rhs>(rhsTmp, rhs, k, n, kstride,
  //                                                   nstride);
  //     last_rhs0 = rhs;
  //     last_n0 = n;
  //     last_rk0 = k;
  //   }
  //   break;
  // }
  // case 0b01: {
  //   lhsTmp = lhs1Tmp;
  //   rhsTmp = rhs0Tmp;
  //   if (last_lhs1 != lhs || last_m1 != m || last_lk1 != k) {
  //     buffer_init<F, transpose_lhs, transpose_lhs>(lhsTmp, lhs, m, k, mstride,
  //                                                  kstride);
  //     last_lhs1 = lhs;
  //     last_m1 = m;
  //     last_lk1 = k;
  //   }
  //   if (last_rhs0 != rhs || last_n0 != n || last_rk0 != k) {
  //     buffer_init<F, transpose_rhs, !transpose_rhs>(rhsTmp, rhs, k, n, kstride,
  //                                                   nstride);
  //     last_rhs0 = rhs;
  //     last_n0 = n;
  //     last_rk0 = k;
  //   }
  //   break;
  // }
  // case 0b10: {
  //   lhsTmp = lhs0Tmp;
  //   rhsTmp = rhs1Tmp;
  //   if (last_lhs0 != lhs || last_m0 != m || last_lk0 != k) {
  //     buffer_init<F, transpose_lhs, transpose_lhs>(lhsTmp, lhs, m, k, mstride,
  //                                                  kstride);
  //     last_lhs0 = lhs;
  //     last_m0 = m;
  //     last_lk0 = k;
  //   }
  //   if (last_rhs1 != rhs || last_n1 != n || last_rk1 != k) {
  //     buffer_init<F, transpose_rhs, !transpose_rhs>(rhsTmp, rhs, k, n, kstride,
  //                                                   nstride);
  //     last_rhs1 = rhs;
  //     last_n1 = n;
  //     last_rk1 = k;
  //   }
  //   break;
  // }
  // case 0b11: {
  //   lhsTmp = lhs1Tmp;
  //   rhsTmp = rhs1Tmp;
  //   if (last_lhs1 != lhs || last_m1 != m || last_lk1 != k) {
  //     buffer_init<F, transpose_lhs, transpose_lhs>(lhsTmp, lhs, m, k, mstride,
  //                                                  kstride);
  //     last_lhs1 = lhs;
  //     last_m1 = m;
  //     last_lk1 = k;
  //   }
  //   if (last_rhs1 != rhs || last_n1 != n || last_rk1 != k) {
  //     buffer_init<F, transpose_rhs, !transpose_rhs>(rhsTmp, rhs, k, n, kstride,
  //                                                   nstride);
  //     last_rhs1 = rhs;
  //     last_n1 = n;
  //     last_rk1 = k;
  //   }
  //   break;
  // }
  // default:
  //   assert(false && "Unknown order");
  //   return;
  // }

  const int64_t mBlk = mVBlk;
  const int64_t mW = 1 * mBlk;
  const int64_t nBlk = nVec;
  const int64_t nW = 1 * nBlk;

  // Handle a vectorizable hyperrectangle whose sides are multiples of
  // nW and mW.  This hyperrectangle can be handled fully with vector
  // operations using matmul_vec_op.
  for (int64_t jj = 0; jj < n / nW; ++jj) {
    for (int64_t ii = 0; ii < m / mW; ++ii) {
      for (int64_t j = jj * (nW / nBlk); j < (jj + 1) * (nW / nBlk); ++j) {
	for (int64_t i = ii * (mW / mBlk); i < (ii + 1) * (mW / mBlk); ++i) {
          matmul_vec_x8<F, (mBlk / mVec), nBlk, false, true>(
              out, lhsTmp, rhsTmp, mBlk * i, nBlk * j, 0, k, m, n, k,
              outstride);

          // matmul_vec_op_ext<F, (mBlk / mVec), nBlk>(out, lhsTmp, rhsTmp,
          //                                           mBlk * i, nBlk * j, 0, k,
          //                                           m, n, k, outstride);
	}
      }
    }
  }

  if (mW * (m / mW) < m) {
    // Handle extra entries in the m dimension.
    int64_t i = (mW * (m / mW));
    int64_t mRem = m - i;
    const int64_t mBlk2 = 2 * mVec;
    const int64_t mBlk1 = 1 * mVec;
    switch (mRem / mVec) {
    case 2: {
      for (int64_t jj = 0; jj < n / nW; ++jj) {
        for (int64_t j = jj * (nW / nBlk); j < (jj + 1) * (nW / nBlk); ++j) {
          matmul_vec_x8<F, 2, nBlk, false, true>(
              out, lhsTmp, rhsTmp, i, nBlk * j, 0, k, m, n, k, outstride);
        }
      }
      i += mBlk2;
      mRem -= mBlk2;
      break;
    }
    case 1: {
      for (int64_t jj = 0; jj < n / nW; ++jj) {
        for (int64_t j = jj * (nW / nBlk); j < (jj + 1) * (nW / nBlk); ++j) {
          matmul_vec_x8<F, 1, nBlk, false, true>(
              out, lhsTmp, rhsTmp, i, nBlk * j, 0, k, m, n, k, outstride);
        }
      }
      i += mBlk1;
      mRem -= mBlk1;
      break;
    }
    }

    if (mRem) {
      for (int64_t jj = 0; jj < n / nW; ++jj) {
	for (int64_t j = jj * (nW / nBlk); j < (jj + 1) * (nW / nBlk); ++j) {
	  matmul_vec_flex_col<F, nBlk, mW>(out, lhsTmp, rhsTmp, i, j * nBlk, 0,
					   mRem, k, m, n, k, outstride);
	}
      }
    }
  }
  if (nW * (n / nW) < n) {
    // Handle extra entries in the n dimension.
    for (int64_t ii = 0; ii < m / mVec; ++ii) {
      matmul_vec_flex<F, mVec, nW>(
          out, lhsTmp, rhsTmp, mVec * ii, nW * (n / nW), 0,
          n - (nW * (n / nW)), k, m, n, k, outstride);
    }
    // We permute the order of loops here to exploit spatial locality
    // in out and lhs.
    for (int64_t j = nW * (n / nW); j < n; ++j) {
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
                int64_t mstride, int64_t nstride, int64_t kstride, char order) noexcept {
  // if (m == 0 || n == 0 || k == 0)
  //   return;

  // const int64_t pref_n = 128, pref_m = 128, pref_k = 2 * kBlk; // 256
  // const int64_t pref_n = 128, pref_m = 144, pref_k = 1 * kBlk; // 384

  // const int64_t pref_n = 24, pref_m = 192, pref_k = 1 * kBlk; // 384
  // const int64_t pref_n = 24, pref_m = 192, pref_k = 1 * kBlk; // 384
  // const int64_t pref_n = 192, pref_m = 24, pref_k = 1 * kBlk; // 384

  // // low number of copy, transpose, base-case invokes, LLC refs, but poor Cilk performance.
  // const int64_t pref_n = 360, pref_m = 240, pref_k = 1 * kBlk; // 384

  // // low number of copy, transpose, base-case invokes, LLC refs
  // const int64_t pref_n = 384, pref_m = 216, pref_k = 1 * kBlk; // 384, BASE 2506752
  // const int64_t pref_n = 384, pref_m = 192, pref_k = 1 * kBlk; // 384, BASE 2359296
  // const int64_t pref_n = 384, pref_m = 168, pref_k = 1 * kBlk; // 384, BASE 2211840

  // const int64_t pref_n = 360, pref_m = 192, pref_k = 1 * kBlk; // 384, BASE 2248704
  // const int64_t pref_n = 240, pref_m = 384, pref_k = 1 * kBlk; // 384, BASE 2654208
  // const int64_t pref_n = 4096, pref_m = 24, pref_k = 1 * kBlk; // 384, BASE 13443072
  // const int64_t pref_n = 4096, pref_m = 48, pref_k = 1 * kBlk; // 384, BASE 14303232

  // const int64_t pref_n = 128, pref_m = 48, pref_k = 1 * kBlk; // 512, BASE 770048, Best 20230502  (P=48 ~> 0.053s)
  const int64_t pref_n = 128, pref_m = 48, pref_k = 1 * kBlk; // 384, BASE 589824; 256, BASE 409600
  // const int64_t pref_n = 128, pref_m = 72, pref_k = 1 * kBlk; // 384, BASE 589824; 256, BASE 409600

  // const int64_t pref_n = 128, pref_m = 24, pref_k = 1 * kBlk; // 512, BASE 647168
  // const int64_t pref_n = 144, pref_m = 24, pref_k = 1 * kBlk; // 512, BASE 647168

  // const int64_t pref_n = 144, pref_m = 72, pref_k = 1 * kBlk; // 512, BASE 

  // const int64_t pref_n = 288, pref_m = 192, pref_k = 1 * kBlk; // 384
  // const int64_t pref_n = 128, pref_m = 192, pref_k = 1 * kBlk; // 512

  // const int64_t pref_n = 128, pref_m = 72, pref_k = 1 * kBlk; // 512, BASE 32k * 28, best 20230501
  // const int64_t pref_n = 120, pref_m = 72, pref_k = 1 * kBlk; // 512

  // const int64_t pref_n = 168, pref_m = 48, pref_k = 1 * kBlk; // 512, BASE 32k * 29, best 20230501?

  // const int64_t pref_n = 64, pref_m = 96, pref_k = 1 * kBlk; // 768
  // const int64_t pref_n = 64, pref_m = 48, pref_k = 1 * kBlk; // 1024

  
  // Check that the total size of the submatrices fits within BASE
  // bytes.
  if ((m * k) + (n * k) + (m * n) <= BASE / sizeof(F)) {
    matmul_base<F, pref_m, pref_n, pref_k, transpose_lhs, transpose_rhs>(
        out, lhs, rhs, m, n, k, mstride, nstride, kstride, order);
    return;
  }

  // if (n > 2048) {
  //   const int64_t split_rounded = ((split_dim(n) / pref_n) * pref_n);
  //   const int64_t split = (split_rounded == 0) ? pref_n : split_rounded;
  //   cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
  //       out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, split,
  //       k, mstride, nstride, kstride, order);
  //   matmul_dac<F, transpose_lhs, transpose_rhs>(
  //       out + (split * mstride),
  //       &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //       &ARG_INDEX(rhs, split, nstride, 0, kstride, transpose_rhs), m,
  //       n - split, k, mstride, nstride, kstride, order);
  //   cilk_sync;

  // } else
    if (m > 1024) {
    const int64_t split_rounded = ((split_dim(m) / pref_m) * pref_m);
    const int64_t split = (split_rounded == 0) ? pref_m : split_rounded;
    cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n,
        k, mstride, nstride, kstride, order);
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out + split,
        &ARG_INDEX(lhs, 0, kstride, split, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m - split, n, k,
        mstride, nstride, kstride, order);
    // cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
    //     out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
    //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), (m - split), n,
    //     k, mstride, nstride, kstride, order);
    // matmul_dac<F, transpose_lhs, transpose_rhs>(
    //     out + (m - split),
    //     &ARG_INDEX(lhs, 0, kstride, (m - split), mstride, transpose_lhs),
    //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n, k,
    //     mstride, nstride, kstride, order);
    cilk_sync;

  } else if (k > pref_k) {
    // const int64_t split = split_dim(k);
    const int64_t split = pref_k;
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, n, split,
        mstride, nstride, kstride, order);
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, split, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, split, kstride, transpose_rhs), m, n,
        (k - split), mstride, nstride, kstride, order);

    // matmul_dac<F, transpose_lhs, transpose_rhs>(
    //     out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
    //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, n, (k - split),
    //     mstride, nstride, kstride, order);
    // matmul_dac<F, transpose_lhs, transpose_rhs>(
    //     out, &ARG_INDEX(lhs, (k - split), kstride, 0, mstride, transpose_lhs),
    //     &ARG_INDEX(rhs, 0, nstride, (k - split), kstride, transpose_rhs), m, n,
    //     split, mstride, nstride, kstride, order);

  // } else if (n > 1024) {
  //   const int64_t split_rounded = ((split_dim(n) / pref_n) * pref_n);
  //   const int64_t split = (split_rounded == 0) ? pref_n : split_rounded;
  //   cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
  //       out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, split,
  //       k, mstride, nstride, kstride, order);
  //   matmul_dac<F, transpose_lhs, transpose_rhs>(
  //       out + (split * mstride),
  //       &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //       &ARG_INDEX(rhs, split, nstride, 0, kstride, transpose_rhs), m,
  //       n - split, k, mstride, nstride, kstride, order);
  //   cilk_sync;

  // } else if (m > 2048) {
  //   const int64_t split_rounded = ((split_dim(m) / pref_m) * pref_m);
  //   const int64_t split = (split_rounded == 0) ? pref_m : split_rounded;
  //   cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
  //       out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n,
  //       k, mstride, nstride, kstride, order);
  //   matmul_dac<F, transpose_lhs, transpose_rhs>(
  //       out + split,
  //       &ARG_INDEX(lhs, 0, kstride, split, mstride, transpose_lhs),
  //       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m - split, n, k,
  //       mstride, nstride, kstride, order);
  //   // cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
  //   //     out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //   //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), (m - split), n,
  //   //     k, mstride, nstride, kstride, order);
  //   // matmul_dac<F, transpose_lhs, transpose_rhs>(
  //   //     out + (m - split),
  //   //     &ARG_INDEX(lhs, 0, kstride, (m - split), mstride, transpose_lhs),
  //   //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n, k,
  //   //     mstride, nstride, kstride, order);
  //   cilk_sync;

  } else if (n > pref_n) {
    const int64_t split_rounded = ((split_dim(n) / pref_n) * pref_n);
    const int64_t split = (split_rounded == 0) ? pref_n : split_rounded;
    // if (order & 0b01) {
    //   cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
    //       out + (split * mstride),
    //       &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
    //       &ARG_INDEX(rhs, split, nstride, 0, kstride, transpose_rhs), m,
    //       (n - split), k, mstride, nstride, kstride, (order & 0b01) | 0b10);
    //   matmul_dac<F, transpose_lhs, transpose_rhs>(
    //       out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
    //       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, split, k,
    //       mstride, nstride, kstride, (order & 0b01) | 0b00);
    // } else {
      cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
          out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
          &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, split, k,
          mstride, nstride, kstride, (order & 0b01) | 0b00);
      matmul_dac<F, transpose_lhs, transpose_rhs>(
          out + (split * mstride),
          &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
          &ARG_INDEX(rhs, split, nstride, 0, kstride, transpose_rhs), m,
          (n - split), k, mstride, nstride, kstride, (order & 0b01) | 0b10);

      // cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
      //     out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
      //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, (n - split), k,
      //     mstride, nstride, kstride, (order & 0b01) | 0b00);
      // matmul_dac<F, transpose_lhs, transpose_rhs>(
      // 						  out + ((n - split) * mstride),
      //     &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
      //     &ARG_INDEX(rhs, (n - split), nstride, 0, kstride, transpose_rhs), m,
      //     split, k, mstride, nstride, kstride, (order & 0b01) | 0b10);
    // }
    cilk_sync;
  } else if (m > pref_m) {
    // const int64_t split = split_dim(m);
    const int64_t split_rounded = ((split_dim(m) / pref_m) * pref_m);
    const int64_t split = (split_rounded == 0) ? pref_m : split_rounded;
    if (order & 0b10) {
      cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
          out + split,
          &ARG_INDEX(lhs, 0, kstride, split, mstride, transpose_lhs),
          &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), (m - split),
          n, k, mstride, nstride, kstride, (order & 0b10) | 0b01);
      matmul_dac<F, transpose_lhs, transpose_rhs>(
          out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
          &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n, k,
          mstride, nstride, kstride, (order & 0b10) | 0b00);

      // cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
      //     out + (m - split),
      //     &ARG_INDEX(lhs, 0, kstride, (m - split), mstride, transpose_lhs),
      //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n, k,
      //     mstride, nstride, kstride, (order & 0b10) | 0b01);
      // matmul_dac<F, transpose_lhs, transpose_rhs>(
      //     out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
      //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), (m - split),
      //     n, k, mstride, nstride, kstride, (order & 0b10) | 0b00);
    } else {
      cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
          out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
          &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n, k,
          mstride, nstride, kstride,  (order & 0b10) | 0b00);
      matmul_dac<F, transpose_lhs, transpose_rhs>(
          out + split,
          &ARG_INDEX(lhs, 0, kstride, split, mstride, transpose_lhs),
          &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m - split, n,
          k, mstride, nstride, kstride, (order & 0b10) | 0b01);

      // cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
      //     out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
      //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), (m - split), n, k,
      //     mstride, nstride, kstride,  (order & 0b10) | 0b00);
      // matmul_dac<F, transpose_lhs, transpose_rhs>(
      // 						  out + (m - split),
      //     &ARG_INDEX(lhs, 0, kstride, (m - split), mstride, transpose_lhs),
      //     &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split,
      //     n, k, mstride, nstride, kstride, (order & 0b10) | 0b01);
    }
    cilk_sync;
  } else { // max_dim == k
    const int64_t split = split_dim(k);
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, n, split,
        mstride, nstride, kstride, order);
    matmul_dac<F, transpose_lhs, transpose_rhs>(
        out, &ARG_INDEX(lhs, split, kstride, 0, mstride, transpose_lhs),
        &ARG_INDEX(rhs, 0, nstride, split, kstride, transpose_rhs), m, n,
        (k - split), mstride, nstride, kstride, order);
  }

  // const int64_t skew_m = 1, skew_n = 1, skew_k = 1; // 4; 2; 1;
  // // const int64_t skew_m = 2, skew_n = 1;
  // // Split the maximum dimension
  // const int64_t max_dim = std::max(std::max(m / skew_m, n / skew_n), k / skew_k);
  // // We prefer to spawn higher in the recursion tree than lower.
  // // Because the base case vectorizes over dimension m, which is the
  // // fastest moving dimension of the output matrix, we prefer to split
  // // n instead of m.
  // if (max_dim == n / skew_n) {
  //   const int64_t split = split_dim(n);
  //   if (order & 0b01) {
  //     cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
  //         out + (split * mstride),
  //         &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //         &ARG_INDEX(rhs, split, nstride, 0, kstride, transpose_rhs), m,
  //         (n - split), k, mstride, nstride, kstride, (order & 0b01) | 0b10);
  //     matmul_dac<F, transpose_lhs, transpose_rhs>(
  //         out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //         &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, split, k,
  //         mstride, nstride, kstride, (order & 0b01) | 0b00);
  //   } else {
  //     cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
  //         out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //         &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, split, k,
  //         mstride, nstride, kstride, (order & 0b01) | 0b00);
  //     matmul_dac<F, transpose_lhs, transpose_rhs>(
  //         out + (split * mstride),
  //         &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //         &ARG_INDEX(rhs, split, nstride, 0, kstride, transpose_rhs), m,
  //         (n - split), k, mstride, nstride, kstride, (order & 0b01) | 0b10);
  //   }
  //   cilk_sync;
  // } else if (max_dim == m / skew_m) {
  //   const int64_t split = split_dim(m);
  //   if (order & 0b10) {
  //     cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
  //         out + split,
  //         &ARG_INDEX(lhs, 0, kstride, split, mstride, transpose_lhs),
  //         &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), (m - split),
  //         n, k, mstride, nstride, kstride, (order & 0b10) | 0b01);
  //     matmul_dac<F, transpose_lhs, transpose_rhs>(
  //         out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //         &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n, k,
  //         mstride, nstride, kstride, (order & 0b10) | 0b00);
  //   } else {
  //     cilk_spawn matmul_dac<F, transpose_lhs, transpose_rhs>(
  //         out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //         &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), split, n, k,
  //         mstride, nstride, kstride,  (order & 0b10) | 0b00);
  //     matmul_dac<F, transpose_lhs, transpose_rhs>(
  //         out + split,
  //         &ARG_INDEX(lhs, 0, kstride, split, mstride, transpose_lhs),
  //         &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), (m - split),
  //         n, k, mstride, nstride, kstride, (order & 0b10) | 0b01);
  //   }
  //   cilk_sync;
  // } else { // max_dim == k
  //   const int64_t split = split_dim(k);
  //   matmul_dac<F, transpose_lhs, transpose_rhs>(
  //       out, &ARG_INDEX(lhs, 0, kstride, 0, mstride, transpose_lhs),
  //       &ARG_INDEX(rhs, 0, nstride, 0, kstride, transpose_rhs), m, n, split,
  //       mstride, nstride, kstride, order);
  //   matmul_dac<F, transpose_lhs, transpose_rhs>(
  //       out, &ARG_INDEX(lhs, split, kstride, 0, mstride, transpose_lhs),
  //       &ARG_INDEX(rhs, 0, nstride, split, kstride, transpose_rhs), m, n,
  //       (k - split), mstride, nstride, kstride, order);
  // }

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

template <typename F, bool initOutput = false>
INLINEATTR void matmul(F *__restrict__ out, const F *__restrict__ lhs,
                       const F *__restrict__ rhs, int64_t m, int64_t n,
                       int64_t k, int32_t transpose_lhs,
                       int32_t transpose_rhs) noexcept {
  if (initOutput) {
    // Initialize output to zero.
    zero_init(out, m, n, m, n);
  }

  if (transpose_lhs && transpose_rhs) {
    matmul_dac<F, true, true>(out, lhs, rhs, m, n, k, m, n, k, 0);
  } else if (transpose_lhs && !transpose_rhs) {
    matmul_dac<F, true, false>(out, lhs, rhs, m, n, k, m, n, k, 0);
  } else if (!transpose_lhs && transpose_rhs) {
    matmul_dac<F, false, true>(out, lhs, rhs, m, n, k, m, n, k, 0);
  } else { // (!transpose_lhs && !transpose_rhs)
    matmul_dac<F, false, false>(out, lhs, rhs, m, n, k, m, n, k, 0);
  }
}

template <typename F>
INLINEATTR void matmul_ploops(F *__restrict__ out, const F *__restrict__ lhs,
                              const F *__restrict__ rhs, int64_t m, int64_t n,
                              int64_t k, int32_t transpose_lhs,
                              int32_t transpose_rhs) {
  if (transpose_lhs && transpose_rhs) {
    if (n > m) {
      cilk_for(int64_t i = 0; i < m; ++i) {
        cilk_for(int64_t j = 0; j < n; ++j) {
          out[j * m + i] = 0.0;
          for (int64_t l = 0; l < k; ++l)
            out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, true) *
                              ARG_INDEX(rhs, j, n, l, k, true);
        }
      }
    } else {
      cilk_for(int64_t j = 0; j < n; ++j) {
        cilk_for(int64_t i = 0; i < m; ++i) {
          out[j * m + i] = 0.0;
          for (int64_t l = 0; l < k; ++l)
            out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, true) *
                              ARG_INDEX(rhs, j, n, l, k, true);
        }
      }
    }
  } else if (transpose_lhs && !transpose_rhs) {
    if (n > m) {
      cilk_for(int64_t i = 0; i < m; ++i) {
        cilk_for(int64_t j = 0; j < n; ++j) {
          out[j * m + i] = 0.0;
          for (int64_t l = 0; l < k; ++l)
            out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, true) *
                              ARG_INDEX(rhs, j, n, l, k, false);
        }
      }
    } else {
      cilk_for(int64_t j = 0; j < n; ++j) {
        cilk_for(int64_t i = 0; i < m; ++i) {
          out[j * m + i] = 0.0;
          for (int64_t l = 0; l < k; ++l)
            out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, true) *
                              ARG_INDEX(rhs, j, n, l, k, false);
        }
      }
    }
  } else if (!transpose_lhs && transpose_rhs) {
    if (n > m) {
      cilk_for(int64_t i = 0; i < m; ++i) {
        cilk_for(int64_t j = 0; j < n; ++j) {
          out[j * m + i] = 0.0;
          for (int64_t l = 0; l < k; ++l)
            out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, false) *
                              ARG_INDEX(rhs, j, n, l, k, true);
        }
      }
    } else {
      cilk_for(int64_t j = 0; j < n; ++j) {
        cilk_for(int64_t i = 0; i < m; ++i) {
          out[j * m + i] = 0.0;
          for (int64_t l = 0; l < k; ++l)
            out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, false) *
                              ARG_INDEX(rhs, j, n, l, k, true);
        }
      }
    }
  } else { // (!transpose_lhs && !transpose_rhs)
    if (n > m) {
      cilk_for(int64_t i = 0; i < m; ++i) {
        cilk_for(int64_t j = 0; j < n; ++j) {
          out[j * m + i] = 0.0;
          for (int64_t l = 0; l < k; ++l)
            out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, false) *
                              ARG_INDEX(rhs, j, n, l, k, false);
        }
      }
    } else {
      cilk_for(int64_t j = 0; j < n; ++j) {
        cilk_for(int64_t i = 0; i < m; ++i) {
          out[j * m + i] = 0.0;
          for (int64_t l = 0; l < k; ++l)
            out[j * m + i] += ARG_INDEX(lhs, l, k, i, m, false) *
                              ARG_INDEX(rhs, j, n, l, k, false);
        }
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
