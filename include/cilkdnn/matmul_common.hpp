#include <cstdint>

#define PRINT 0

// const int64_t BASE = 32768 * 28; // 25; // 32; // 24;
const int64_t BASE = 409600; // 483328; // 770048; // 589824;

// Old vec sizes for AVX512
// const int64_t nVec = 8;
// const int64_t mVec = 16;

const int64_t nVec = 8;
const int64_t mVec = 8;
const int64_t mVBlk = 3 * mVec;
const int64_t nVBlk = nVec;
const int64_t kBlk = 256; // 512; // 384; // 128;

template <bool rhs>
__attribute__((always_inline)) static int64_t index(int64_t i, int64_t m,
                                                    int64_t j, int64_t n) {
  // LHS: (i, m) -> k dimension, (j, n) -> m dimension
  // RHS: (i, m) -> n dimension, (j, n) -> k dimension
  int64_t res =
      rhs ? (((i / nVBlk) * n * nVBlk) + (j * nVBlk) + (i % nVBlk))
          : (((j / mVBlk) * m * mVBlk) + (i * mVBlk) + (j % mVBlk));
  return res;
}

#define BUF_INDEX(arg, ii, m, jj, n, rhs) (arg[index<rhs>(ii, m, jj, n)])

template <bool transpose>
__attribute__((always_inline)) static int64_t a_index(int64_t ii, int64_t m,
                                                      int64_t jj, int64_t n) {
  return transpose ? ((jj * m) + ii) : ((ii * n) + jj);
}

#define ARG_INDEX(arg, ii, m, jj, n, transpose)                                \
  (arg[a_index<transpose>(ii, m, jj, n)])
