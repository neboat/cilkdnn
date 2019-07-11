#pragma once

#include <cstdint>
#include <cilk/cilk.h>

#include "common_sysdep.hpp"


#define OUT_IDX(T, b, i, j, ch, batches, rows, cols, channels)          \
  ((T)[((b) * (rows) * (cols) * (channels)) +                           \
       ((i) * (cols) * (channels)) +                                    \
       ((j) * (channels)) +                                             \
       (ch)])

#define INPUT_IDX(T, b, i, j, ch, batches, rows, cols, channels)        \
  ((T)[((b) * (rows) * (cols) * (channels)) +                           \
       ((i) * (cols) * (channels)) +                                    \
       ((j) * (channels)) +                                             \
       (ch)])

#define KERNEL_IDX(T, i, j, ch, f, rows, cols, channels, filters)       \
  ((T)[((i) * (cols) * (channels) * (filters)) +                        \
       ((j) * (channels) * (filters)) +                                 \
       ((ch) * (filters)) +                                             \
       (f)])

// 2D convolution (actually 2D cross-correlation, but ML terminology
// calls this convolution).  Assumes standard NHWC data format in TF.
template <typename F>
INLINEATTR
void conv2d_loops(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
                  int64_t input_batch, int64_t input_rows, int64_t input_cols, int64_t input_channels,
                  int64_t kernel_rows, int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
                  int64_t output_rows, int64_t output_cols,
                  // Stride of the sliding window in each dimension
                  int64_t row_stride, int64_t col_stride,
                  int64_t padding_top, int64_t padding_bottom, int64_t padding_left, int64_t padding_right,
                  int64_t lhs_row_dilation, int64_t lhs_col_dilation,
                  int64_t rhs_row_dilation, int64_t rhs_col_dilation) {
  // In Eigen terms:
  //   kernel = patch
  //   rhs_col_dilation = in_row_stride
  //   rhs_row_dilation = in_col_stride
  //   lhs_col_dilation = row_inflate_stride
  //   lhs_row_dilation = col_inflate_stride

  // int64_t rows_with_pad = input_rows + padding_top + padding_bot;
  // int64_t cols_with_pad = input_rows + padding_left + padding_right;
  // F *padded_lhs = new F[input_batch *
  //                       rows_with_pad *
  //                       cols_with_pad *
  //                       input_channels];
  // for (int64_t b = 0; b < input_batch; ++b)
  //   for (int64_t in_i = 0; in_i < input_rows; ++in_i)
  //     for (int64_t in_j = 0; in_j < input_cols; ++in_j)
  //       for (int64_t in_ch = 0; in_ch < input_channels; ++in_ch)
  //         padded_lhs[(b * rows_with_pad * cols_with_pad * input_channels) +
  //                    ((in_i + padding_top) * cols_with_pad * input_channels) +
  //                    ((in_j + padding_left) * input_channels) +
  //                    in_ch] =
  //           lhs[(b * input_rows * input_cols * input_channels) +
  //               (in_i * input_cols * input_channels) +
  //               (in_j * input_channels) +
  //               in_ch];

  if (lhs_row_dilation == 1 && lhs_col_dilation == 1) {
    // std::cout << "Normal convolution\n";
    // Normal convolution.
    cilk_for (int64_t b = 0; b < input_batch; ++b) {
      cilk_for (int64_t oi = 0; oi < output_rows; ++oi) {
        cilk_for (int64_t oj = 0; oj < output_cols; ++oj) {
          cilk_for (int64_t och = 0; och < kernel_filters; ++och) {
            F accum = 0.0;
            for (int64_t di = 0; di < kernel_rows; ++di) {
              for (int64_t dj = 0; dj < kernel_cols; ++dj) {
                for (int64_t in_ch = 0; in_ch < input_channels; ++in_ch) {
                  int64_t in_i = (row_stride * oi) + (di * rhs_row_dilation) - padding_top;
                  int64_t in_j = (col_stride * oj) + (dj * rhs_col_dilation) - padding_left;
                  if (in_i < 0 || in_i >= input_rows || in_j < 0 || in_j >= input_cols)
                    continue;

                  accum +=
                    INPUT_IDX(lhs, b, in_i, in_j, in_ch,
                              input_batch, input_rows, input_cols, input_channels) *
                    KERNEL_IDX(rhs, di, dj, in_ch, och,
                               kernel_rows, kernel_cols, kernel_channels, kernel_filters);
                  // accum +=
                  //   lhs[(b * input_rows * input_cols * input_channels) +
                  //       (in_i * input_cols * input_channels) +
                  //       (in_j * input_channels) +
                  //       in_ch] *
                  //   rhs[(di * kernel_cols * kernel_channels * kernel_filters) +
                  //       (dj * kernel_channels * kernel_filters) +
                  //       (in_ch * kernel_filters) +
                  //       och];
                }
              }
            }
            OUT_IDX(out, b, oi, oj, och,
                    input_batch, output_rows, output_cols, kernel_filters) = accum;
            // out[(b * output_rows * output_cols * kernel_filters) +
            //     (oi * output_cols * kernel_filters) +
            //     (oj * kernel_filters) +
            //     och] = accum;
          }
        }
      }
    }
  } else {
    // Transpose convolution
    // std::cout << "Transpose convolution\n";
    cilk_for (int64_t b = 0; b < input_batch; ++b) {
      cilk_for (int64_t oi = 0; oi < output_rows; ++oi) {
        cilk_for (int64_t oj = 0; oj < output_cols; ++oj) {
          cilk_for (int64_t och = 0; och < kernel_filters; ++och) {
            F accum = 0.0;
            for (int64_t di = 0; di < kernel_rows / lhs_row_dilation; ++di) {
              for (int64_t dj = 0; dj < kernel_cols / lhs_col_dilation; ++dj) {
                for (int64_t in_ch = 0; in_ch < input_channels; ++in_ch) {
                  int64_t in_i = (oi + padding_top)/lhs_row_dilation - di;
                  int64_t in_j = (oj + padding_left)/lhs_col_dilation - dj;

                  int64_t k_i = (lhs_row_dilation * di) + ((oi + padding_top) % lhs_row_dilation);
                  int64_t k_j = (lhs_col_dilation * dj) + ((oj + padding_left) % lhs_col_dilation);
                  if (in_i < 0 || in_i >= input_rows || in_j < 0 || in_j >= input_cols)
                    continue;
                  if (in_i < 0 || in_i >= input_rows || in_j < 0 || in_j >= input_cols)
                    continue;

                  accum +=
                    INPUT_IDX(lhs, b, in_i, in_j, in_ch,
                              input_batch, input_rows, input_cols, input_channels) *
                    KERNEL_IDX(rhs, k_i, k_j, in_ch, och,
                               kernel_rows, kernel_cols, kernel_channels, kernel_filters);
                  // accum +=
                  //   lhs[(b * input_rows * input_cols * input_channels) +
                  //       (in_i * input_cols * input_channels) +
                  //       (in_j * input_channels) +
                  //       in_ch] *
                  //   rhs[(k_i * kernel_cols * kernel_channels * kernel_filters) +
                  //       (k_j * kernel_channels * kernel_filters) +
                  //       (in_ch * kernel_filters) +
                  //       och];
                }
              }
            }
            OUT_IDX(out, b, oi, oj, och,
                    input_batch, output_rows, output_cols, kernel_filters) = accum;
            
            // out[(b * output_rows * output_cols * kernel_filters) +
            //     (oi * output_cols * kernel_filters) +
            //     (oj * kernel_filters) +
            //     och] = accum;
          }
        }
      }
    }
  }
}

template void
conv2d_loops<float>(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
                    int64_t input_batch, int64_t input_rows, int64_t input_cols, int64_t input_channels,
                    int64_t kernel_rows, int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
                    int64_t output_rows, int64_t output_cols,
                    int64_t row_stride, int64_t col_stride,
                    int64_t padding_top, int64_t padding_bottom, int64_t padding_left, int64_t padding_right,
                    int64_t lhs_row_dilation, int64_t lhs_col_dilation,
                    int64_t rhs_row_dilation, int64_t rhs_col_dilation);

extern "C" {
INLINEATTR
void conv2d_f32(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
                int64_t input_batch, int64_t input_rows, int64_t input_cols, int64_t input_channels,
                int64_t kernel_rows, int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
                int64_t output_rows, int64_t output_cols,
                int64_t row_stride, int64_t col_stride,
                int64_t padding_top, int64_t padding_bottom, int64_t padding_left, int64_t padding_right,
                int64_t lhs_row_dilation, int64_t lhs_col_dilation,
                int64_t rhs_row_dilation, int64_t rhs_col_dilation) {
  conv2d_loops<float>(out, lhs, rhs, input_batch, input_rows, input_cols, input_channels, kernel_rows, kernel_cols,
                      kernel_channels, kernel_filters, output_rows, output_cols, row_stride, col_stride,
                      padding_top, padding_bottom, padding_left, padding_right, lhs_row_dilation, lhs_col_dilation,
                      rhs_row_dilation, rhs_col_dilation);
}
}


