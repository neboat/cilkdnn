// ~/tapir/src/build/bin/clang++ matmul.cpp test.cpp -o test -fcilkplus -std=c++11 -ffast-math  -mavx -mfma -mavx2 -O3 -Wall -g

#include <cassert>
#include <cstdlib>
#include <iostream>

template <typename F>
void conv2d_loops(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
                  int64_t input_batch, int64_t input_rows, int64_t input_cols, int64_t input_channels,
                  int64_t kernel_rows, int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
                  int64_t output_rows, int64_t output_cols,
                  // Stride of the sliding window in each dimension
                  int64_t row_stride, int64_t col_stride,
                  int64_t padding_top, int64_t padding_bottom, int64_t padding_left, int64_t padding_right,
                  int64_t lhs_row_dilation, int64_t lhs_col_dilation,
                  int64_t rhs_row_dilation, int64_t rhs_col_dilation);

int main(int argc, char *argv[]) {
  int input_batch = 128;
  int input_rows = 28;
  int input_cols = 28;
  int input_channels = 64;
  int kernel_rows = 28;
  int kernel_cols = 28;
  int kernel_channels = 64;
  int kernel_filters = 128;
  int output_rows = 3;
  int output_cols = 3;
  int row_stride = 1;
  int col_stride = 1;
  int padding_top = 1;
  int padding_bottom = 1;
  int padding_left = 1;
  int padding_right = 1;
  int lhs_row_dilation = 1;
  int lhs_col_dilation = 1;
  int rhs_row_dilation = 1;
  int rhs_col_dilation = 1;
  if (argc > 10) {
    input_batch = std::atoi(argv[1]);
    input_rows = std::atoi(argv[2]);
    input_cols = std::atoi(argv[3]);
    input_channels = std::atoi(argv[4]);
    kernel_rows = std::atoi(argv[5]);
    kernel_cols = std::atoi(argv[6]);
    kernel_channels = std::atoi(argv[7]);
    kernel_filters = std::atoi(argv[8]);
    output_rows = std::atoi(argv[9]);
    output_cols = std::atoi(argv[10]);
  }
  if (argc > 20) {
    row_stride = std::atoi(argv[11]);
    col_stride = std::atoi(argv[12]);
    padding_top = std::atoi(argv[13]);
    padding_bottom = std::atoi(argv[14]);
    padding_left = std::atoi(argv[15]);
    padding_right = std::atoi(argv[16]);
    lhs_row_dilation = std::atoi(argv[17]);
    lhs_col_dilation = std::atoi(argv[18]);
    rhs_row_dilation = std::atoi(argv[19]);
    rhs_col_dilation = std::atoi(argv[20]);
  }

  float *out = new float[input_batch * output_rows * output_cols * kernel_filters];
  float *outTmp = new float[input_batch * output_rows * output_cols * kernel_filters];

  float *input = new float[input_batch * input_rows * input_cols * input_channels];
  float *kernel = new float[kernel_rows * kernel_cols * kernel_channels * kernel_filters];
  // Create and initialize input matrices
  for (int64_t b = 0; b < input_batch; ++b)
    for (int64_t i = 0; i < input_rows; ++i)
      for (int64_t j = 0; j < input_cols; ++j)
        for (int64_t ch = 0; ch < input_channels; ++ch)
          input[(b * input_rows * input_cols * input_channels) +
                (i * input_cols * input_channels) +
                (j * input_channels) +
                ch] = (float)rand() / (float)RAND_MAX;
  for (int64_t i = 0; i < kernel_rows; ++i)
    for (int64_t j = 0; j < kernel_cols; ++j)
      for (int64_t ch = 0; ch < kernel_channels; ++ch)
        for (int64_t f = 0; f < kernel_filters; ++f)
          kernel[(i * kernel_cols * kernel_channels * kernel_filters) +
                 (j * kernel_channels * kernel_filters) +
                 (ch * kernel_filters) +
                 f] = (float)rand() / (float)RAND_MAX;

  conv2d_loops<float>(out, input, kernel,
                      input_batch, input_rows, input_cols, input_channels,
                      kernel_rows, kernel_cols, kernel_channels, kernel_filters,
                      output_rows, output_cols,
                      row_stride, col_stride,
                      padding_top, padding_bottom, padding_left, padding_right,
                      lhs_row_dilation, lhs_col_dilation,
                      rhs_row_dilation, rhs_col_dilation);
  conv2d_loops<float>(outTmp, input, kernel,
                      input_batch, input_rows, input_cols, input_channels,
                      kernel_rows, kernel_cols, kernel_channels, kernel_filters,
                      output_rows, output_cols,
                      row_stride, col_stride,
                      padding_top, padding_bottom, padding_left, padding_right,
                      lhs_row_dilation, lhs_col_dilation,
                      rhs_row_dilation, rhs_col_dilation);

  // Verify result
  int64_t diff_count = 0;
  for (int64_t b = 0; b < input_batch; ++b) {
    for (int64_t i = 0; i < output_rows; ++i) {
      for (int64_t j = 0; j < output_cols; ++j) {
        for (int64_t f = 0; f < kernel_filters; ++f) {
	  float outEl = out[(b * output_rows * output_cols * kernel_filters) +
			    (i * output_cols * kernel_filters) +
			    (j * kernel_filters) +
			    f];
	  float outTmpEl = outTmp[(b * output_rows * output_cols * kernel_filters) +
				  (i * output_cols * kernel_filters) +
				  (j * kernel_filters) +
				  f];
          float diff = outEl - outTmpEl;
          if (diff < 0)
            diff = -diff;
          if (!((diff < 0.01) || (diff/outEl < 0.00001))) {
            std::cerr << "Outputs differ: "
                      << "out[" << b << ", " << i << ", " << j << ", " << f << "] = "
                      << outEl << " "
                      << "outTmp[" << b << ", " << i << ", " << j << ", " << f << "] = "
                      << outTmpEl << "\n";
            diff_count++;
          }
        }
      }
    }
  }
  if (diff_count != 0)
    std::cerr << "Total differences " << diff_count << "\n";
  assert(diff_count == 0 && "Verify failed");

  // for (int trial = 0; trial < 50; ++trial)
  //   // matmul<float, false, true>(C, A, B, m, n, k);
  //   matmul<float>(C, A, B, m, n, k, transpose_lhs, transpose_rhs);
    
  delete[] out;
  delete[] outTmp;
  delete[] input;
  delete[] kernel;
  return 0;
}
