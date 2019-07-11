#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numpy.hpp>
#include <iterator>

#include <cilkdnn/conv2d.hpp>
#include <cilkdnn/matmul.hpp>


using namespace boost::python;
namespace bn = boost::python::numpy;

extern "C"
void matmul_f32(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
                int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs);

bn::ndarray matmul_f32_wrap(const bn::ndarray lhs, const bn::ndarray rhs,
			    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
  tuple shape = make_tuple(n, m);
  bn::ndarray out = bn::empty(shape, bn::dtype::get_builtin<float>());
  matmul_f32(reinterpret_cast<float*>(out.get_data()),
	     reinterpret_cast<float*>(lhs.get_data()),
	     reinterpret_cast<float*>(rhs.get_data()),
	     m, n, k, transpose_lhs, transpose_rhs);
  return out;
}

extern "C"
void conv2d_f32(float *__restrict__ out, const float *__restrict__ lhs, const float *__restrict__ rhs,
                int64_t input_batch, int64_t input_rows, int64_t input_cols, int64_t input_channels,
                int64_t kernel_rows, int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
                int64_t output_rows, int64_t output_cols,
                int64_t row_stride, int64_t col_stride,
                int64_t padding_top, int64_t padding_bottom, int64_t padding_left, int64_t padding_right,
                int64_t lhs_row_dilation, int64_t lhs_col_dilation,
                int64_t rhs_row_dilation, int64_t rhs_col_dilation);

// We have to pack some of the arguments of conv2d_f32 to appease boost.python.
struct lhs_dims_t {
  int64_t input_batch;
  int64_t input_rows;
  int64_t input_cols;
  int64_t input_channels;
};

struct rhs_dims_t {
  int64_t kernel_rows;
  int64_t kernel_cols;
  int64_t kernel_channels;
  int64_t kernel_filters;
};

struct out_dims_t {
  int64_t output_rows;
  int64_t output_cols;
};

struct strides_t {
  int64_t row_stride;
  int64_t col_stride;
};

struct padding_t {
  int64_t padding_top;
  int64_t padding_bottom;
  int64_t padding_left;
  int64_t padding_right;
};

struct dilation_t {
  int64_t lhs_row_dilation;
  int64_t lhs_col_dilation;
  int64_t rhs_row_dilation;
  int64_t rhs_col_dilation;
};

bn::ndarray conv2d_f32_wrap(const bn::ndarray lhs, const bn::ndarray rhs,
			    tuple lhs_dims, tuple rhs_dims, tuple out_dims,
			    list strides, list padding, list dilation) {
			    // struct lhs_dims_t lhs_dims, struct rhs_dims_t rhs_dims, struct out_dims_t out_dims,
			    // struct strides_t strides, struct padding_t padding, struct dilation_t dilation) {
  tuple shape = make_tuple(lhs_dims[0],
			   out_dims[0],
			   out_dims[1],
			   rhs_dims[3]);
  bn::ndarray out = bn::empty(shape, bn::dtype::get_builtin<float>());
  conv2d_f32(reinterpret_cast<float*>(out.get_data()),
	     reinterpret_cast<float*>(lhs.get_data()),
	     reinterpret_cast<float*>(rhs.get_data()),
	     extract<int64_t>(lhs_dims[0]), extract<int64_t>(lhs_dims[1]), extract<int64_t>(lhs_dims[2]), extract<int64_t>(lhs_dims[3]),
	     extract<int64_t>(rhs_dims[0]), extract<int64_t>(rhs_dims[1]), extract<int64_t>(rhs_dims[2]), extract<int64_t>(rhs_dims[3]),
	     extract<int64_t>(out_dims[0]), extract<int64_t>(out_dims[1]),
	     extract<int64_t>(strides[0]), extract<int64_t>(strides[1]),
	     extract<int64_t>(padding[0]), extract<int64_t>(padding[1]), extract<int64_t>(padding[2]), extract<int64_t>(padding[3]),
	     extract<int64_t>(dilation[0]), extract<int64_t>(dilation[1]),
	     extract<int64_t>(dilation[2]), extract<int64_t>(dilation[3]));
	     // lhs_dims.input_batch, lhs_dims.input_rows, lhs_dims.input_cols, lhs_dims.input_channels,
	     // rhs_dims.kernel_rows, rhs_dims.kernel_cols, rhs_dims.kernel_channels, rhs_dims.kernel_filters,
	     // out_dims.output_rows, out_dims.output_cols,
	     // strides.row_stride, strides.col_stride,
	     // padding.padding_top, padding.padding_bottom, padding.padding_left, padding.padding_right,
	     // dilation.lhs_row_dilation, dilation.lhs_col_dilation,
	     // dilation.rhs_row_dilation, dilation.rhs_col_dilation);
  return out;
}

BOOST_PYTHON_MODULE(cilknn)
{
  bn::initialize();
  // Add regular functions to the module.
  def("matmul_f32", matmul_f32_wrap);
  def("conv2d_f32", conv2d_f32_wrap);
}
