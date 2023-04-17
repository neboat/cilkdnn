import numpy as np
from PIL import Image
import requests
# from StringIO import StringIO
from io import StringIO
import tensorflow as tf
import cilknn

# Simple matmul test
A = np.array([[1.0, 2.0], [3.0, 4.0]],
             dtype=np.float32)
B = np.array([[1.0, 0.0], [0.0, 1.0]],
             dtype=np.float32)
print(np.array_equal(A, cilknn.matmul_f32(A, B, 2, 2, 2, 0, 0)))

def testCilkNNConv(array_4d, kernel_4d, strides, padding):
    graph = tf.Graph()
    with graph.as_default():
        tf_input_image = tf.Variable(np.array(array_4d, dtype = np.float32))
        tf_blur_kernel = tf.Variable(np.array(kernel_4d, dtype = np.float32))
        tf_convolution_output = tf.nn.conv2d(tf_input_image, tf_blur_kernel, [1] + strides + [1], padding)

    with tf.Session(graph = graph) as sess:
        # tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        transformed_image = tf_convolution_output.eval()
        transformed_image = transformed_image[0, :, :, 0]

    cilknn_input = np.array(array_4d, dtype=np.float32)
    cilknn_kernel = np.array(kernel_4d, dtype=np.float32)
    # padding is either 'SAME' or 'VALID'.
    cilknn_padding = [0,0,0,0]  # 'VALID' padding
    if padding == 'SAME':
        cilknn_padding = [cilknn_kernel.shape[0] / 2,
                          cilknn_kernel.shape[0] / 2,
                          cilknn_kernel.shape[1] / 2,
                          cilknn_kernel.shape[1] / 2]
    cilknn_out = cilknn.conv2d_f32(
        cilknn_input, cilknn_kernel, cilknn_input.shape, cilknn_kernel.shape,
        # Compute the output dimensions explicitly.
        (1 + ((cilknn_input.shape[1] + cilknn_padding[0] + cilknn_padding[1] - cilknn_kernel.shape[0]) / strides[0]),
         1 + ((cilknn_input.shape[2] + cilknn_padding[2] + cilknn_padding[3] - cilknn_kernel.shape[1]) / strides[1])),
        strides, cilknn_padding, [1,1,1,1])

    print(np.allclose(transformed_image, cilknn_out[0, :, :, 0]))

response = requests.get('http://vignette2.wikia.nocookie.net/grayscale/images/4/47/Lion.png/revision/latest?cb=20130926182831')
lion_arr = np.array(Image.open(StringIO(response.content)))

padded_array = np.pad(lion_arr, (1, 1), 'constant')
kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

blur_box_kernel = np.ones((3, 3)) / 9
blur_gaussian_kernel = np.array([[1,2,1],
                                 [2,4,2],
                                 [1,2,1]]) / 16

lion_array_4d = lion_arr.reshape(-1, 303, 497, 1)
blur_kernel_4d = blur_box_kernel.reshape(3, 3, 1, 1)
gauss_kernel_4d = blur_gaussian_kernel.reshape(3, 3, 1, 1)

testCilkNNConv(lion_array_4d, blur_kernel_4d, [1, 1], 'SAME')
testCilkNNConv(lion_array_4d, blur_kernel_4d, [2, 2], 'VALID')

testCilkNNConv(lion_array_4d, gauss_kernel_4d, [1, 1], 'SAME')
testCilkNNConv(lion_array_4d, gauss_kernel_4d, [2, 2], 'VALID')

# from tensorflow.contrib.compiler import xla

# def model_fn(x, y, z):
#     return tf.reduce_sum(x + y * z)

# def create_and_run_graph():
#     with tf.Session() as sess:
#         x = tf.placeholder(tf.float32, name='x')
#         y = tf.placeholder(tf.float32, name='y')
#         z = tf.placeholder(tf.float32, name='z')
#         result = xla.compile(computation=model_fn, inputs=(x, y, z))[0]
#         # `result` is a normal Tensor (albeit one that is computed by an XLA
#         # compiled executable) and can be used like any other Tensor.
#         result = tf.add(result, result)
#         return sess.run(result, feed_dict={ ... })
