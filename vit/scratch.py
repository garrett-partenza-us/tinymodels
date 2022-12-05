import numpy as np

def transposed_conv3d(input_image, filter_size, stride, padding):
  # Get the dimensions of the input image
  n_c, n_h, n_w = input_image.shape

  # Calculate the dimensions of the output image
  o_h = (n_h - 1) * stride - 2 * padding + filter_size
  o_w = (n_w - 1) * stride - 2 * padding + filter_size

  # Create an empty output image with the calculated dimensions
  output_image = np.zeros((n_c, o_h, o_w))

  # Loop over the output image, applying the transposed convolution operation
  for i in range(o_h):
    for j in range(o_w):
      # Apply the filter on the input image, using the current stride and padding
        print(input_image[:, i: i + filter_size, j: j + filter_size] * kernel)

  return output_image

kernel = np.ones((1,3,3))
patch = np.random.normal(size=(1,20,20))
out = transposed_conv3d(patch, 3, 2, 3)

print(out)