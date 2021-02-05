from typing import *

import pyiree as iree
import pyiree.jax

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def correct_pad(inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.

  https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/
  applications/imagenet_utils.py#L388-L408

  Arguments:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.

  Returns:
    A tuple.
  """
  img_dim = 1
  input_size = inputs.shape[img_dim:(img_dim + 2)]
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if input_size[0] is None:
    adjust = (1, 1)
  else:
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
  correct = (kernel_size[0] // 2, kernel_size[1] // 2)
  return (
      (correct[0] - adjust[0], correct[0]),
      (correct[1] - adjust[1], correct[1]),
  )


def global_average_pooling(x):
  return jnp.mean(jnp.mean(x, 1), 1)


def flatten(x):
  return x.reshape((x.shape[0], -1))


def zero_pad_2d(padding):
  if isinstance(padding, int):
    padding = [padding, padding]
  padding = list(padding)
  assert len(padding) == 2
  if not isinstance(padding[0], Sequence):
    padding = [[pad, pad] for pad in padding]
  padding = [[0, 0]] + padding + [[0, 0]]
  return lambda x: jnp.pad(x, padding)


class DepthwiseConv2D(nn.Module):
  """
  Based on haiku.DepthwiseConv2D:
  https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/depthwise_conv.py
  and flax.linen.Conv:
  https://github.com/google/flax/blob/master/flax/linen/linear.py#L194-L287
  """
  kernel_size: Sequence[int]
  strides: Sequence[int] = (1, 1)
  feature_multiplier: int = 1
  padding: Union[str, Sequence[Tuple[int, int]]] = "same"
  use_bias: bool = True
  dtype: jax.numpy.dtype = jnp.float32
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    features = self.feature_multiplier * inputs.shape[-1]
    kernel = self.param("kernel", self.kernel_init,
                        self.kernel_size + (1, features))
    kernel = jnp.asarray(kernel, self.dtype)

    y = jax.lax.conv_general_dilated(inputs,
                                     kernel,
                                     window_strides=self.strides,
                                     padding=self.padding,
                                     lhs_dilation=(1,) * len(self.kernel_size),
                                     rhs_dilation=(1,) * len(self.kernel_size),
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                     feature_group_count=inputs.shape[-1])

    if self.use_bias:
      bias = self.param("bias", self.bias_init, (features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias

    return y


def hard_sigmoid(x):
  return jax.nn.relu6(x + 3) / 6


def hard_swish(x):
  return x * hard_sigmoid(x)


def _depth(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


class SEBlock(nn.Module):
  se_ratio: Optional[float] = 0.25
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    input_filters = inputs.shape[-1]

    x = global_average_pooling(inputs)
    x = x.reshape(-1, 1, 1, input_filters)
    x = nn.Conv(features=_depth(input_filters * self.se_ratio),
                kernel_size=(1, 1),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
    x = nn.relu(x)
    x = nn.Conv(features=input_filters,
                kernel_size=(1, 1),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
    x = hard_sigmoid(x)
    return inputs * x


class ResidualInvertedBottleneck(nn.Module):
  expansion: float
  filters: int
  kernel_size: int
  stride: int
  se_ratio: Optional[float] = 0.25
  activation: Callable = nn.relu
  batch_norm: bool = False

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    input_filters = x.shape[-1]

    # Expand (block_id controls this block in the keras implementation).
    x = nn.Conv(_depth(input_filters * self.expansion),
                kernel_size=(1, 1),
                use_bias=False)(x)
    if self.batch_norm:
      x = nn.BatchNorm(epsilon=1e-3, momentum=0.999)(x)
    x = self.activation(x)

    if self.stride == 2:
      x = zero_pad_2d(correct_pad(x, self.kernel_size))(x)
    x = DepthwiseConv2D(kernel_size=(self.kernel_size, self.kernel_size),
                        strides=(self.stride, self.stride),
                        padding="same" if self.stride == 1 else "valid",
                        use_bias=False)(x)
    if self.batch_norm:
      x = nn.BatchNorm(epsilon=1e-3, momentum=0.999)(x)
    x = self.activation(x)

    if self.se_ratio:
      x = SEBlock(self.se_ratio)(x)

    x = nn.Conv(self.filters, kernel_size=(1, 1), use_bias=False)(x)
    if self.batch_norm:
      x = nn.BatchNorm(epsilon=1e-3, momentum=0.999)(x)

    if self.stride == 1 and input_filters == self.filters:
      x = inputs + x
    return x


def _get_small_args(kernel, activation, se_ratio, alpha):
  depth = lambda d: _depth(d * alpha)
  # yapf: disable
  args = [
  #    expansion filters    kernel_size stride se_ratio  activation
      [1,        depth(16), 3,          2,     se_ratio, nn.relu],
      [72 / 16,  depth(24), 3,          2,     None,     nn.relu],
      [88 / 24,  depth(24), 3,          1,     None,     nn.relu],
      [4,        depth(40), kernel,     2,     se_ratio, activation],
      [6,        depth(40), kernel,     1,     se_ratio, activation],
      [6,        depth(40), kernel,     1,     se_ratio, activation],
      [3,        depth(48), kernel,     1,     se_ratio, activation],
      [3,        depth(48), kernel,     1,     se_ratio, activation],
      [6,        depth(96), kernel,     2,     se_ratio, activation],
      [6,        depth(96), kernel,     1,     se_ratio, activation],
      [6,        depth(96), kernel,     1,     se_ratio, activation],
  ]
  # yapf: enable
  return args


def _get_large_args(kernel, activation, se_ratio, alpha):
  depth = lambda d: _depth(d * alpha)
  # yapf: disable
  args = [
  #    expansion filters     kernel_size stride se_ratio  activation
      [1,        depth(16),  3,          1,     None,     relu],
      [4,        depth(24),  3,          2,     None,     relu],
      [3,        depth(24),  3,          1,     None,     relu],
      [3,        depth(40),  kernel,     2,     se_ratio, relu],
      [3,        depth(40),  kernel,     1,     se_ratio, relu],
      [3,        depth(40),  kernel,     1,     se_ratio, relu],
      [6,        depth(80),  3,          2,     None,     activation],
      [2.5,      depth(80),  3,          1,     None,     activation],
      [2.3,      depth(80),  3,          1,     None,     activation],
      [2.3,      depth(80),  3,          1,     None,     activation],
      [6,        depth(112), 3,          1,     se_ratio, activation],
      [6,        depth(112), 3,          1,     se_ratio, activation],
      [6,        depth(160), kernel,     2,     se_ratio, activation],
      [6,        depth(160), kernel,     1,     se_ratio, activation],
      [6,        depth(160), kernel,     1,     se_ratio, activation],
  ]
  # yapf: enable
  return args


class MobileNetV3(nn.Module):
  last_point_features: int = 1024
  alpha: float = 1.0
  classes: int = 1000
  minimalistic: bool = False
  large: bool = False
  classifier_activation: Callable = nn.log_softmax
  batch_norm: bool = False  # TODO: Remove

  @nn.compact
  def __call__(self, x):

    if self.minimalistic:
      kernel = 3
      activation = relu
      se_ratio = None
    else:
      kernel = 5
      activation = hard_swish
      se_ratio = 0.25

    # Input processing (shared between small and large variants).
    x = x / 255
    x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2),
                use_bias=False)(x)
    x = activation(x)

    # Main network
    get_args = _get_large_args if self.large else _get_small_args
    for args in get_args(kernel, activation, se_ratio, self.alpha):
      x = ResidualInvertedBottleneck(*args, batch_norm=self.batch_norm)(x)

    # Last stages (shared between small and large variants).
    x = nn.Conv(features=_depth(x.shape[-1] * 6),
                kernel_size=(1, 1),
                use_bias=False)(x)
    if self.batch_norm:
      x = nn.BatchNorm(epsilon=1e-3, momentum=0.999)(x)
    x = activation(x)

    if self.alpha > 1.0:
      last_point_features = _depth(self.last_point_features * self.alpha)
    else:
      last_point_features = self.last_point_features
    x = nn.Conv(features=last_point_features, kernel_size=(1, 1),
                use_bias=True)(x)
    x = activation(x)

    x = global_average_pooling(x)
    x = x.reshape((x.shape[0], 1, 1, last_point_features))

    x = nn.Conv(features=self.classes, kernel_size=(1, 1))(x)
    x = flatten(x)
    # x = self.classifier_activation(x)

    return x


@iree.jax.jit
def cross_entropy(variables, batch):
  inputs, labels = batch
  logits = ResidualInvertedBottleneck(1, _depth(16), 3, 1).apply(variables, inputs)
  logits = flatten(logits)
  return -jnp.mean(jnp.sum(logits * labels, axis=1))


@iree.jax.jit
def update(optimizer, batch):
  loss, gradient = jax.value_and_grad(cross_entropy)(optimizer.target, batch)
  optimizer = optimizer.apply_gradient(gradient)
  return optimizer, loss


def normal(shape, dtype=np.float32):
  return np.random.normal(0, 1, shape).astype(dtype)


def numerically_verify(jitted_func, *args, **kwargs):
  # if not isinstance(jitted_func, iree.jax._JittedFunction):
  #   print("adding jit decorator")
  #   jitted_func = iree.jax.jit(jitted_func)

  # Test that the base function first to ensure it doesn't error.
  print("getting expected results")
  expected_results = jitted_func._function(*args, **kwargs)
  expected_values, expected_tree = jax.tree_flatten(expected_results)

  print("getting compiled results")
  compiled_results = jitted_func(*args, **kwargs)
  compiled_values, compiled_tree = jax.tree_flatten(compiled_results)

  assert expected_tree == compiled_tree

  absolute_differences = []

  for compiled, expected in zip(compiled_values, expected_values):
    absolute_differences.append(np.abs(compiled - expected))
    np.testing.assert_allclose(compiled, expected, 1e-4, 1e-4)

  return compiled_results, absolute_differences


def main():
  key = jax.random.PRNGKey(0)
  images = normal((1, 16, 16, 16))
  # images = normal((1, 32, 32, 3))
  labels = jnp.array([1], dtype=np.int32)
  batch = [images, labels]

  print("initializing model")
  variables = ResidualInvertedBottleneck(1, _depth(16), 3, 1).init(key, images)
  # variables = MobileNetV3().init(key, images)

  print("running cross_entropy")
  numerically_verify(cross_entropy, variables, batch)

  optimizer = flax.optim.GradientDescent(1e-3).create(variables)
  print("running update")
  flat_args, _ = jax.tree_flatten((optimizer, batch))
  print("update complete")
  print([(arg.shape, arg.dtype) for arg in flat_args])
  print("verifying")
  numerically_verify(update, optimizer, batch)

if __name__ == "__main__":
  main()
