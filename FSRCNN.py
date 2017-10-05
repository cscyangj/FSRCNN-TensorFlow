import tensorflow as tf

class FSRCNN(object):

  def __init__(self, config):
    self.name = "FSRCNN"
    # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (d, s, m) in paper
    model_params = [[56, 12, 4], [32, 8, 1]]
    self.model_params = model_params[config.fast]
    self.scale = config.scale
    self.radius = config.radius
    self.padding = config.padding
    self.images = config.images
    self.batch = config.batch
    self.label_size = config.label_size
    self.c_dim = config.c_dim
    self.weights, self.biases, self.alphas = {}, {}, {}

  def model(self):

    d, s, m = self.model_params

    # Feature Extraction
    size = self.padding + 1
    self.weights['w1'] = tf.get_variable('w1', initializer=tf.random_normal([size, size, 1, d], stddev=0.0378, dtype=tf.float32))
    self.biases['b1'] = tf.get_variable('b1', initializer=tf.zeros([d]))
    conv = self.prelu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'], 1)

    # Shrinking
    if self.model_params[1] > 0:
      self.weights['w2'] = tf.get_variable('w2', initializer=tf.random_normal([1, 1, d, s], stddev=0.3536, dtype=tf.float32))
      self.biases['b2'] = tf.get_variable('b2', initializer=tf.zeros([s]))
      conv = self.prelu(tf.nn.conv2d(conv, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'], 2)
    else:
      s = d

    # Mapping (# mapping layers = m)
    for i in range(3, m + 3):
      weights = tf.get_variable('w{}'.format(i), initializer=tf.random_normal([3, 3, s, s], stddev=0.1179, dtype=tf.float32))
      biases = tf.get_variable('b{}'.format(i), initializer=tf.zeros([s]))
      self.weights['w{}'.format(i)], self.biases['b{}'.format(i)] = weights, biases
      conv = self.prelu(tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='SAME') + biases, i)

    # Expanding
    if self.model_params[1] > 0:
      expand_weights = tf.get_variable('w{}'.format(m + 3), initializer=tf.random_normal([1, 1, s, d], stddev=0.189, dtype=tf.float32))
      expand_biases = tf.get_variable('b{}'.format(m + 3), initializer=tf.zeros([d]))
      self.weights['w{}'.format(m + 3)], self.biases['b{}'.format(m + 3)] = expand_weights, expand_biases
      conv = self.prelu(tf.nn.conv2d(conv, expand_weights, strides=[1,1,1,1], padding='SAME') + expand_biases, m + 3)

    # Deconvolution
    deconv_size = self.radius * self.scale * 2 + 1
    deconv_weights = tf.get_variable('w{}'.format(m + 4), initializer=tf.random_normal([deconv_size, deconv_size, 1, d], stddev=0.0001, dtype=tf.float32))
    deconv_biases = tf.get_variable('b{}'.format(m + 4), initializer=tf.zeros([1]))
    self.weights['w{}'.format(m + 4)], self.biases['b{}'.format(m + 4)] = deconv_weights, deconv_biases
    deconv_output = [self.batch, self.label_size, self.label_size, self.c_dim]
    deconv_stride = [1,  self.scale, self.scale, 1]
    deconv = tf.nn.conv2d_transpose(conv, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + deconv_biases

    return deconv

  def prelu(self, _x, i):
    """
    PreLU tensorflow implementation
    """
    alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    self.alphas['alpha{}'.format(i)] = alphas

    return tf.nn.relu(_x) - alphas * tf.nn.relu(-_x)


