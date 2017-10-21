import tensorflow as tf
import math
from utils import bilinear_upsample_weights

class LapSRN(object):

  def __init__(self, config):
    self.name = "LapSRN"
    self.model_params = [32, 2, 2]  #f, d, r in paper
    self.scale = config.scale
    self.radius = config.radius
    self.padding = config.padding
    self.images = config.images
    self.labels = config.labels
    self.batch = config.batch
    self.image_size = config.image_size - self.padding
    self.label_size = config.label_size
    self.c_dim = config.c_dim
    self.weights, self.biases, self.alphas = {}, {}, {}

  def model(self):
    d, m, r = self.model_params

    # Feature Extraction
    size = self.padding + 1
    self.weights['w1'] = tf.get_variable('w1', initializer=tf.random_normal([size, size, 1, d], stddev=0.0378, dtype=tf.float32))
    self.biases['b1'] = tf.get_variable('b1', initializer=tf.zeros([d]))
    conv = tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1']

    img, loss, reuse = tf.image.resize_image_with_crop_or_pad(self.images, self.image_size, self.image_size), 0.0, False
    for s in range(0, int(math.log(self.scale, 2))):
        with tf.variable_scope("recursive_block", reuse=reuse) as scope:
            features = conv
            for ri in range(r):
                # Mapping (# mapping layers = m)
                for i in range(3, m+3):
                  weights = tf.get_variable('w{}'.format(i), initializer=tf.random_normal([3, 3, d, d], stddev=0.1179, dtype=tf.float32))
                  biases = tf.get_variable('b{}'.format(i), initializer=tf.zeros([d]))
                  self.weights['w{}'.format(i)], self.biases['b{}'.format(i)] = weights, biases
                  conv = self.lrelu(conv, i)
                  conv = tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='SAME') + biases
                conv = tf.add(conv, features)
                scope.reuse_variables()

        with tf.variable_scope("deconvolution", reuse=reuse):
            # Deconvolution 1
            deconv_weights = tf.get_variable('w{}'.format(m+4), initializer=tf.random_normal([4, 4, d, d], stddev=0.0001, dtype=tf.float32))
            deconv_biases = tf.get_variable('b{}'.format(m+4), initializer=tf.zeros([d]))
            self.weights['w{}'.format(m+4)], self.biases['b{}'.format(m+4)] = deconv_weights, deconv_biases
            deconv_output = [self.batch, self.image_size * 2**(s+1), self.image_size * 2**(s+1), d]
            deconv_stride = [1, 2, 2, 1]
            conv = self.lrelu(conv, m+4)
            conv = tf.nn.conv2d_transpose(conv, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + deconv_biases

            # Predicting the sub-band residuals
            weights = tf.get_variable('w{}'.format(m+5), initializer=tf.random_normal([3, 3, d, 1], stddev=0.1179, dtype=tf.float32))
            biases = tf.get_variable('b{}'.format(m+5), initializer=tf.zeros([1]))
            self.weights['w{}'.format(m+5)], self.biases['b{}'.format(m+5)] = weights, biases
            res = tf.nn.conv2d(conv, weights, strides=[1,1,1,1], padding='SAME') + biases

        # Deconvolution 2
        deconv_weights = bilinear_upsample_weights(2, 1)
        deconv_output = [self.batch, self.image_size * 2**(s+1), self.image_size * 2**(s+1), 1]
        deconv_stride = [1, 2, 2, 1]
        img = tf.nn.conv2d_transpose(img, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + res

        if s < int(math.log(self.scale, 2)) - 1:
            label_size = self.image_size * 2**(s+1)
            labels = tf.image.resize_bicubic(self.labels, [label_size, label_size])
        else:
            labels = self.labels
        loss = loss + tf.reduce_mean(tf.sqrt(tf.square(img - labels) + 1e-6))

        reuse = True

    self.loss_sum = loss
    return img

  def lrelu(self, _x, i):
    """
    LreLU tensorflow implementation
    """
    return tf.nn.relu(_x) - 0.2 * tf.nn.relu(-_x)

  def loss(self, Y, X):
    return self.loss_sum
