from utils import (
  read_data, 
  thread_train_setup,
  train_input_setup,
  test_input_setup,
  save_params,
  merge,
  array_image_save,
  tf_ssim,
  tf_ms_ssim
)

import time
import os

import numpy as np
import tensorflow as tf

from PIL import Image
import pdb

# Based on http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
class FSRCNN(object):
  
  def __init__(self, sess, config):
    self.sess = sess
    self.fast = config.fast
    self.train = config.train
    self.c_dim = config.c_dim
    self.is_grayscale = (self.c_dim == 1)
    self.epoch = config.epoch
    self.scale = config.scale
    self.batch_size = config.batch_size
    self.learning_rate = config.learning_rate
    self.threads = config.threads
    self.distort = config.distort
    self.params = config.params

    # Different image/label sub-sizes for different scaling factors x2, x3, x4
    scale_factors = [[14, 20], [11, 21], [10, 24]]
    self.image_size, self.label_size = scale_factors[self.scale - 2]
    # Testing uses different strides to ensure sub-images line up correctly
    if not self.train:
      self.stride = [10, 7, 6][self.scale - 2]
    else:
      self.stride = [6, 4, 4][self.scale - 2]

    # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (d, s, m) in paper
    model_params = [[56, 12, 4], [32, 8, 1]]
    self.model_params = model_params[self.fast]

    self.deconv_radius = [3, 5, 7][self.scale - 2]
    
    self.checkpoint_dir = config.checkpoint_dir
    self.output_dir = config.output_dir
    self.data_dir = config.data_dir
    self.init_model()


  def init_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    # Batch size differs in training vs testing
    self.batch = tf.placeholder(tf.int32, shape=[], name='batch')
    self.weights, self.biases, self.alphas = {}, {}, {}
 
    self.pred = self.model()

    # Loss function (structural dissimilarity)
    ssim = tf_ms_ssim(self.labels, self.pred, level=2)
    self.loss = (1 - ssim) / 2

    self.saver = tf.train.Saver()

  def run(self):
    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    tf.global_variables_initializer().run()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if self.params:
      d, s, m = self.model_params
      save_params(self.sess, self.weights, self.biases, self.alphas, d, s, m)
    elif self.train:
      self.run_train()
    else:
      self.run_test()

  def run_train(self):
    start_time = time.time()
    print("Beginning training setup...")
    if self.threads == 1:
      train_input_setup(self)
    else:
      thread_train_setup(self)
    print("Training setup took {} seconds with {} threads".format(time.time() - start_time, self.threads))

    data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "train.h5")
    train_data, train_label = read_data(data_dir)
    print("Total setup time took {} seconds with {} threads".format(time.time() - start_time, self.threads))

    print("Training...")
    start_time = time.time()
    start_average, end_average, counter = 0, 0, 0

    for ep in range(self.epoch):
      # Run by batch images
      batch_idxs = len(train_data) // self.batch_size
      batch_average = 0
      for idx in range(0, batch_idxs):
        batch_images = train_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = train_label[idx * self.batch_size : (idx + 1) * self.batch_size]

        counter += 1
        _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels, self.batch: self.batch_size})
        batch_average += err

        if counter % 10 == 0:
          print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
            % ((ep+1), counter, time.time() - start_time, err))

        # Save every 200 steps
        if counter % 200 == 0:
          self.save(self.checkpoint_dir, counter)

      batch_average = float(batch_average) / batch_idxs
      if ep < (self.epoch * 0.2):
        start_average += batch_average
      elif ep >= (self.epoch * 0.8):
        end_average += batch_average

    # Compare loss of the first 20% and the last 20% epochs
    start_average = float(start_average) / (self.epoch * 0.2)
    end_average = float(end_average) / (self.epoch * 0.2)
    print("Start Average: [%.6f], End Average: [%.6f], Improved: [%.2f%%]" \
      % (start_average, end_average, 100 - (100*end_average/start_average)))

    # Linux desktop notification when training has been completed
    # title = "Training complete - FSRCNN"
    # notification = "{}-{}-{} done training after {} epochs".format(self.image_size, self.label_size, self.stride, self.epoch);
    # notify_command = 'notify-send "{}" "{}"'.format(title, notification)
    # os.system(notify_command)

  
  def run_test(self):
    nx, ny = test_input_setup(self)
    data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "test.h5")
    test_data, test_label = read_data(data_dir)

    print("Testing...")

    start_time = time.time()
    result = self.pred.eval({self.images: test_data, self.labels: test_label, self.batch: nx * ny})
    print("Took %.3f seconds" % (time.time() - start_time))

    result = merge(result, [nx, ny, self.c_dim])
    result = result.squeeze()
    image_path = os.path.join(os.getcwd(), self.output_dir)
    image_path = os.path.join(image_path, "test_image.png")

    array_image_save(result * 255, image_path)

  def model(self):

    d, s, m = self.model_params

    # Feature Extraction
    self.weights['w1'] = tf.get_variable('w1', initializer=tf.random_normal([5, 5, 1, d], stddev=0.0378, dtype=tf.float32))
    self.biases['b1'] = tf.get_variable('b1', initializer=tf.zeros([d]))
    conv = self.prelu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'], 1)

    # Shrinking
    if self.model_params[1] > 0:
      self.weights['w2'] = tf.get_variable('w2', initializer=tf.random_normal([1, 1, d, s], stddev=0.3536, dtype=tf.float32))
      self.biases['b2'] = tf.get_variable('b2', initializer=tf.zeros([s]))
      conv = self.prelu(tf.nn.conv2d(conv, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'], 2)

    # Mapping (# mapping layers = m)
    if s == 0:
      s = d
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
    deconv_size = self.deconv_radius * 2 + 1
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
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

  def save(self, checkpoint_dir, step):
    model_name = "FSRCNN.model"
    d, s, m = self.model_params
    model_dir = "%s_%s_%s-%s-%s" % ("fsrcnn", self.label_size, d, s, m)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    d, s, m = self.model_params
    model_dir = "%s_%s_%s-%s-%s" % ("fsrcnn", self.label_size, d, s, m)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
