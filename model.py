from utils import (
  read_data, 
  thread_train_setup,
  train_input_setup,
  test_input_setup,
  save_params,
  merge,
  array_image_save
)

import time
import os

import numpy as np
import tensorflow as tf

from PIL import Image
import pdb

# Based on http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
class Model(object):
  
  def __init__(self, sess, config):
    self.sess = sess
    self.arch = config.arch
    self.fast = config.fast
    self.train = config.train
    self.c_dim = config.c_dim
    self.is_grayscale = (self.c_dim == 1)
    self.epoch = config.epoch
    self.scale = config.scale
    self.radius = config.radius
    self.batch_size = config.batch_size
    self.learning_rate = config.learning_rate
    self.threads = config.threads
    self.distort = config.distort
    self.params = config.params

    self.padding = self.radius * 2
    # Different image/label sub-sizes for different scaling factors x2, x3, x4
    scale_factors = [[10 + self.padding, 20], [7 + self.padding, 21], [6 + self.padding, 24]]
    self.image_size, self.label_size = scale_factors[self.scale - 2]
    # Testing uses different strides to ensure sub-images line up correctly
    if not self.train:
      self.stride = [10, 7, 6][self.scale - 2]
    else:
      self.stride = [6, 4, 4][self.scale - 2]

    self.checkpoint_dir = config.checkpoint_dir
    self.output_dir = config.output_dir
    self.data_dir = config.data_dir
    self.init_model()


  def init_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    # Batch size differs in training vs testing
    self.batch = tf.placeholder(tf.int32, shape=[], name='batch')
 
    if self.arch == 1:
        from FSRCNN import FSRCNN
        self.model = FSRCNN(self)
    elif self.arch == 2:
        from ESPCN import ESPCN
        self.model = ESPCN(self)
    elif self.arch == 3:
        from LapSRN import LapSRN
        self.model = LapSRN(self)

    self.pred = self.model.model()

    model_dir = "%s_%s_%s_%s" % (self.model.name.lower(), self.label_size, '-'.join(str(i) for i in self.model.model_params), "r"+str(self.radius))
    self.model_dir = os.path.join(self.checkpoint_dir, model_dir)

    self.loss = self.model.loss(self.labels, self.pred)

    self.saver = tf.train.Saver()

  def run(self):
    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    tf.global_variables_initializer().run()

    if self.load():
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if self.params:
      save_params(self.sess, self.model.weights, self.model.biases, self.model.alphas, self.model.model_params)
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

        # Save every 500 steps
        if counter % 500 == 0:
          self.save(counter)

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

  def save(self, step):
    model_name = self.model.name + ".model"

    if not os.path.exists(self.model_dir):
        os.makedirs(self.model_dir)

    self.saver.save(self.sess,
                    os.path.join(self.model_dir, model_name),
                    global_step=step)

  def load(self):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(self.model_dir, ckpt_name))
        return True
    else:
        return False
