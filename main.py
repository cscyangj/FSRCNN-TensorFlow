from model import Model

import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("arch", 1, "1 - FSRCNN, 2 - ESPCN, 3 - LapSRN [1]")
flags.DEFINE_boolean("fast", False, "Use the fast model (FSRCNN-s) [False]")
flags.DEFINE_integer("epoch", 10, "Number of epochs [10]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of the adam optimizer [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color [1]")
flags.DEFINE_integer("scale", 2, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("radius", 2, "Max radius of the deconvolution input tensor [2]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("output_dir", "result", "Name of test output directory [result]")
flags.DEFINE_string("data_dir", "Train", "Name of data directory to train on [FastTrain]")
flags.DEFINE_boolean("train", True, "True for training, false for testing [True]")
flags.DEFINE_boolean("adversarial", False, "Adversarial training [False]")
flags.DEFINE_integer("threads", 1, "Number of processes to pre-process data with [1]")
flags.DEFINE_boolean("distort", False, "Distort some images with JPEG compression artifacts after downscaling [False]")
flags.DEFINE_boolean("params", False, "Save weight and bias parameters [False]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.fast:
    FLAGS.checkpoint_dir = 'fast_{}'.format(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)


  with tf.Session() as sess:
    model = Model(sess, config=FLAGS)
    model.run()
    
if __name__ == '__main__':
  tf.app.run()
