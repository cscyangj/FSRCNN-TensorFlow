"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
from math import ceil
import io
from random import randrange, shuffle

import tensorflow as tf
from PIL import Image
import numpy as np
from multiprocessing import Pool, Lock, active_children

FLAGS = tf.app.flags.FLAGS

downsample = True

def preprocess(path, scale=3, distort=False):
  """
  Preprocess single image file
    (1) Read original image
    (2) Downsample by scale factor
    (3) Normalize
  """
  try:
    from wand.image import Image
  except:
    from PIL import Image
    image = Image.open(path).convert('L')
    (width, height) = image.size

    if downsample:
        (width, height) = image.size
        image = image.crop((0, 0, width - width % scale, height - height % scale))

        (width, height) = image.size
        label_ = np.fromstring(image.tobytes(), dtype=np.uint8).reshape((height, width))

        new_width, new_height = int(width / scale), int(height / scale)
        scaled_image = image.resize((new_width, new_height), Image.BICUBIC)
        image.close()

        (width, height) = scaled_image.size
        if randrange(3) == 0 and distort==True:
            buf = io.BytesIO()
            scaled_image.convert('RGB').save(buf, "JPEG", quality=randrange(80, 95, 5))
            buf.seek(0)
            scaled_image = Image.open(buf).convert('L')

        input_ = np.fromstring(scaled_image.tobytes(), dtype=np.uint8).reshape((height, width))
    else:
        input_ = np.fromstring(image.tobytes(), dtype=np.uint8).reshape(height, width)
        scaled_image = image.resize((width * scale, height * scale), Image.BICUBIC)
        (width, height) = scaled_image.size
        label_ = np.fromstring(scaled_image.tobytes(), dtype=np.uint8).reshape(height, width)
  else:
    with Image(filename=path) as img:
        img.alpha_channel = False
        if downsample:
            img.crop(width = img.width - img.width % scale, height = img.height - img.height % scale)
            label_ = np.fromstring(img.make_blob('gray'), dtype=np.uint8).reshape(img.height, img.width)
            img.resize(width = img.width // scale, height = img.height // scale, filter = "lanczos2", blur=0.9)
            input_ = np.fromstring(img.make_blob('gray'), dtype=np.uint8).reshape(img.height, img.width)
            if randrange(3) == 0 and distort==True:
                img.compression_quality = randrange(80, 95, 5)
                img.format = 'jpeg'
                jpeg_bin = img.make_blob()
                with Image(blob=jpeg_bin) as jpeg:
                    input_ = np.fromstring(jpeg.make_blob('gray'), dtype=np.uint8).reshape(jpeg.height, jpeg.width)
        else:
            input_ = np.fromstring(img.make_blob('gray'), dtype=np.uint8).reshape(img.height, img.width)
            img.resize(width = img.width * scale, height = img.height * scale, filter = "catrom")
            label_ = np.fromstring(img.make_blob('gray'), dtype=np.uint8).reshape(img.height, img.width)

  return input_ / 255, label_ / 255

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.train:
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    shuffle(data)
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))

  return data

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def train_input_worker(args):
  image_data, config = args
  image_size, label_size, stride, scale, padding, distort = config

  single_input_sequence, single_label_sequence = [], []

  input_, label_ = preprocess(image_data, scale, distort=distort)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  for x in range(0, h - image_size + 1, stride):
    for y in range(0, w - image_size + 1, stride):
      sub_input = input_[x : x + image_size, y : y + image_size]
      x_loc, y_loc = x + padding, y + padding
      sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])
      
      single_input_sequence.append(sub_input)
      single_label_sequence.append(sub_label)

  return [single_input_sequence, single_label_sequence]


def thread_train_setup(config):
  """
  Spawns |config.threads| worker processes to pre-process the data

  This has not been extensively tested so use at your own risk.
  Also this is technically multiprocessing not threading, I just say thread
  because it's shorter to type.
  """
  if downsample == False:
    import sys
    sys.exit()

  sess = config.sess

  # Load data path
  data = prepare_data(sess, dataset=config.data_dir)

  # Initialize multiprocessing pool with # of processes = config.threads
  pool = Pool(config.threads)

  # Distribute |images_per_thread| images across each worker process
  config_values = [config.image_size, config.label_size, config.stride, config.scale, config.radius, config.distort]
  images_per_thread = len(data) // config.threads
  workers = []
  for thread in range(config.threads):
    args_list = [(data[i], config_values) for i in range(thread * images_per_thread, (thread + 1) * images_per_thread)]
    worker = pool.map_async(train_input_worker, args_list)
    workers.append(worker)
  print("{} worker processes created".format(config.threads))

  pool.close()

  results = []
  for i in range(len(workers)):
    print("Waiting for worker process {}".format(i))
    results.extend(workers[i].get(timeout=240))
    print("Worker process {} done".format(i))

  print("All worker processes done!")

  sub_input_sequence, sub_label_sequence = [], []

  for image in range(len(results)):
    single_input_sequence, single_label_sequence = results[image]
    sub_input_sequence.extend(single_input_sequence)
    sub_label_sequence.extend(single_label_sequence)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  return (arrdata, arrlabel)

def train_input_setup(config):
  """
  Read image files, make their sub-images, and save them as a h5 file format.
  """
  if downsample == False:
    import sys
    sys.exit()

  sess = config.sess
  image_size, label_size, stride, scale, padding = config.image_size, config.label_size, config.stride, config.scale, config.radius

  # Load data path
  data = prepare_data(sess, dataset=config.data_dir)

  sub_input_sequence, sub_label_sequence = [], []

  for i in range(len(data)):
    input_, label_ = preprocess(data[i], scale, distort=config.distort)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    for x in range(0, h - image_size + 1, stride):
      for y in range(0, w - image_size + 1, stride):
        sub_input = input_[x : x + image_size, y : y + image_size]
        x_loc, y_loc = x + padding, y + padding
        sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

        sub_input = sub_input.reshape([image_size, image_size, 1])
        sub_label = sub_label.reshape([label_size, label_size, 1])
        
        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  return (arrdata, arrlabel)

def test_input_setup(config):
  """
  Read image files, make their sub-images, and save them as a h5 file format.
  """
  sess = config.sess
  image_size, label_size, stride, scale, padding = config.image_size, config.label_size, config.stride, config.scale, config.radius

  # Load data path
  data = prepare_data(sess, dataset="Test")

  sub_input_sequence, sub_label_sequence = [], []

  pic_index = 2 # Index of image based on lexicographic order in data folder
  input_, label_ = preprocess(data[pic_index], config.scale)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  nx, ny = 0, 0
  for x in range(0, h - image_size + 1, stride):
    nx += 1
    ny = 0
    for y in range(0, w - image_size + 1, stride):
      ny += 1
      sub_input = input_[x : x + image_size, y : y + image_size]
      x_loc, y_loc = x + padding, y + padding
      sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])

      sub_input_sequence.append(sub_input)
      sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  return (arrdata, arrlabel, nx, ny)

# You can ignore, I just wanted to see how much space all the parameters would take up
def save_params(sess, weights, biases, alphas, params):
  param_dir = "params/"

  if not os.path.exists(param_dir):
    os.makedirs(param_dir)

  h = open(param_dir + "weights{}.txt".format('_'.join(str(i) for i in params)), 'w')

  for layer in weights:
    h.write("{} =\n  [".format(layer))
    layer_weights = sess.run(weights[layer])
    sep = False

    for filter_x in range(len(layer_weights)):
      for filter_y in range(len(layer_weights[filter_x])):
        filter_weights = layer_weights[filter_x][filter_y]
        for input_channel in range(len(filter_weights)):
          for output_channel in range(len(filter_weights[input_channel])):
            val = filter_weights[input_channel][output_channel]
            if sep:
                h.write(', ')
            h.write("{}".format(val))
            sep = True
          h.write("\n  ")

    h.write("]\n\n")

  for layer, tensor in list(biases.items()) + list(alphas.items()):
    h.write("{} = [".format(layer))
    vals = sess.run(tensor)
    h.write(",".join(map(str, vals)))
    h.write("]\n")

  h.close()

def merge(images, size):
  """
  Merges sub-images back into original image size
  """
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], size[2]))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img

def array_image_save(array, image_path):
  """
  Converts np array to image and saves it
  """
  image = Image.fromarray(array)
  if image.mode != 'RGB':
    image = image.convert('RGB')
  image.save(image_path)
  print("Saved image: {}".format(image_path))

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, l=False, mean_metric=True, size=3):
    window = tf.fill([size, size, 1, 1], 1.0 / size**2)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2
    if cs_map:
        value = ((2.0*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1) if l==True else 1.0,
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, level=5, size=3):
    weight = tf.constant([[1.0], [0.5, 0.5], None, None, [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]][level-1], dtype=tf.float32)
    window = _tf_fspecial_gauss(3, 0.5)
    ml = []
    mcs = []
    for i in range(level):
        l_map, cs_map = tf_ssim(img1, img2, cs_map=True, l=(i==level-1), mean_metric=False, size=size)
        ml.append(tf.reduce_mean(l_map))
        mcs.append(tf.reduce_mean(cs_map))
        #img1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
        #img2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='SAME')
        size = img1.shape[1].value // 2 + 1
        img1 = tf.image.resize_bilinear(img1, [size, size])
        img2 = tf.image.resize_bilinear(img2, [size, size])

    # list to tensor of dim D+1
    ml = tf.stack(ml, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(tf.pow(mcs, weight)) * tf.pow(ml[level-1], weight[level-1])

    return value

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, channels):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = 2 * factor
    weights = np.zeros((filter_size, filter_size, channels, channels), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(channels):
        weights[:, :, i, i] = upsample_kernel
    return weights
