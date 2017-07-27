import os
import sys
import glob
import numpy as np
from PIL import Image
import pdb
from random import randint

# Artifically expands the dataset by a factor of 19 by scaling and then rotating every image
def main():
  if len(sys.argv) == 2:
    data = prepare_data(sys.argv[1])
  else:
    print("Missing argument: You must specify a folder with images to expand")
    return

  for i in range(len(data)):
    scale(data[i], randint(0, 3))
    rotate(data[i], randint(0, 2))
    c = randint(0, 1)
    if c==0:
      flip_left_right(data[i])
    else:
      flip_top_bottom(data[i])

def prepare_data(dataset):
  filenames = os.listdir(dataset)
  data_dir = os.path.join(os.getcwd(), dataset)
  data = glob.glob(os.path.join(data_dir, "*.bmp"))

  return data

def scale(file, scale):
  image = Image.open(file)
  width, height = image.size

  scales = [0.9, 0.8, 0.7, 0.6]
  new_width, new_height = int(width * scales[scale]), int(height * scales[scale])
  new_image = image.resize((new_width, new_height), Image.BICUBIC)
  new_path = '{}-{}.bmp'.format(file[:-4], scales[scale])
  new_image.save(new_path)
  """
  rotate(new_path, randint(0, 2))
  c = randint(0, 1)
  if c==0:
    flip_left_right(new_path)
  else:
    flip_top_bottom(new_path)
  """

def rotate(file, rotation):
  image = Image.open(file)

  rotations = [90, 180, 270]
  new_image = image.rotate(rotations[rotation], expand=True)
  new_path = '{}-{}.bmp'.format(file[:-4], rotations[rotation])
  new_image.save(new_path)

def flip_left_right(file):
  image = Image.open(file)
  new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
  new_path = '{}-{}.bmp'.format(file[:-4], "lr")
  new_image.save(new_path)

def flip_top_bottom(file):
  image = Image.open(file)
  new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
  new_path = '{}-{}.bmp'.format(file[:-4], "tb")
  new_image.save(new_path)

if __name__ == '__main__':
  main()
