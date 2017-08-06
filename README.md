# FSRCNN-TensorFlow
TensorFlow implementation of the Fast Super-Resolution Convolutional Neural Network (FSRCNN). This implements two models: FSRCNN which is more accurate but slower and FSRCNN-s which is faster but less accurate. Based on this [project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).

## Prerequisites
 * Python 3
 * TensorFlow-gpu >= 1.3
 * CUDA & cuDNN >= 6.0
 * h5py
 * Pillow

## Usage
For training: `python main.py`
<br>
For testing: `python main.py --train False`

To use FSCRNN-s instead of FSCRNN: `python main.py --fast True`

Can specify epochs, data directory, etc:
<br>
`python main.py --epoch 100 --data_dir Train`
<br>
Check `main.py` for all the possible flags

Also includes script `expand_data.py` which scales and rotates all the images in the specified training set to expand it

## Result

Original butterfly image:

![orig](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/original.png?raw=true)


Ewa_lanczos interpolated image:

![ewa_lanczos](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/ewa_lanczos.png?raw=true)


Super-resolved image:

![fsrcnn](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/fsrcnn.png?raw=true)

### More

![baby](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/baby.png?raw=true "original") ![baby-fsrcnn](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/baby-fsrcnn.png?raw=true "fsrcnn")


With --distort=True:

![woman](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/woman.png?raw=true "original") ![woman-fsrcnn](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/woman-fsrcnn.png?raw=true "fsrcnn")

## TODO

* Add RGB support (Increase each layer depth to 3)
* Speed up pre-processing for large datasets

## References

* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)

* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 

* [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 
