import numpy
import math
import tensorflow
from matplotlib import pyplot
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(5, 50)),
    iaa.Crop(percent=(0, 0.3))
])

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')

x_aug_before = 0.1 * x_train

x_aug = seq.augment_images(x_aug_before)

# for xab in range(len(x_aug_before)):
#     images = x_train[xab]
#     images_aug = seq(images=images)
#     x_aug.append(images_aug)

x_aug = numpy.array(x_aug)

print(x_aug[5])

if x_aug[5].max() > 1:
    x_aug[5] = x_aug[5] / 255

pyplot.imshow(x_aug[5], cmap=pyplot.get_cmap)
pyplot.show()
