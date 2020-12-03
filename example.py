from tensorflow.keras.datasets import cifar10
import numpy
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from sklearn.decomposition import PCA

from imgaug import augmenters as iaa
import imgaug as ia

# matplotlib inline
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train[1])
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')


def show_nine(imgs):
    if imgs.max() > 1:
        imgs = imgs / 255.

    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(imgs[i], cmap=plt.get_cmap())
    plt.show()


def show_img_augs(imgs, imgaug_seq):
    ia.seed(1)
    imgs_aug = seq.augment_images(imgs)
    show_nine(imgs_aug)


seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.SaltAndPepper(p=(0, 0.1)),
    iaa.Crop(percent=(0, 0.25))
])

imgs = X_train[0:9]

print(imgs[1])

show_img_augs(imgs, seq)

# imgs = X_train[0:9]
# show_nine(imgs)
