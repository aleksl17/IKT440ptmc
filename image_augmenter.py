import tensorflow
import imgaug
import math

# Global Variables
augmentation_percentage = 30

# Import and manipulate dataset
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.cifar10.load_data()
a_p = math.floor((augmentation_percentage / 100) * len(x_train))
x_train_aug_before = x_train[0:a_p]
x_train_aug_before = x_train_aug_before.astype("float32")

# Augmentation variables
seq = imgaug.augmenters.Sequential([
    imgaug.augmenters.Affine(rotate=(-25, 25)),
    imgaug.augmenters.SaltAndPepper(p=(0, 0.1)),
    imgaug.augmenters.Crop(percent=(0, 0.25))
])

# Image augmentation
x_aug = seq.augment_images(x_train_aug_before)

# print(x_aug)
