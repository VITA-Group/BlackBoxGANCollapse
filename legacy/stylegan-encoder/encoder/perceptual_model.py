import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
import operator
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def cropND(img, bounding):
    print(img.shape)
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    print(start)
    end = tuple(map(operator.add, start, bounding))
    print(end)
    slices = tuple(map(slice, start, end))
    print(slices)
    return img[slices]

def pil_resize(img, target_size):
    width_height_tuple = (target_size[1], target_size[0])
    resample = pil_image.NEAREST
    img = img.resize(width_height_tuple, resample)
    return img

def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path)
        img = np.asarray(img)
        img = cropND(img, (800, 800))
        img = pil_image.fromarray(img)
        img = pil_resize(img, target_size=(img_size, img_size))
        w, h = img.size
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None
        self.normal_prior = None
        self.alpha = None
        self.image_features = None
        self.generated_img_features = None

    def build_perceptual_model(self, generated_image_tensor):
        print('##########################################################')
        print("generated_image_tensor shape: " + str(generated_image_tensor.shape))
        print('##########################################################')
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_image_tensor = tf.slice(generated_image_tensor, [0, 112, 112, 0], [self.batch_size, 800, 800, 3])
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
                                                                  (self.img_size, self.img_size), method=1))

        print('##########################################################')
        print("generated_image shape: " + str(generated_image.shape))
        print('##########################################################')
        self.generated_img_features = self.perceptual_model(generated_image)


    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_reference_images(self, loaded_image):
        # assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        # loaded_image = load_images(images_list, self.img_size)
        self.image_features = self.perceptual_model(loaded_image)

    def compute_feature_loss(self, weight_mask):
        return tf.losses.mean_squared_error(weight_mask * self.image_features,
                                                 weight_mask * self.generated_img_features) / 82890.0