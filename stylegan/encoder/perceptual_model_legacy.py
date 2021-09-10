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
    def __init__(self, img_size, layer=9, batch_size=1, sess=None, lr=0.1):
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
        self.generated_img_features = None
        self.image_features = None
        self.loaded_image = None

    def build_perceptual_model(self, generated_image_tensor):
        print('##########################################################')
        print("generated_image_tensor shape: " + str(generated_image_tensor.shape))
        print('##########################################################')
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_image_tensor = tf.slice(generated_image_tensor, [0, 112, 112, 0], [1, 800, 800, 3])
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
                                                                  (self.img_size, self.img_size), method=1))

        print('##########################################################')
        print("generated_image shape: " + str(generated_image.shape))
        print('##########################################################')
        self.generated_img_features = self.perceptual_model(generated_image)

        self.ref_img_features = tf.get_variable('ref_img_features', shape=self.generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight = tf.get_variable('features_weight', shape=self.generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.sess.run([self.features_weight.initializer])

        self.loss = tf.losses.mean_squared_error(self.features_weight * self.ref_img_features,
                                                 self.features_weight * self.generated_img_features) / 82890.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        self.loaded_image = load_images(images_list, self.img_size)
        self.image_features = self.perceptual_model.predict_on_batch(self.loaded_image)

        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            self.image_features = np.vstack([self.image_features, np.zeros(empty_features_shape)])

        print(weight_mask.shape)
        print(self.image_features.shape)
        self.sess.run(tf.assign(self.features_weight, weight_mask))
        self.sess.run(tf.assign(self.ref_img_features, self.image_features))

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1.):
        self.normal_prior =  -1 * self.alpha * tf.reduce_mean(tfd.Normal(loc=0.0, scale=1.0).log_prob(vars_to_optimize))
        hybrid_loss = self.loss + self.normal_prior
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        min_op = optimizer.minimize(hybrid_loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        #grads = optimizer.compute_gradients(hybrid_loss, var_list=[vars_to_optimize])
        #print(tf.global_variables())
        for _ in range(iterations):
            #print(self.sess.run([grad[0] for grad in grads]))
            _, loss, normal_prior = self.sess.run([min_op, self.loss, self.normal_prior])
            #_, loss = self.sess.run([min_op, self.loss])
            yield (loss, normal_prior)
            #yield loss