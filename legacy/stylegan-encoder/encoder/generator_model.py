
import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial
# from tensorflow import set_random_seed
# set_random_seed(2)

def create_stub_mapping(batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

def create_stub_synthesis(batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

def create_variable_for_generator_1d(batch_size):
    return tf.Variable(name='learnable_latents',
                       initial_value=tf.random_normal([batch_size, 512]),
                       dtype='float32')

def create_variable_for_generator_2d(batch_size):
    return tf.Variable(name='learnable_latents',
                       initial_value=tf.random_normal([batch_size, 18, 512]),
                       dtype='float32', trainable=False)

class Generator:
    def __init__(self, model, batch_size, randomize_noise=False, z_space=True, w_full_space=False):
        self.batch_size = batch_size
        self.z_space = z_space
        self.w_full_space = w_full_space
        if z_space:
            #self.initial_latents = np.random.RandomState(6).randn(self.batch_size, 512)
            self.initial_latents = np.random.randn(self.batch_size, 512)
            self.initial_labels = np.zeros((self.batch_size, 0))
            self.initial_latents_variable = create_variable_for_generator_1d(self.batch_size)
            self.initial_labels_variable = create_stub_mapping(self.batch_size)

            self.dlatents = model.components.mapping.get_output_for(self.initial_latents_variable, None)
            #print(self.dlatents.get_shape())
            self.generator_output = model.components.synthesis.get_output_for(self.dlatents, use_noise=True)

        elif w_full_space:
            #self.initial_latents = np.random.randn(self.batch_size, 18, 512)
            self.initial_latents = np.zeros((self.batch_size, 18, 512))
            self.initial_latents_variable = create_variable_for_generator_2d(self.batch_size)
            self.generator_output = model.components.synthesis.get_output_for(self.initial_latents_variable,
                                                                              randomize_noise=randomize_noise,
                                                                              structure='fixed')

        else:
            self.initial_latents = np.zeros((self.batch_size, 512))
            self.initial_latents_variable = create_variable_for_generator_1d(self.batch_size)
            initial_latents_broadcast = tf.tile(self.initial_latents_variable[:, np.newaxis], [1, 18, 1])
            self.generator_output = model.components.synthesis.get_output_for(initial_latents_broadcast,
                                                                              randomize_noise=randomize_noise,
                                                                              structure='fixed')

        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        self.set_latents(self.initial_latents)
        #print(self.generator_output.get_shape())
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        #print(self.generated_image.get_shape())
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)
        #print(self.generated_image_uint8.get_shape())

    def reset_latents(self):
        if self.z_space:
            initial_latents = np.random.randn(self.batch_size, 512)
        elif self.w_full_space:
            initial_latents = np.zeros((self.batch_size, 18, 512))
        else:
            initial_latents = np.zeros((self.batch_size, 512))

        self.set_latents(initial_latents)

    def set_latents(self, latents):
        if self.z_space:
            assert (latents.shape == (self.batch_size, 512))
        elif self.w_full_space:
            assert (latents.shape == (self.batch_size, 18, 512))
        else:
            assert (latents.shape == (self.batch_size, 512))
        self.sess.run(tf.assign(self.initial_latents_variable, latents))

    def get_latents(self):
        return self.sess.run(self.initial_latents_variable)

    def generate_images(self, latents=None):
        if latents:
            self.set_latents(latents)
        return self.sess.run(self.generated_image_uint8)