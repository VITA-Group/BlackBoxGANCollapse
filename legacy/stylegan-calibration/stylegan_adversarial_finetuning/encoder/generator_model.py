import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import backbones.modifiedResNet_v2 as modifiedResNet_v2
import tensorflow.contrib.slim as slim

import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial
# from tensorflow import set_random_seed
# set_random_seed(2)

def create_stub_mapping(batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

def create_variable_for_generator_1d(batch_size):
    return tf.Variable(name='learnable_latents',
                       initial_value=tf.random_normal([batch_size, 512]),
                       dtype='float32')

def build_InsightFace_model(images, reuse):
    #with tf.variable_scope('embd_extractor', reuse=False):
        #images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image')
        #net = images
    with tf.variable_scope('embd_extractor', reuse=reuse):
        arg_sc = modifiedResNet_v2.resnet_arg_scope(weight_decay=5e-4, batch_norm_decay=0.9)
        with slim.arg_scope(arg_sc):
            net, end_points = modifiedResNet_v2.resnet_v2_m_50(images, is_training=True, return_raw=True)
            net = slim.batch_norm(net, activation_fn=None, is_training=True)
            net = slim.dropout(net, keep_prob=0.4, is_training=False)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 512, normalizer_fn=None, activation_fn=None)
            embds = slim.batch_norm(net, scale=False, activation_fn=None, is_training=True)
        return embds

class Generator:
    def __init__(self, model, batch_size):
        self.batch_size = batch_size

        #self.initial_latents = np.random.RandomState(6).randn(self.batch_size, 512)
        self.initial_latents = np.random.randn(self.batch_size, 512)
        print(self.initial_latents.shape)
        collapsed = np.load('../sampling/monte_carlo_sampling_1m/neighbors/0.25/clustered_latents/13530_0.npy')
        self.initial_latents[0] = collapsed
        print(collapsed.shape)
        self.initial_labels = np.zeros((self.batch_size, 0))
        self.initial_latents_variable = create_variable_for_generator_1d(self.batch_size)
        self.initial_labels_variable = create_stub_mapping(self.batch_size)

        self.dlatents = model.components.mapping.get_output_for(self.initial_latents_variable, None)
        #print(self.dlatents.get_shape())
        self.generator_output = model.components.synthesis.get_output_for(self.dlatents, use_noise=True)

        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        self.set_latents(self.initial_latents)
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

        fake_images_out = tf.cast(self.generator_output, tf.float32)
        fake_images_out = tf.transpose(fake_images_out, [0, 2, 3, 1])
        self.images = fake_images_out
        self.fake_images_out = tf.image.resize_bilinear(fake_images_out, [112, 112])
        self.fake_images_out_flip = tf.image.flip_left_right(self.fake_images_out)
        embds_arr = build_InsightFace_model(self.fake_images_out, reuse=False)
        embds_flip_arr = build_InsightFace_model(self.fake_images_out_flip, reuse=True)
        embds_arr = embds_arr / tf.norm(embds_arr, ord='euclidean', axis=1, keepdims=True) + embds_flip_arr / tf.norm(
            embds_flip_arr, ord='euclidean', axis=1, keepdims=True)
        self.embds_arr = embds_arr / tf.norm(embds_arr, ord='euclidean', axis=1, keepdims=True)

    def reset_latents(self):
        initial_latents = np.random.randn(self.batch_size, 512)
        self.set_latents(initial_latents)

    def set_latents(self, latents):
        assert (latents.shape == (self.batch_size, 512))
        self.sess.run(tf.assign(self.initial_latents_variable, latents))

    def get_latents(self):
        return self.sess.run(self.initial_latents_variable)

    def generate_images(self, latents=None):
        if latents:
            self.set_latents(latents)
        return self.sess.run(self.generated_image_uint8)

    def get_embds_arr(self):
        return self.sess.run(self.embds_arr)

    def get_fake_images_out_flip(self):
        return self.sess.run(self.fake_images_out_flip)

    def get_images(self):
        return self.sess.run(self.images)