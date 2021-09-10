# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""
import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import tensorflow as tf

#----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()

def load_Gs(url):
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]

def create_stub_mapping(batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

def create_stub_synthesis(batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

def create_variable_for_generator_1d(batch_size):
    return tf.get_variable('learnable_latents',
                           shape=(batch_size, 512),
                           dtype='float32',
                           initializer=tf.initializers.random_normal())

def create_variable_for_generator_2d(batch_size):
    return tf.get_variable('learnable_latents',
                           shape=(batch_size, 18, 512),
                           dtype='float32',
                           initializer=tf.initializers.random_normal(), trainable=False)

def draw_uncurated_result_figure(Gs, cx, cy, cw, ch, rows, lods, seed):

    latents = np.random.RandomState(seed).randn(sum(rows * 2**lod for lod in lods), Gs.input_shape[1])
    initial_labels = np.zeros((1, 0))
    #initial_latents_variable = create_variable_for_generator_1d(1)
    #initial_labels_variable = create_stub_mapping(1)
    dlatents = Gs.components.mapping.get_output_for(latents, None)
    generator_output = Gs.components.synthesis.get_output_for(dlatents, use_noise=False)

    generated_image = tflib.convert_images_to_uint8(generator_output, nchw_to_nhwc=True, uint8_cast=False)
    generated_image_uint8 = tf.saturate_cast(generated_image, tf.uint8)
    canvas = PIL.Image.new('RGB', (sum(cw // 2**lod for lod in lods), ch * rows), 'white')
    sess = tf.get_default_session()
    #graph = tf.get_default_graph()
    #sess.run(tf.assign(initial_latents_variable, latents))
    image_iter = iter(list(sess.run(generated_image_uint8)))
    for col, lod in enumerate(lods):
        for row in range(rows * 2**lod):
            image = PIL.Image.fromarray(next(image_iter), 'RGB')
            image = image.crop((cx, cy, cx + cw, cy + ch))
            image = image.resize((cw // 2**lod, ch // 2**lod), PIL.Image.ANTIALIAS)
            canvas.paste(image, (sum(cw // 2**lod for lod in lods[:col]), row * ch // 2**lod))
    canvas.save('ffhq-uncurated-new.png')
#----------------------------------------------------------------------------
# Main program.

def main():
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
    draw_uncurated_result_figure(load_Gs(url_ffhq), cx=0, cy=0, cw=1024, ch=1024, rows=1, lods=[0], seed=6)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
