# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Loss functions."""
import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import tensorflow.contrib.slim as slim
import backbones.modifiedResNet_v2 as modifiedResNet_v2
import PIL.Image
import numpy as np
import math
#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# WGAN & WGAN-GP loss functions.

def G_wgan(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -fake_scores_out
    return loss

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_epsilon = 0.001): # Weight for the epsilon term, \epsilon_{drift}.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss

#----------------------------------------------------------------------------
# Hinge loss functions. (Use G_wgan with these)

def D_hinge(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1.+fake_scores_out) + tf.maximum(0., 1.-real_scores_out)
    return loss

def D_hinge_gp(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1.+fake_scores_out) + tf.maximum(0., 1.-real_scores_out)

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    return loss


#----------------------------------------------------------------------------
# Loss functions advocated by the paper
# "Which Training Methods for GANs do actually Converge?"

def G_logistic_saturating(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -tf.nn.softplus(fake_scores_out)  # log(1 - logistic(fake_scores_out))
    return loss

def build_InsightFace_model(images):
    #with tf.variable_scope('embd_extractor', reuse=False):
        #images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image')
        #net = images
    with tf.variable_scope('embd_extractor', reuse=tf.AUTO_REUSE):
        arg_sc = modifiedResNet_v2.resnet_arg_scope(weight_decay=5e-4, batch_norm_decay=0.9)
        with slim.arg_scope(arg_sc):
            net, end_points = modifiedResNet_v2.resnet_v2_m_50(images, is_training=True, return_raw=True)
            net = slim.batch_norm(net, activation_fn=None, is_training=True)
            net = slim.dropout(net, keep_prob=0.4, is_training=False)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 512, normalizer_fn=None, activation_fn=None)
            embds = slim.batch_norm(net, scale=False, activation_fn=None, is_training=True)
        return embds

def G_logistic_nonsaturating(G, D, opt, training_set, minibatch_size, collapsed): # pylint: disable=unused-argument
#def G_logistic_nonsaturating(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = tf.nn.softplus(-fake_scores_out)  # -log(logistic(fake_scores_out))
    #loss -= tf.losses.mean_squared_error(fake_images_out, collapsed)

    fake_images_out = tf.cast(fake_images_out, tf.float32)
    fake_images_out = tf.transpose(fake_images_out, [0, 2, 3, 1])
    fake_images_out = tf.image.resize_bilinear(fake_images_out, [112, 112])
    fake_images_out_flip = tf.image.flip_left_right(fake_images_out)
    print(tf.shape(fake_images_out))
    print(fake_images_out.get_shape())
    fake_embds_arr = build_InsightFace_model(fake_images_out)
    fake_flip_embds_arr = build_InsightFace_model(fake_images_out_flip)
    fake_embds_arr = fake_embds_arr / tf.norm(fake_embds_arr, ord='euclidean', axis=1, keepdims=True)+fake_flip_embds_arr / tf.norm(fake_flip_embds_arr, ord='euclidean', axis=1, keepdims=True)
    fake_embds_arr = fake_embds_arr / tf.norm(fake_embds_arr, ord='euclidean', axis=1, keepdims=True)

    # cos(pi/4) = 0.7071
    #dist_arr = tf.math.acos(tf.linalg.matmul(fake_embds_arr, tf.constant(collapsed), transpose_b=True)) / math.pi
    dist_arr = tf.linalg.matmul(fake_embds_arr, tf.constant(collapsed), transpose_b=True)
    #hinge_loss = tf.reduce_sum(tf.maximum(0.0, tf.fill(tf.shape(dist_arr), 0.25)-dist_arr))
    hinge_loss = tf.reduce_sum(tf.maximum(0.0, dist_arr - tf.fill(tf.shape(dist_arr), 0.7071)))
    # var_list=[v.name for v in tf.trainable_variables() if
    #                         any(x in v.name.split('/')[0] for x in ["embd_extractor"])]
    #
    # for var in var_list:
    #     print(var)
    #tf.get_variable_scope().reuse_variables()
    return loss+hinge_loss

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type
    return loss

def D_logistic_simplegp(G, D, opt, training_set, minibatch_size, reals, labels, r1_gamma=10.0, r2_gamma=0.0):  # pylint: disable=unused-argument
#def D_logistic_simplegp(G, D, opt, training_set, minibatch_size, reals, labels, collapsed, r1_gamma=10.0, r2_gamma=0.0): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type

    # collapsed_scores_out = fp32(D.get_output_for(collapsed, labels, is_training=True))
    # loss += tf.nn.softplus(-collapsed_scores_out)

    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty'):
            real_loss = opt.apply_loss_scaling(tf.reduce_sum(real_scores_out))
            real_grads = opt.undo_loss_scaling(fp32(tf.gradients(real_loss, [reals])[0]))
            r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
            r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss += r1_penalty * (r1_gamma * 0.5)

    if r2_gamma != 0.0:
        with tf.name_scope('R2Penalty'):
            fake_loss = opt.apply_loss_scaling(tf.reduce_sum(fake_scores_out))
            fake_grads = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss, [fake_images_out])[0]))
            r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])
            r2_penalty = autosummary('Loss/r2_penalty', r2_penalty)
        loss += r2_penalty * (r2_gamma * 0.5)
    return loss
#----------------------------------------------------------------------------
