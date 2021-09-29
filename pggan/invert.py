# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import numpy as np
import tensorflow as tf

import config
import tfutil
import dataset
import misc
import PIL.Image
import tensorflow_probability as tfp
tfd = tfp.distributions

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.
def setup_snapshot_image_grid(G, training_set):
#def setup_snapshot_image_grid(rand_state, G, training_set):
    i = 0
    while i < 40:
        training_set.get_minibatch_np(1)
        i += 1

    gw = 4
    gh = 1

    # Fill in reals and labels.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=training_set.dtype)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    for idx in range(gw * gh):
        real, label = training_set.get_minibatch_np(1)
        reals[idx] = real[0]
        labels[idx] = label[0]
    # Generate latents.
    #latents = misc.random_latents(gw * gh, G, random_state=rand_state)
    latents = misc.random_latents(gw * gh, G)
    return (gw, gh), reals, labels, latents

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tfutil.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 4,        # Image resolution used at the beginning.
        lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
        ): # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        # Level-of-detail and resolution.
        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))


        self.minibatch = 4


def convert_img_drange(image, drange):
    assert image.ndim == 2 or image.ndim == 3
    image = misc.adjust_dynamic_range(image, drange, [0, 255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    return image
#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def invert_progressive_gan(
    invert_repeats       = 2000,            # How many times the discriminator is trained per G iteration.
    total_kimg              = 15000,        # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,        # Enable mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    resume_run_id           = 'example_import_script/karras2018iclr-celebahq-1024x1024.pkl',
    resume_snapshot         = None):        # Snapshot index to resume training from, None = autodetect.

    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    cur_nimg = total_kimg * 1000
    sched = TrainingSchedule(cur_nimg, training_set, **config.sched)

    # Construct networks.
    with tf.device('/gpu:0'):
        network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
        print('Loading networks from "%s"...' % network_pkl)
        G, D, Gs = misc.load_pkl(network_pkl)

    G.print_layers()

    lr = 5e-3
    alpha = 1e-6
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in = tf.placeholder(tf.float32, name='lod_in', shape=[])
    invert_opt = tf.train.RMSPropOptimizer(learning_rate=lr)
    with tf.name_scope('GPU%d' % 0), tf.device('/gpu:%d' % 0):
        lod_assign_ops = [tf.assign(G.find_var('lod'), lod_in)]
        with tf.name_scope('invert_loss'), tf.control_dependencies(lod_assign_ops):
            Zrand_placeholder = tf.placeholder(tf.float32)
            Reals_placeholder = tf.placeholder(tf.float32)
            reals_gpu = process_reals(Reals_placeholder, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            Labels_placeholder = tf.placeholder(tf.float32)
            labels_gpu = Labels_placeholder
            Zinit = tf.get_variable('Zinit', shape=[sched.minibatch]+G.input_shapes[0][1:], dtype=tf.float32, trainable=True)
            Zinit_assign = Zinit.assign(Zrand_placeholder)
            fake_images_out = G.get_output_for(Zinit, labels_gpu, is_training=False)
            recLoss_op = tf.reduce_mean(tf.square(reals_gpu - fake_images_out))
            logProb_op = tf.reduce_mean(tfd.Normal(loc=0.0, scale=1.0).log_prob(Zinit))
            loss_op = recLoss_op - alpha * logProb_op
        invert_op = invert_opt.minimize(loss=loss_op, var_list=[Zinit])

    print('Inverting...')
    with tf.get_default_session() as sess:
        # Choose training parameters and configure training ops.
        training_set.configure(sched.minibatch, sched.lod)
        print('########################################################################################')
        print("LOD: {:10.4f}".format(sched.lod))
        print('########################################################################################')

        sess.run(tf.variables_initializer(invert_opt.variables()))

        init_op = tf.group(tf.local_variables_initializer())
        sess.run(init_op)

        # Run training ops.
        #latents = np.random.normal(size=[sched.minibatch] + G.input_shapes[0][1:])
        #print("Latents Shape: "+str(latents.shape))
        # print("Grid Latents Shape: "+str(grid_latents.shape))
        # print("Grid Labels Shape: "+str(grid_labels.shape))
        #rand_state = np.random.RandomState(1234)
        #grid_size, grid_reals, grid_labels, grid_latents = setup_snapshot_image_grid(rand_state, G, training_set)
        grid_size, grid_reals, grid_labels, grid_latents = setup_snapshot_image_grid(G, training_set)
        sess.run([Zinit_assign], {Zrand_placeholder: grid_latents})
        result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
        for iter in range(invert_repeats):
            if iter % 10 == 0:
                rec_latents = sess.run(Zinit)
                grid_fakes = Gs.run(rec_latents, grid_labels, minibatch_size=sched.minibatch)
                # misc.save_image_grid(grid_reals, os.path.join(result_subdir, 'reals.png'),
                #                      drange=training_set.dynamic_range, grid_size=grid_size)
                # misc.save_image_grid(fakes, os.path.join(result_subdir, 'fakes%06d.png' % 0),
                #                      drange=drange_net, grid_size=grid_size)
                reals = np.asarray([convert_img_drange(grid_reals[i,...], drange=training_set.dynamic_range) for i in range(grid_reals.shape[0])])
                fakes = np.asarray([convert_img_drange(grid_fakes[i,...], drange=drange_net) for i in range(grid_fakes.shape[0])])
                images = np.concatenate((reals, fakes), axis=0)
                grid_images = misc.create_image_grid(images, (4, 2))
                grid_images = grid_images.transpose(1, 2, 0)  # CHW -> HWC
                format = 'RGB' if grid_images.ndim == 3 else 'L'
                PIL.Image.fromarray(grid_images, format).save(os.path.join(result_subdir, 'fakes%06d.png' % iter))
                #misc.convert_to_pil_image(misc.create_image_grid(grid_reals, grid_size), drange=training_set.dynamic_range)
                #misc.convert_to_pil_image(misc.create_image_grid(grid_fakes, grid_size), drange=drange_net)
            recLoss, _ = sess.run([recLoss_op, invert_op], {lod_in: sched.lod, Reals_placeholder: grid_reals, Labels_placeholder: grid_labels})
            print(recLoss)
#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    print('Running %s()...' % config.invert['func'])
    tfutil.call_func_by_name(**config.invert)
    print('Exiting...')

#----------------------------------------------------------------------------
