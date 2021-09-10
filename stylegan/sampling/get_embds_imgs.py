import numpy as np
from scipy import misc
import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import pickle
import dnnlib.tflib as tflib
import operator
import argparse
import backbones.modifiedResNet_v2 as modifiedResNet_v2
import backbones.inception_resnet_v1 as inception_resnet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim
from argparse import Namespace
import yaml
import glob
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


def build_InsightFace_model():
    FLAGS = Namespace()
    FLAGS.CONFIG_PATH = '../InsightFace/configs/config_ms1m_100.yaml'
    FLAGS.MODEL_PATH = '../InsightFace/pretrained/config_ms1m_100_1006k/best-m-1006000'
    print('building InsightFace model...')
    config = yaml.load(open(FLAGS.CONFIG_PATH))

    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope('embd_extractor', reuse=False):
            images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image')
            train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
            train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
            net = images
            arg_sc = modifiedResNet_v2.resnet_arg_scope(weight_decay=config['weight_decay'],
                                                        batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                net, end_points = modifiedResNet_v2.resnet_v2_m_50(net, is_training=train_phase_bn, return_raw=True)

            with slim.arg_scope(arg_sc):
                net = slim.batch_norm(net, activation_fn=None, is_training=train_phase_bn)
                net = slim.dropout(net, keep_prob=config['keep_prob'], is_training=train_phase_dropout)
                net = slim.flatten(net)
                net = slim.fully_connected(net, config['embd_size'], normalizer_fn=None, activation_fn=None)
                net = slim.batch_norm(net, scale=False, activation_fn=None, is_training=train_phase_bn)
                end_points['embds'] = net

            embds = net
            tf_config = tf.ConfigProto(allow_soft_placement=True)
            tf_config.gpu_options.allow_growth = True
            sess = tf.Session(graph=graph, config=tf_config)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(var_list=[v for v in tf.trainable_variables() if
                           any(x in v.name.split('/')[0] for x in ["embd_extractor"])])
            saver.restore(sess, FLAGS.MODEL_PATH)
            return sess, embds, images, train_phase_dropout, train_phase_bn

def run_identity_embds(sess, images, batch_size, embds_ph, image_ph, train_ph_dropout, train_ph_bn):
    batch_num = len(images)//batch_size
    left = len(images)%batch_size
    embds = []
    for i in range(batch_num):
        image_batch = images[i*batch_size: (i+1)*batch_size]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: False, train_ph_bn: True})
        embds += list(cur_embd)
    if left > 0:
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: images[-left:], train_ph_dropout: False, train_ph_bn: True})
        embds += list(cur_embd)[:left]
    return np.array(embds)

def load_image(path, image_size, batch_size):
    from skimage.io import imread_collection

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    print('reading %s' % path)
    if os.path.isdir(path):
        #img_paths = list(os.listdir(path))
        img_paths = glob.glob(os.path.join(path, '**/*.png'), recursive=True)
    else:
        raise ValueError('The given path is not a directory')
    img_paths_split = list(chunks(img_paths, batch_size))

    for img_paths in img_paths_split:
        images = []
        images_f = []
        for path in img_paths:
            img = misc.imread(path)
            img = misc.imresize(img, [image_size, image_size])
            # img = img[s:s+image_size, s:s+image_size, :]
            img_f = np.fliplr(img)
            img = img/127.5-1.0
            img_f = img_f/127.5-1.0
            images.append(img)
            images_f.append(img_f)
        yield np.array(images), np.array(images_f)

def get_embds(read_imgs_dir, save_embds_dir):
    sess_face, embds_ph, images_ph_face, train_phase_dropout, train_phase_bn = build_InsightFace_model()
    if not os.path.exists(save_embds_dir):
        os.makedirs(save_embds_dir)

    image_size = 112
    batch_size = 8
    embds_lst = []

    imgs_iter = load_image(read_imgs_dir, image_size, batch_size)

    try:
        while True:
            imgs, imgs_f = next(imgs_iter)
            embds_arr = run_identity_embds(sess_face, imgs, batch_size, embds_ph, images_ph_face, train_phase_dropout,
                                           train_phase_bn)
            embds_f_arr = run_identity_embds(sess_face, imgs_f, batch_size, embds_ph, images_ph_face,
                                             train_phase_dropout, train_phase_bn)
            embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True) + embds_f_arr / np.linalg.norm(
                embds_f_arr, axis=1, keepdims=True)
            embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)
            embds_lst.append(embds_arr)
    except StopIteration:
        pickle.dump(np.concatenate(embds_lst, axis=0),
                    open(os.path.join(save_embds_dir, 'ffhq.pkl'), 'wb'))
    finally:
        del imgs_iter


def main():
    parser = argparse.ArgumentParser(description='Random sampling')
    parser.add_argument('--imgs_dir', required=True, help='The directory of images to read and get embds', type=str)
    parser.add_argument('--save_embds_dir', required=True, help='The saving directory for the sampled images', type=str)
    args, other_args = parser.parse_known_args()
    tflib.init_tf()
    get_embds(args.imgs_dir, args.save_embds_dir)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
