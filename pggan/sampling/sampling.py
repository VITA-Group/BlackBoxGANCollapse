import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import pickle
import numpy as np
import argparse
import backbones.modifiedResNet_v2 as modifiedResNet_v2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from argparse import Namespace
import yaml
from scipy import misc


def load_Gs(model_path):
    #pkl = os.listdir(model_path)[0]
    #with open(os.path.join(model_path, pkl), 'rb') as file:
    with open(model_path, 'rb') as file:
        print(file)
        G, D, Gs = pickle.load(file)
        return Gs

def proc_images(imgs, imsize):
    img_lst = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        #img = cropND(img, (800, 800))
        img = misc.imresize(img, [imsize, imsize])
        img_lst.append(img)
    return img_lst

def flip_images(imgs):
    images = []
    images_f = []
    for img in imgs:
        img_f = np.fliplr(img)
        img = img/127.5-1.0
        img_f = img_f/127.5-1.0
        images.append(img)
        images_f.append(img_f)
    return (np.array(images), np.array(images_f))

def build_InsightFace_model():
    FLAGS = Namespace()
    FLAGS.CONFIG_PATH = '../InsightFace/configs/config_ms1m_100.yaml'
    FLAGS.MODEL_PATH = '../InsightFace/pretrained/config_ms1m_100_334k/best-m-334000'
    print('building InsightFace model...')
    config = yaml.load(open(FLAGS.CONFIG_PATH))

    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope('embd_extractor', reuse=False):
            images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image')
            train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
            train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
            net = images
            end_points = {}
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
    #print(batch_num)
    for i in range(batch_num):
        image_batch = images[i*batch_size: (i+1)*batch_size]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: False, train_ph_bn: True})
        embds += list(cur_embd)
        #print('%d/%d' % (i, batch_num))
    if left > 0:
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: images[-left:], train_ph_dropout: False, train_ph_bn: True})
        embds += list(cur_embd)[:left]
    return np.array(embds)

def sampling(Gs, start, end, save_dir):
    sess_face, embds_ph, images_ph_face, train_phase_dropout, train_phase_bn = build_InsightFace_model()
    embds_dir = os.path.join(save_dir, 'embds', '{}_{}'.format(start, end))
    latents_dir = os.path.join(save_dir, 'latents', '{}_{}'.format(start, end))

    if not os.path.exists(embds_dir):
        os.makedirs(embds_dir)
    if not os.path.exists(latents_dir):
        os.makedirs(latents_dir)
    batch_size = 8
    latents_lst, embds_lst = [], []
    for i in range(start, end, batch_size):
        latents = np.random.randn(batch_size, Gs.input_shape[1])
        latents_lst.append(latents)
        images = Gs.run(latents, np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:]))
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

        images_112 = proc_images(images, 112)
        imgs, imgs_f = flip_images(images_112)
        embds_arr = run_identity_embds(sess_face, imgs, batch_size, embds_ph, images_ph_face, train_phase_dropout, train_phase_bn)
        embds_f_arr = run_identity_embds(sess_face, imgs_f, batch_size, embds_ph, images_ph_face, train_phase_dropout, train_phase_bn)
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)+embds_f_arr/np.linalg.norm(embds_f_arr,axis=1,keepdims=True)
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)
        embds_lst.append(embds_arr)

        if (i+batch_size) % 10000 == 0:
            print(i)
            s, e = i+batch_size-10000, i+batch_size
            pickle.dump(np.concatenate(embds_lst, axis=0), open(os.path.join(embds_dir, '{}_{}.pkl'.format(s,e)), 'wb'))
            np.save(os.path.join(latents_dir, '{}_{}.npy'.format(s,e)), np.concatenate(latents_lst, axis=0))
            latents_lst, embds_lst = [], []

def main():
    parser = argparse.ArgumentParser(description='Random sampling')
    parser.add_argument('--start', required=True, help='Starting index', type=int)
    parser.add_argument('--end', required=True, help='Ending index', type=int)
    parser.add_argument('--save_dir', required=True, help='The saving directory for the sampled images', type=str)
    args, other_args = parser.parse_known_args()

    # Initialize TensorFlow session.
    tf.InteractiveSession()
    sampling(load_Gs('../pretrained_models/karras2018iclr-celebahq-1024x1024.pkl'), args.start, args.end, args.save_dir)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------