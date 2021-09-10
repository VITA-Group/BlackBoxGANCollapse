import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import operator
import argparse
import backbones.modifiedResNet_v2 as modifiedResNet_v2
import backbones.ResNet_v2 as ResNet_v2
import backbones.inception_resnet_v1 as inception_resnet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim
from argparse import Namespace
import yaml
from scipy import misc
from face_recognition import face_locations
import face_recognition
import pandas as pd
import re
from scipy.spatial import Delaunay

url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

def load_Gs(model_path):
    #pkl = os.listdir(model_path)[0]
    #with open(os.path.join(model_path, pkl), 'rb') as file:
    with open(model_path, 'rb') as file:
        print(file)
        G, D, Gs = pickle.load(file)
        return Gs

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

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

def predict_race(imgs, clf, labels):
    COLS = ['Asian', 'White', 'Black']
    N_UPSCLAE = 1
    race_lst = []
    for img in imgs:
        locs = face_locations(img, number_of_times_to_upsample = N_UPSCLAE)
        if len(locs) == 0:
            locs = [(1, img.shape[0], img.shape[1], 1)]
        face_encodings = face_recognition.face_encodings(img, known_face_locations=locs)
        pred = pd.DataFrame(clf.predict_proba(face_encodings), columns = labels)
        pred_df = pred.loc[:, COLS]
        row = next(pred_df.iterrows())
        ind = np.argmax(np.asarray(row[1][:3]))
        race = COLS[ind]
        race_lst.append(race)
    return race_lst

def build_age_gender_model():
    graph = tf.Graph()
    with graph.as_default():
        images_ph = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_ph)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=False,
                                                                     weight_decay=1e-5)
        gender_op = tf.argmax(tf.nn.softmax(gender_logits), 1)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age_op = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph, config=config)
        sess.run(init_op)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('age_gender_models')
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("age-gender prediction model is restored from {}!".format(ckpt.model_checkpoint_path))
        else:
            raise ValueError("unable to find pretrained age-gender prediction model at {}!".format(ckpt.model_checkpoint_path))

    return sess, age_op, gender_op, images_ph

def eval_age_gender(sess, images, batch_size, age_op, gender_op, images_ph):
    batch_num = len(images)//batch_size
    left = len(images)%batch_size
    age_lst, gender_lst = [], []
    for i in range(batch_num):
        image_batch = images[i*batch_size: (i+1)*batch_size]
        ages, genders = sess.run([age_op, gender_op], feed_dict={images_ph: image_batch})
        age_lst.extend(ages)
        gender_lst.extend(genders)
    if left > 0:
        ages, genders = sess.run([age_op, gender_op], feed_dict={images_ph: images[-left:]})
        age_lst.extend(ages)
        gender_lst.extend(genders)
    return age_lst, gender_lst

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

def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(ind))

def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def monte_carlo_sampling(Gs, start, end, save_dir):
    latents_dir = 'monte_carlo_sampling_1m_128_1/neighbors/0.25/664058/sampled/latents'
    latents_files = [os.path.join(latents_dir, latent) for latent in os.listdir(latents_dir)]
    latents_files.sort(key=lambda txt: grep(r"(\d+)_(\d+)_(\d+)\.npy", txt, 3))
    latents_mat = np.stack([np.load(file) for file in latents_files], axis=0)
    convex_hull = Delaunay(latents_mat)
    # with open('race_model/face_model.pkl', 'rb') as f:
    #     clf, labels = pickle.load(f, encoding='latin1')

    # sess_age_gender, age_op, gender_op, images_ph_age_gender = build_age_gender_model()
    sess_face, embds_ph, images_ph_face, train_phase_dropout, train_phase_bn = build_InsightFace_model()

    embds_dir = os.path.join(save_dir, 'embds', '{}_{}'.format(start, end))
    latents_dir = os.path.join(save_dir, 'latents', '{}_{}'.format(start, end))
    # genders_dir = os.path.join(save_dir, 'genders', '{}_{}'.format(start, end))
    # ages_dir = os.path.join(save_dir, 'ages', '{}_{}'.format(start, end))
    # races_dir = os.path.join(save_dir, 'races', '{}_{}'.format(start, end))
    if not os.path.exists(embds_dir):
        os.makedirs(embds_dir)
    if not os.path.exists(latents_dir):
        os.makedirs(latents_dir)
    # if not os.path.exists(genders_dir):
    #     os.makedirs(genders_dir)
    # if not os.path.exists(ages_dir):
    #     os.makedirs(ages_dir)
    # if not os.path.exists(races_dir):
    #     os.makedirs(races_dir)
    batch_size = 8
    latents_lst, embds_lst, race_lst, gender_lst, age_lst = [], [], [], [], []
    for i in range(start, end, batch_size):
        latents = np.random.randn(batch_size, Gs.input_shape[1])
        latents_lst.append(latents)
        images = Gs.run(latents, None, **synthesis_kwargs)
        images_112 = proc_images(images, 112)
        # races = predict_race(images_112, clf, labels)
        # race_lst.extend(races)
        imgs, imgs_f = flip_images(images_112)
        embds_arr = run_identity_embds(sess_face, imgs, batch_size, embds_ph, images_ph_face, train_phase_dropout, train_phase_bn)
        embds_f_arr = run_identity_embds(sess_face, imgs_f, batch_size, embds_ph, images_ph_face, train_phase_dropout, train_phase_bn)
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)+embds_f_arr/np.linalg.norm(embds_f_arr,axis=1,keepdims=True)
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)
        embds_lst.append(embds_arr)
        # images_160 = proc_images(images, 160)
        # ages, genders = eval_age_gender(sess_age_gender, images_160, batch_size, age_op, gender_op, images_ph_age_gender)
        # age_lst.extend(ages)
        # gender_lst.extend(genders)
        if (i+batch_size) % 10000 == 0:
            print(i)
            s, e = i+batch_size-10000, i+batch_size
            pickle.dump(np.concatenate(embds_lst, axis=0), open(os.path.join(embds_dir, '{}_{}.pkl'.format(s,e)), 'wb'))
            # pickle.dump(race_lst, open(os.path.join(races_dir, '{}_{}.pkl'.format(s,e)), 'wb'))
            # pickle.dump(age_lst, open(os.path.join(ages_dir, '{}_{}.pkl'.format(s,e)), 'wb'))
            # pickle.dump(gender_lst, open(os.path.join(genders_dir, '{}_{}.pkl'.format(s,e)), 'wb'))
            np.save(os.path.join(latents_dir, '{}_{}.npy'.format(s,e)), np.concatenate(latents_lst, axis=0))
            latents_lst, embds_lst, race_lst, gender_lst, age_lst = [], [], [], [], []

def main():
    parser = argparse.ArgumentParser(description='Random sampling')
    parser.add_argument('--start', required=True, help='Starting index for images to be sampled', type=int)
    parser.add_argument('--end', required=True, help='Ending index for images to be sampled', type=int)
    parser.add_argument('--save_dir', required=True, help='The saving directory for the sampled images', type=str)
    parser.add_argument('--model_dir', required=True, help='The model dir for reading', type=str)
    args, other_args = parser.parse_known_args()
    tflib.init_tf()
    monte_carlo_sampling(load_Gs(args.model_dir), args.start, args.end, args.save_dir)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
