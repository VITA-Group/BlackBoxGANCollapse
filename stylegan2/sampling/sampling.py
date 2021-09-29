import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import pickle
import numpy as np
import dnnlib.tflib as tflib
import operator
import argparse
import backbones.modifiedResNet_v2 as modifiedResNet_v2
import backbones.inception_resnet_v1 as inception_resnet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim
from argparse import Namespace
import yaml
from scipy import misc
from utils import load_Gs, url_ffhq, url_celebahq
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

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
        race = np.asarray(row[1][:3])
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
        #gender_op = tf.argmax(tf.nn.softmax(gender_logits), 1)
        gender_op = tf.nn.softmax(gender_logits)
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

def sampling(Gs, start, end, save_dir):
    sess_face, embds_ph, images_ph_face, train_phase_dropout, train_phase_bn = build_InsightFace_model()
    embds_dir = os.path.join(save_dir, 'embds', '{}_{}'.format(start, end))
    latents_dir = os.path.join(save_dir, 'latents', '{}_{}'.format(start, end))

    if not os.path.exists(embds_dir):
        os.makedirs(embds_dir)
    if not os.path.exists(latents_dir):
        os.makedirs(latents_dir)

    batch_size = 8
    latents_lst, embds_lst, race_lst, gender_lst, age_lst = [], [], [], [], []
    for i in range(start, end, batch_size):
        latents = np.random.randn(batch_size, Gs.input_shape[1])
        latents_lst.append(latents)
        images = Gs.run(latents, None, **synthesis_kwargs)
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
            latents_lst, embds_lst, race_lst, gender_lst = [], [], [], []

def main():
    parser = argparse.ArgumentParser(description='Random sampling')
    parser.add_argument('--start', required=True, help='Starting index for images to be sampled', type=int)
    parser.add_argument('--end', required=True, help='Ending index for images to be sampled', type=int)
    parser.add_argument('--save_dir', required=True, help='The saving directory for the sampled images', type=str)
    args, other_args = parser.parse_known_args()
    tflib.init_tf()
    sampling(load_Gs('../pretrained_models/lsun-car/karras2019stylegan-cars-512x384.pkl'), args.start, args.end, args.save_dir)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
