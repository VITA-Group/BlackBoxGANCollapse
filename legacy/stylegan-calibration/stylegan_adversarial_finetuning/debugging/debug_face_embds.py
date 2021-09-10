import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import argparse
import pickle
import PIL.Image
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
URL_CELEBAHQ    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl


def create_save_dir(generated_images_dir, embd_dir):
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)
    if not os.path.exists(embd_dir):
        os.makedirs(embd_dir)

def debug_style_gan():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--save_dir', required=True, help='The directory for saving latent codes and recovered images', type=str)

    args, other_args = parser.parse_known_args()

    tflib.init_tf()
    print(config.cache_dir)
    with open('/mnt/ilcompf5d1/user/zwu/stylegan_adversarial_finetuning/results/00012-sgan-celebahq-4gpu_none_128/network-snapshot-012000.pkl', 'rb') as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    print('Building TensorFlow graph...')


    with tf.name_scope('GPU%d' % 0), tf.device('/gpu:%d' % 0):
        G_gpu = Gs_network
        generator = Generator(G_gpu, args.batch_size)

    with tf.get_default_session() as sess:
        saver = tf.train.Saver(var_list=[v for v in tf.trainable_variables() if
                                         any(x in v.name.split('/')[0] for x in ["embd_extractor"])])
        saver.restore(sess, '../InsightFace/pretrained/config_ms1m_100_334k/best-m-334000')
        var_list = [v.name for v in tf.trainable_variables() if
                    any(x in v.name.split('/')[0] for x in ["embd_extractor"])]

        for var in var_list:
            print(var)

        generated_images_dir = os.path.join(args.save_dir, 'images')
        embd_dir = os.path.join(args.save_dir, 'embds')
        create_save_dir(generated_images_dir, embd_dir)

        generated_images = generator.generate_images()
        generated_embds = generator.get_embds_arr()
        fake_images_out = generator.get_fake_images_out_flip()
        images_out = generator.get_images()
        i = 1
        for img_array, embd, fake_images, images in zip(generated_images, generated_embds, fake_images_out, images_out):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(generated_images_dir, '{}.png'.format(i)), 'PNG')
            pickle.dump(embd, open(os.path.join(embd_dir, '{}.pkl'.format(i)), 'wb'))
            #fake_images[::,0].tofile('fake_images.txt', sep=" ", format="%s")
            fake_images = np.clip(np.rint((fake_images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
            fake_images = PIL.Image.fromarray(fake_images, 'RGB')
            #image = image.crop((112, 112, 912, 912))
            #image = image.resize((112, 112), PIL.Image.ANTIALIAS)
            fake_images.save('{}.png'.format(i), 'PNG')
            print(np.asarray(fake_images).shape)
            i += 1


if __name__ == "__main__":
    debug_style_gan()
