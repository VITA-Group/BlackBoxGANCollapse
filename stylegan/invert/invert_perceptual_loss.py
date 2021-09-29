import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import operator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
import tensorflow_probability as tfp
tfd = tfp.distributions

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
URL_CELEBAHQ    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl


def load_images(images_list, img_size):
    def cropND(img, bounding):
        print(img.shape)
        start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
        print(start)
        end = tuple(map(operator.add, start, bounding))
        print(end)
        slices = tuple(map(slice, start, end))
        print(slices)
        return img[slices]

    def pil_resize(img, target_size):
        width_height_tuple = (target_size[1], target_size[0])
        resample = pil_image.NEAREST
        img = img.resize(width_height_tuple, resample)
        return img
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path)
        img = np.asarray(img)
        img = cropND(img, (800, 800))
        img = pil_image.fromarray(img)
        img = pil_resize(img, target_size=(img_size, img_size))
        w, h = img.size
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


def create_save_dir(generated_images_dir, latent_dir):
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)
    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def invert_style_gan():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--src_dir', help='Directory with images for encoding')
    parser.add_argument('--generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('--latent_dir', help='Directory for storing latent representations')

    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    parser.add_argument('--use_ffhq_pretrained', default=False, help='Whether or not to use the generator pretrained on FFHQ', type=str2bool)

    parser.add_argument('--invert_ffhq', default=False, help='Whether or not to invert the FFHQ dataset', type=str2bool)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=5e-3, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to latents during optimization', type=str2bool)

    parser.add_argument('--opt_z_space', required=True, help='Whether to optimize in z space or w space', type=str2bool)
    parser.add_argument('--opt_w_full_space', required=True, help='Whether to optimize in w full space or w sub space', type=str2bool)
    parser.add_argument('--num_gpus', required=True, help='Number of gpus to be used in training', type=int)
    parser.add_argument('--save_dir', required=True, help='The directory for saving latent codes and recovered images', type=str)
    parser.add_argument('--start', required=True, help='starting index for the recovered image', type=int)
    parser.add_argument('--end', required=True, help='ending index for the recovered image', type=int)

    args, other_args = parser.parse_known_args()

    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    #os.makedirs(args.generated_images_dir, exist_ok=True)
    #os.makedirs(args.latent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    if args.use_ffhq_pretrained:
        with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)

    else:
        with dnnlib.util.open_url(URL_CELEBAHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
    print('Building TensorFlow graph...')


    lr = 1e-2
    alpha = 1
    #invert_opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    invert_opt = tf.train.RMSPropOptimizer(learning_rate=lr)
    grads_all = []
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    generator_lst = []
    losses = []
    logProbs = []

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(args.num_gpus):
            with tf.name_scope('GPU%d' % gpu_index), tf.device('/gpu:%d' % gpu_index):
                real_images_in = tf.get_variable('real', shape=[args.num_gpus*args.batch_size]+[256, 256, 3], dtype=tf.float32, trainable=False)
                weight_mask_in = tf.get_variable('weight_mask', shape=[args.num_gpus * args.batch_size] + [64, 64, 256],
                                                 dtype=tf.float32, trainable=False)
                G_gpu = Gs_network if gpu_index == 0 else Gs_network.clone(Gs_network.name + '_shadow')
                generator = Generator(G_gpu, args.batch_size, randomize_noise=args.randomize_noise,
                                  z_space=args.opt_z_space, w_full_space=args.opt_w_full_space)
                generator_lst.append(generator)
                perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=args.batch_size)
                perceptual_model.build_perceptual_model(generator.generated_image)
                perceptual_model.set_reference_images(real_images_in[gpu_index * args.batch_size:(gpu_index+1) * args.batch_size])

                recLoss = perceptual_model.compute_feature_loss(
                    weight_mask_in[gpu_index * args.batch_size:(gpu_index+1) * args.batch_size])
                logProb = tf.reduce_mean(
                        tfd.Normal(loc=0.0, scale=1.0).log_prob(generator.initial_latents_variable))

                loss = recLoss - alpha * logProb
                losses.append(loss)
                logProbs.append(logProb)
                grads = invert_opt.compute_gradients(loss=loss, var_list=[generator.initial_latents_variable])
                grads_all = grads_all + grads
                tf.get_variable_scope().reuse_variables()
    print(grads_all)
    loss_op = tf.reduce_mean(losses)
    logProb_op = tf.reduce_mean(logProbs)
    invert_op = invert_opt.apply_gradients(grads_all, global_step=global_step)

    opt_z_space = lambda bool: "opt_z_space" if bool else "opt_w_subspace"
    use_FFHQ_pretrained = lambda bool: "FFHQ_pretrained" if bool else "CELEBAHQ_pretrained"
    invert_FFHQ = lambda bool: "invert_FFHQ" if bool else "invert_CELEBAHQ"
    print('Inverting...')

    with tf.get_default_session() as sess:
        init_op = tf.group(tf.variables_initializer(invert_opt.variables()+[global_step]), tf.local_variables_initializer())
        # init_op = tf.group(tf.local_variables_initializer(), tf.variables_initializer([global_step]))
        sess.run(init_op)
        generated_images_dir = os.path.join(args.save_dir, opt_z_space(args.opt_z_space), args.generated_images_dir, use_FFHQ_pretrained(args.use_ffhq_pretrained), invert_FFHQ(args.invert_ffhq), str(alpha))
        latent_dir = os.path.join(args.save_dir, opt_z_space(args.opt_z_space), args.latent_dir, use_FFHQ_pretrained(args.use_ffhq_pretrained), invert_FFHQ(args.invert_ffhq), str(alpha))
        create_save_dir(generated_images_dir, latent_dir)
        for images_batch in tqdm(split_to_batches(ref_images, args.num_gpus*args.batch_size), total=len(ref_images)//(args.num_gpus*args.batch_size)):
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
            assert (len(images_batch) != 0 and len(images_batch) <= args.num_gpus * args.batch_size)
            loaded_images = load_images(images_batch, args.image_size)
            empty_images_shape = [args.num_gpus * args.batch_size - len(names)] + list(loaded_images.shape[1:])
            real_images = np.vstack([loaded_images, np.zeros(shape=empty_images_shape)])

            loaded_feature_mask = np.ones(shape=[loaded_images.shape[0]] + [64, 64, 256])
            empty_features_mask = np.zeros(shape=[args.num_gpus * args.batch_size - len(names)] + [64, 64, 256])
            weight_mask = np.vstack([loaded_feature_mask, empty_features_mask])

            sess.run(tf.assign(real_images_in, real_images))
            sess.run(tf.assign(weight_mask_in, weight_mask))
            for i in range(args.start, args.end):
                def train_op():
                    for _ in range(args.iterations):
                        #print(real_images)
                        #print(np.mean(real_images))
                        #print(sess.run(perceptual_model.image_features))
                        #print(np.mean(sess.run(perceptual_model.image_features)))
                        _, loss, normal_prior = sess.run([invert_op, loss_op, logProb_op])
                        # loss, normal_prior = sess.run([loss_op, logProb_op])
                        yield (loss, normal_prior)

                pbar = tqdm(train_op(), leave=False, total=args.iterations)

                for (loss, normal_prior) in pbar:
                    pbar.set_description(
                        ' '.join(names) + ' Loss: %.6f' % loss + ' Normal Prior: %.6f' % normal_prior)

                print(' '.join(names), ' loss:', loss)
                loss_summary = '{:10.8f},{:10.8f},'.format(loss, normal_prior)
                # Generate images from found latents and save them
                generated_images_lst = []
                generated_latents_lst = []
                for generator in generator_lst:
                    generated_images = generator.generate_images()
                    generated_latents = generator.get_latents()
                    generated_images_lst.append(generated_images)
                    generated_latents_lst.append(generated_latents)
                    generator.reset_latents()
                generated_images = np.concatenate(generated_images_lst, axis=0)
                generated_latents = np.concatenate(generated_latents_lst, axis=0)
                for img_array, latent, img_name in zip(generated_images, generated_latents, names):
                    img = PIL.Image.fromarray(img_array, 'RGB')
                    img.save(os.path.join(generated_images_dir, '{}_{}.png'.format(img_name, i)), 'PNG')
                    np.save(os.path.join(latent_dir, '{}_{}.npy'.format(img_name, i)), latent)


if __name__ == "__main__":
    invert_style_gan()
