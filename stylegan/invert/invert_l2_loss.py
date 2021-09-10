#import misc
import PIL.Image
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import tqdm

import os
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

import config
#from training import misc
import pickle
import argparse
from encoder.generator_model import Generator
import operator

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
URL_CELEBAHQ    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl

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

def load_images(images_list):
    def cropND(img, bounding):
        start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]

    loaded_images = list()
    for img_path in images_list:
        with pil_image.open(img_path) as img:
            img = np.asarray(img)
            img = cropND(img, (800, 800))
            img = np.expand_dims(img, 0)
            loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    loaded_images = loaded_images.astype(np.float32)
    loaded_images = adjust_dynamic_range(loaded_images, [0, 255], [-1, 1])
    #print(loaded_images)
    return loaded_images


# def setup_snapshot_image_grid(G, training_set):
# #def setup_snapshot_image_grid(rand_state, G, training_set):
#     i = 0
#     while i < 40:
#         training_set.get_minibatch_np(1)
#         i += 1
#
#     gw = 4
#     gh = 1
#
#     # Fill in reals and labels.
#     reals = np.zeros([gw * gh] + training_set.shape, dtype=training_set.dtype)
#     labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
#     for idx in range(gw * gh):
#         real, label = training_set.get_minibatch_np(1)
#         reals[idx] = real[0]
#         labels[idx] = label[0]
#     # Generate latents.
#     #latents = misc.random_latents(gw * gh, G, random_state=rand_state)
#     latents = misc.random_latents(gw * gh, G)
#     return (gw, gh), reals, labels, latents

def convert_img_drange(image, drange):
    assert image.ndim == 2 or image.ndim == 3
    image = misc.adjust_dynamic_range(image, drange, [0, 255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    return image

def create_save_dir(generated_images_dir, latent_dir):
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)
    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

# Average the gradients for each shared variable across all towers
def sum_gradients_over_towers(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    sum_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_sum(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        sum_grads.append(grad_and_var)
    return sum_grads

def invert_style_gan():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--src_dir', help='Directory with images for encoding')
    parser.add_argument('--generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('--latent_dir', help='Directory for storing latent representations')

    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    parser.add_argument('--use_ffhq_pretrained', required=True, help='Whether or not to use the generator pretrained on FFHQ', type=str2bool)

    parser.add_argument('--invert_ffhq', required=True, help='Whether or not to invert the FFHQ dataset', type=str2bool)

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

    # Initialize generator and perceptual model
    tflib.init_tf()
    if args.use_ffhq_pretrained:
        with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)

    else:
        with dnnlib.util.open_url(URL_CELEBAHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
    print('Building TensorFlow graph...')

    lr = 5e-3
    alpha = 1e-2
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
                real_images_in = tf.get_variable('real', shape=[args.num_gpus*args.batch_size]+[800, 800, 3], dtype=tf.float32, trainable=False)
                weight_mask_in = tf.get_variable('weight_mask', shape=[args.num_gpus*args.batch_size]+[800, 800, 3], dtype=tf.float32, trainable=False)
                G_gpu = Gs_network if gpu_index == 0 else Gs_network.clone(Gs_network.name + '_shadow')
                generator = Generator(G_gpu, args.batch_size, randomize_noise=args.randomize_noise,
                                  z_space=args.opt_z_space, w_full_space=args.opt_w_full_space)
                generator_lst.append(generator)
                fake_images_out = generator.generator_output
                fake_images_out = tf.transpose(fake_images_out, [0, 2, 3, 1])
                fake_images_out = tf.slice(fake_images_out, [0, 112, 112, 0], [args.batch_size, 800, 800, 3])
                recLoss = tf.reduce_mean(tf.square(
                    weight_mask_in[gpu_index * args.batch_size:(gpu_index+1) * args.batch_size] * real_images_in[gpu_index * args.batch_size:(gpu_index+1) * args.batch_size] -
                    weight_mask_in[gpu_index * args.batch_size:(gpu_index+1) * args.batch_size] * fake_images_out))
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
        #init_op = tf.group(tf.local_variables_initializer(), tf.initialize_variables([global_step]))
        sess.run(init_op)
        generated_images_dir = os.path.join(args.save_dir, opt_z_space(args.opt_z_space), args.generated_images_dir, use_FFHQ_pretrained(args.use_ffhq_pretrained), invert_FFHQ(args.invert_ffhq), str(alpha))
        latent_dir = os.path.join(args.save_dir, opt_z_space(args.opt_z_space), args.latent_dir, use_FFHQ_pretrained(args.use_ffhq_pretrained), invert_FFHQ(args.invert_ffhq), str(alpha))
        create_save_dir(generated_images_dir, latent_dir)
        for images_batch in tqdm(split_to_batches(ref_images, args.num_gpus*args.batch_size), total=len(ref_images)//(args.num_gpus*args.batch_size)):
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
            loaded_images = load_images(images_batch)
            empty_images_shape = [args.num_gpus * args.batch_size - len(names)] + list(loaded_images.shape[1:])
            real_images = np.vstack([loaded_images, np.zeros(shape=empty_images_shape)])

            loaded_images_mask = np.ones(shape=loaded_images.shape)
            empty_images_mask = np.zeros(shape=empty_images_shape)
            weight_mask = np.vstack([loaded_images_mask, empty_images_mask])

            sess.run(tf.assign(weight_mask_in, weight_mask))
            sess.run(tf.assign(real_images_in, real_images))
            for i in range(args.start, args.end):
                def train_op():
                    for _ in range(args.iterations):
                        _, loss, normal_prior = sess.run([invert_op, loss_op, logProb_op])
                        #loss, normal_prior = sess.run([loss_op, logProb_op])
                        yield (loss, normal_prior)

                pbar = tqdm(train_op(), leave=False, total=args.iterations)

                for (loss, normal_prior) in pbar:
                    pbar.set_description(' '.join(names)+' Loss: %.6f' % loss +' Normal Prior: %.6f' % normal_prior)

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