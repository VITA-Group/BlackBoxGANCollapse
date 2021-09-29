import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model_legacy import PerceptualModel

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

def main():
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
    parser.add_argument('--opt_w_full_space', default=False, help='Add noise to latents during optimization', type=str2bool)
    parser.add_argument('--save_dir', required=True, help='The directory for saving latent codes and recovered images', type=str)

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

    generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise, z_space=args.opt_z_space, w_full_space=args.opt_w_full_space)
    perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=args.batch_size)

    perceptual_model.build_perceptual_model(generator.generated_image)

    opt_z_space = lambda bool: "opt_z_space" if bool else "opt_w_subspace"
    use_FFHQ_pretrained = lambda bool: "FFHQ_pretrained" if bool else "CELEBAHQ_pretrained"
    invert_FFHQ = lambda bool: "invert_FFHQ" if bool else "invert_CELEBAHQ"

    for alpha in [1]:
        # Optimize (only) latents by minimizing perceptual loss between reference and generated images in feature space
        perceptual_model.set_alpha(alpha)

        generated_images_dir = os.path.join(args.save_dir, opt_z_space(args.opt_z_space), args.generated_images_dir, use_FFHQ_pretrained(args.use_ffhq_pretrained), invert_FFHQ(args.invert_ffhq), str(alpha))
        latent_dir = os.path.join(args.save_dir, opt_z_space(args.opt_z_space), args.latent_dir, use_FFHQ_pretrained(args.use_ffhq_pretrained), invert_FFHQ(args.invert_ffhq), str(alpha))
        if not os.path.exists(generated_images_dir):
            os.makedirs(generated_images_dir)
        if not os.path.exists(latent_dir):
            os.makedirs(latent_dir)

        for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
            for i in range(100):
                names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

                perceptual_model.set_reference_images(images_batch)

                op = perceptual_model.optimize(generator.initial_latents_variable, iterations=args.iterations, learning_rate=args.lr)

                pbar = tqdm(op, leave=False, total=args.iterations)

                for (loss, normal_prior) in pbar:
                    pbar.set_description(' '.join(names)+' Loss: %.6f' % loss +' Normal Prior: %.6f' % normal_prior)

                print(' '.join(names), ' loss:', loss)
                loss_summary = '{:10.8f},{:10.8f},'.format(loss, normal_prior)
                # Generate images from found latents and save them
                generated_images = generator.generate_images()
                generated_latents = generator.get_latents()
                for img_array, latent, img_name in zip(generated_images, generated_latents, names):
                    img = PIL.Image.fromarray(img_array, 'RGB')
                    img.save(os.path.join(generated_images_dir, '{}_{}.png'.format(img_name, i)), 'PNG')
                    np.save(os.path.join(latent_dir, '{}_{}.npy'.format(img_name, i)), latent)

                generator.reset_latents()


if __name__ == "__main__":
    main()
