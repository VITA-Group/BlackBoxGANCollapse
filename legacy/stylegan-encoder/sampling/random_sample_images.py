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

url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

_Gs_cache = dict()

def load_Gs(model_path):
    #pkl = os.listdir(model_path)[0]
    pkl = 'network-snapshot-007107.pkl'
    with open(os.path.join(model_path, pkl), 'rb') as file:
        print(file)
        G, D, Gs = pickle.load(file)
        return Gs

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def random_sampling(Gs, start, end, id):
    save_dir = 'monte_carlo_sampling_100k'
    images_dir = os.path.join(save_dir, str(id), 'images', '{}_{}'.format(start, end))
    latents_dir = os.path.join(save_dir, str(id), 'latents', '{}_{}'.format(start, end))
    embds_dir = os.path.join(save_dir, str(id), 'embds_pkls')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(latents_dir):
        os.makedirs(latents_dir)
    if not os.path.exists(embds_dir):
        os.makedirs(embds_dir)
    batch_size = 8
    for i in range(start, end, batch_size):
        #latents = np.random.RandomState(seed).randn(batch_size, Gs.input_shape[1])
        latents = np.random.randn(batch_size, Gs.input_shape[1])
        images = Gs.run(latents, None, **synthesis_kwargs)
        for j in range(batch_size):
            image = PIL.Image.fromarray(images[j], 'RGB')
            #image = image.crop((112, 112, 912, 912))
            #image = image.resize((112, 112), PIL.Image.ANTIALIAS)
            image.save(os.path.join(images_dir, '{}.png'.format(i+j)), 'PNG')
            np.save(os.path.join(latents_dir, '{}.npy'.format(i+j)), latents[j])

def main():
    parser = argparse.ArgumentParser(description='Random sampling')
    parser.add_argument('--start', required=True, help='Starting index', type=int)
    parser.add_argument('--end', required=True, help='Ending index', type=int)
    parser.add_argument('--resolution', required=True, help='Directory for trained model', type=int)
    args, other_args = parser.parse_known_args()

    tflib.init_tf()
    random_sampling(load_Gs('../results_old/00022-sgan-celebahq-4gpu_none_128'), args.start, args.end, args.resolution)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
