import os,sys
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

# _Gs_cache = dict()
#
# def load_Gs(url):
#     if url not in _Gs_cache:
#         with dnnlib.util.open_url(url, cache_dir='../cache') as f:
#             _G, _D, Gs = pickle.load(f)
#         _Gs_cache[url] = Gs
#     return _Gs_cache[url]

def load_Gs(model_path):
    #pkl = os.listdir(model_path)[0]
    with open(model_path, 'rb') as file:
        print(file)
        G, D, Gs = pickle.load(file)
        return Gs


def generate_from_latent(Gs):
    read_path = 'monte_carlo_sampling_10m_celebahq/neighbors/0.2/30576/collision/neighbors_latents'
    latents = [os.path.join(read_path, latent) for latent in os.listdir(read_path)]
    save_path = 'monte_carlo_sampling_10m_celebahq/neighbors/0.2/30576/collision/neighbors_images'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(latents)):
        latent = np.load(latents[i])
        print(latent.shape)
        image = Gs.run(np.expand_dims(latent, axis=0), None, **synthesis_kwargs)
        print(image.shape)
        image = np.squeeze(image)
        image = PIL.Image.fromarray(image, 'RGB')
        dst = os.path.join(save_path, '{}.png'.format(latents[i].split('/')[-1][:-4]))
        print(dst)
        image.save(dst, 'PNG')

def main():
    tflib.init_tf()
    generate_from_latent(load_Gs('/mnt/ilcompf5d1/user/zwu/stylegan/cache/a95ced7481975ccbe1308482d17696dc_karras2019stylegan-celebahq-1024x1024.pkl'))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
