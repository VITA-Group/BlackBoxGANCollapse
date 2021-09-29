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
import re

url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

_Gs_cache = dict()

def load_Gs(url):
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir='../cache') as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]

def latent_lerp(z0, z1):
    """Interpolate between two images in latent space"""
    z = (1 - 0.5) * z0 + 0.5 * z1
    return z

from random import randint

def gencoordinates(m, n):
    seen = set()
    outlier = set([5,6,7,17,21,26,28,39,45,46,49])
    x, y = randint(m, n), randint(m, n)

    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(m, n), randint(m, n)
        while (x, y) in seen or x in outlier or y in outlier:
            x, y = randint(m, n), randint(m, n)

def generate_from_npy(Gs):
    def grep(pat, txt, ind):
        r = re.search(pat, txt)
        return int(r.group(ind))

    save_dir = 'monte_carlo_sampling_1m/neighbors/0.2/664058/inverted/lerp_excluding_outliers'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = 'monte_carlo_sampling_1m/neighbors/0.2/664058/inverted/inverted_latents'
    npys = [os.path.join(path, npy) for npy in os.listdir(path)]
    npys.sort(key=lambda txt: grep(r"(\d+)_(\d+)_(\d+)\.npy", txt, 3))
    print(npys)
    g = gencoordinates(0, len(npys) - 1)
    for i in range(50):
        m, n = next(g)
        latent1 = np.load(npys[m])
        latent2 = np.load(npys[n])
        latent = latent_lerp(latent1, latent2)
        image = Gs.run(np.expand_dims(latent, axis=0), None, **synthesis_kwargs)
        print(image.shape)
        image = np.squeeze(image)
        image = PIL.Image.fromarray(image, 'RGB')
        dst = os.path.join(save_dir, '{}.png'.format(i))
        print(dst)
        image.save(dst, 'PNG')

def main():
    tflib.init_tf()
    generate_from_npy(load_Gs(url_celebahq))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
