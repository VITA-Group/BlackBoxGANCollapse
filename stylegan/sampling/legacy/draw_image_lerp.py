import os, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib

import re
from utils import url_ffhq, url_celebahq, load_Gs_from_url
from random import randint


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))


def latent_lerp(z0, z1):
    """Interpolate between two images in latent space"""
    z = (1 - 0.5) * z0 + 0.5 * z1
    return z


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
    generate_from_npy(load_Gs_from_url(url_celebahq))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
