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
from scipy.spatial import Delaunay
import random
from scipy.optimize import linprog
from scipy.spatial import distance

url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

_Gs_cache = dict()

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

def latent_lerp(z0, z1):
    """Interpolate between two images in latent space"""
    z = (1 - 0.5) * z0 + 0.5 * z1
    return z

from random import randint

def gencoordinates(m, n):
    seen = set()
    #outlier = set([5,6,7,17,21,26,28,39,45,46,49])
    x, y = randint(m, n), randint(m, n)

    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(m, n), randint(m, n)
        #while (x, y) in seen or x in outlier or y in outlier:
        while (x, y) in seen:
            x, y = randint(m, n), randint(m, n)

def draw_from_convex_hull(hull):
    # coef = np.random.dirichlet(np.ones(hull.shape[0])/1000, size=1)
    # # print(np.sum(coef))
    zero_vec = np.ones(hull.shape[0])
    # # print(np.argmax(coef))
    #zero_vec[random.randint(0, hull.shape[0]-1)] = 1
    coef = zero_vec
    # coef = np.random.random(hull.shape[0])
    coef /= np.sum(coef)
    #print(coef)
    coef = np.tile(coef.reshape([hull.shape[0],1]), (1, hull.shape[1]))
    #print(coef.shape)
    #print(hull.shape)
    #print(coef)
    latent = np.sum(np.multiply(hull, coef), axis=0)
    #print(latent.shape)
    #print(latent)
    return latent

# def in_hull(p, hull):
#     """
#     Test if points in `p` are in `hull`
#
#     `p` should be a `NxK` coordinates of `N` points in `K` dimensions
#     `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
#     coordinates of `M` points in `K`dimensions for which Delaunay triangulation
#     will be computed
#     """
#     if not isinstance(hull,Delaunay):
#         hull = Delaunay(hull)
#
#     return hull.find_simplex(p)>=0

def in_sphere(point, center, radius):
    #print(distance.cdist(center, point, 'euclidean'), radius)
    return distance.cdist(center, point, 'euclidean') < radius

def create_sphere(points):
    center = np.mean(points, axis=0)[np.newaxis,:]
    radius = np.min(distance.cdist(center, points, 'euclidean'))
    return radius, center

def generate_from_npy():
    def grep(pat, txt, ind):
        r = re.search(pat, txt)
        return int(r.group(ind))

    save_dir = 'monte_carlo_sampling_10m_celebahq/neighbors/0.2/286667/collision/convex_hull'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    latents_dir = 'monte_carlo_sampling_10m_celebahq/neighbors/0.2/286667/collision/neighbors_latents'
    latents_files = [os.path.join(latents_dir, latent) for latent in os.listdir(latents_dir)]
    print('start loading')
    latents_mat = np.stack([np.load(file) for file in latents_files[:1500]], axis=0)
    radius, center = create_sphere(latents_mat)
    print(radius)
    print('end loading')
    test_latents = np.stack([np.load(file) for file in latents_files[1500:]], axis=0)
    #latents_mat = np.random.rand(29,512)
    #print(latents_mat.shape)
    count = 0
    for i in range(1000000):
        #print(i)
        #latent = draw_from_convex_hull(latents_mat)
        #latent = test_latents[i][np.newaxis,:]
        latent = np.random.randn(1, 512)
        #if in_hull(latent[np.newaxis,:], latents_mat):
        # print(latent.shape)
        # print(center.shape)
        if in_sphere(latent, center, radius):
            print(i)
            #print(latent)
            print('In convex hull')
            print(count)
            count += 1
            #image = Gs.run(np.expand_dims(latent, axis=0), None, **synthesis_kwargs)
            # image = Gs.run(latent, None, **synthesis_kwargs)
            # #print(image.shape)
            # image = np.squeeze(image)
            # image = PIL.Image.fromarray(image, 'RGB')
            # dst = os.path.join(save_dir, '{}.png'.format(i))
            # #print(dst)
            # image.save(dst, 'PNG')
def main():
    tflib.init_tf()
    #generate_from_npy(load_Gs('/mnt/ilcompf5d1/user/zwu/stylegan/cache/a95ced7481975ccbe1308482d17696dc_karras2019stylegan-celebahq-1024x1024.pkl'))
    generate_from_npy()
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
