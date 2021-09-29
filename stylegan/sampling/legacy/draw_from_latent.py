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
from utils import url_ffhq, url_celebahq, load_Gs

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))


def generate_from_latent(Gs):
    read_path = 'monte_carlo_sampling_1m_128_imbalanced_6_finetune/neighbors/0.25/clustered_latents'
    latents = [os.path.join(read_path, latent) for latent in os.listdir(read_path)]
    save_path = 'monte_carlo_sampling_1m_128_imbalanced_6_finetune/neighbors/0.25/clustered_images'
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
    generate_from_latent(load_Gs(os.path.join('../mode_collapse_detection', '128', 'imbalanced', '6_finetune')))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
