import os
import pickle
import numpy as np
import PIL.Image
import operator
import argparse
import tensorflow as tf
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

def load_Gs(model_path):
    pkl = os.listdir(model_path)[0]
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
        latents = np.random.randn(batch_size, *Gs.input_shapes[0][1:])
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
        images = Gs.run(latents, labels)
        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
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

    # Initialize TensorFlow session.
    tf.InteractiveSession()
    random_sampling(load_Gs(os.path.join('../mode_collapse_detection', str(args.resolution))), args.start, args.end, args.resolution)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()