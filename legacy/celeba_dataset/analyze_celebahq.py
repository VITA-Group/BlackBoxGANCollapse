import pickle
import numpy as np
import pickle
import itertools
import os
import math
from sklearn.preprocessing import normalize
import re
import glob
from operator import add
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline


def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(1))

def plot_idx_distribution():
    iml_idx_dict = {}
    with open('celeba/identity_CelebA.txt', 'r') as handle:
        for line in handle:
            iml, idx = line.strip().split()
            iml_idx_dict[int(iml[:-4])] = int(idx)

    imh_idx_dict = {}
    with open('celeba/image_list.txt', 'r') as handle:
        next(handle)
        for line in handle:
            imh, _, iml, _, _ = line.strip().split()
            imh_idx_dict[int(imh)] = iml_idx_dict[int(iml[:-4])]

    print(imh_idx_dict)

    flipped = {}
    for k,v in imh_idx_dict.items():
        if v not in flipped:
            flipped[v] = [k]
        else:
            flipped[v].append(k)

    count_duplicate_lst = []
    for v in flipped.values():
        count_duplicate_lst.append(len(v))
    count_duplicate_lst.sort()
    print(count_duplicate_lst)

    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    bins = np.linspace(0, 30, 30)
    plt.hist([count_duplicate_lst], normed=True,
             histtype='step', cumulative=False, bins=bins, color=['b'], label=['# repeated identities'])
    plt.legend(loc='upper right')
    plt.ylabel('Prob');
    plt.xlabel('Count')
    plt.savefig('distribution.png')  # should before plt.show method
    return flipped


def merge_files_idx(flipped):
    import shutil
    src_dir = 'img_resized'
    for idx, ims in flipped.items():
        dst_dir = os.path.join(src_dir, str(idx))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for im in ims:
            print('{:6d}.png'.format(im))
            shutil.move(os.path.join(src_dir, '{:06d}.png'.format(im)), os.path.join(dst_dir, '{:06d}.png'.format(im)))

flipped = plot_idx_distribution()
merge_files_idx(flipped)
