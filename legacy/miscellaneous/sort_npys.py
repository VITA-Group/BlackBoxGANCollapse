import os,string
from shutil import copyfile

path = './random_sampling'
path = os.path.normpath(path)
res = []
print(os.path.sep)

for root,dirs,files in os.walk(path, topdown=True):
    depth = root[len(path) + len(os.path.sep):].count(os.path.sep)
    if depth == 2:
        print(root)
        print(dirs)
        for f in files:
            if f.endswith('.npy'):
                print(f)
                ind = int(f[:-4])
                start = int(ind // 10000 * 10000)
                end = int((ind // 10000 + 1)*10000)
                src_dir = os.path.join(root, f)
                dst_dir = os.path.join('/mnt/ilcompf5d1/user/zwu/npy', '{}_{}'.format(start,end))
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                dst_dir = os.path.join(dst_dir, f)
                copyfile(src_dir, dst_dir)
        print('################################################################')
        # We're currently two directories in, so all subdirs have depth 3
        dirs[:] = [] # Don't recurse any deepersor
