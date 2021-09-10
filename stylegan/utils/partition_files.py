# coding: utf-8

from os import listdir,environ
from sys import argv
from os.path import isfile,join
import os
import re

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+).png', text) ]

cwd = os.getcwd()
src_dir = 'celebahq-raw'
dst_dir = 'celebahq'
group_size = 100000

filenames = []
images_dir = os.path.join(cwd, src_dir)

for path, subdirs, files in os.walk(images_dir):
    for name in files:
        filenames.append(name)

filenames.sort(key=natural_keys)

start = 0
end = len(filenames)

import shutil
for i in range(start, end, group_size):
    new_group_dir = os.path.join(cwd, dst_dir, '{}_{}'.format(i, i+group_size))
    if not os.path.exists(new_group_dir):
        os.makedirs(new_group_dir)
    for j in range(group_size):
        shutil.move(os.path.join(cwd, src_dir, filenames[i+j]), os.path.join(new_group_dir, filenames[i+j]))