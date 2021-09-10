import os
import pickle
import re

def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(ind))

path = 'monte_carlo_sampling_1m_128_balanced_age/ages/0_1000000'

pkls = []
for root, dirs, files in os.walk(path):
    if len(files) != 0:
        pkls.extend([os.path.join(root, file) for file in files])
#pkls = os.listdir(path)
pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt, 1))
print(pkls)
sample_lst = []

young_count, old_count = 0, 0
for pkl in pkls:
    print(pkl)
    with open(pkl, 'rb') as handle:
        samples = pickle.load(handle)
        for s in samples:
            age = int(s)
            if age <= 30:
                young_count += 1
            else:
                old_count += 1
print(young_count)
print(old_count)