import pickle
with open ('sampling/stylegan_final_neighbors_count_lstoflst_{}.pkl'.format(0.2), 'rb') as fp:
    stylegan_neighbors_count = pickle.load(fp)[99]

import pickle
with open ('pggan_final_neighbors_count_lstoflst_{}.pkl'.format(0.2), 'rb') as fp:
    pggan_neighbors_count = pickle.load(fp)[99]

stylegan_neighbors_count = [count/1e6 for count in stylegan_neighbors_count]
pggan_neighbors_count = [count/1e6 for count in pggan_neighbors_count]

import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
#plt.style.use('seaborn-deep')
bins = np.linspace(0, 0.00005, 10000)
plt.hist([stylegan_neighbors_count,pggan_neighbors_count], normed=True,
         histtype='step', cumulative=True, bins=bins, color=['r','b'], label=['StyleGAN', 'PGGAN'])
plt.legend(loc='upper right')
plt.ylabel('Prob');
plt.xlabel('Count of neighbors ' +'$(\epsilon = 0.2)$' + ' / Population')
plt.savefig('gan_neighbors_count_comp.png', dpi=600)
