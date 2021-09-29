#!/usr/bin/env python
# coding: utf-8

# In[31]:


import re
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import math
import os
from collections import defaultdict
from matplotlib.patches import Rectangle
from matplotlib.patches import Rectangle
import numpy as np
import pylab as P
import random


# In[242]:


def compute_exp(path):
    regex = re.compile(r"\d+:(\d+):\s+(\d+\.\d+):(\d+\.\d+)")
    obs_lst = []
    ref_lst = []


    with open(os.path.join(path, 'monte_carlo_sampling_obs.txt')) as f:
        for line in f:
            line = line.rstrip('\n')
            #print(line)
            r = re.search(regex, line)
            #print(r)
            if r is not None:
                obs_lst.append(float(r.group(3)))
            else:
                print(line)
    with open(os.path.join(path, 'monte_carlo_sampling_ref.txt')) as f:
        for line in f:
            line = line.rstrip('\n')
            r = re.search(regex, line)
            if r is not None:
                ref_lst.append(float(r.group(3)))
            else:
                print(line)
    #print(random_count_lst_dct)

    return np.asarray(obs_lst), np.asarray(ref_lst)


# In[176]:


def plot_box_plt(yobs, yref, fig_name):
    d_lst = list(pl.frange(0.1,0.5,0.01))
    eps = 0.05 #controls amount of jitter
    xobs = [random.uniform(1-eps,1+eps) for i in range(0,yobs.shape[0])]
    xref = [random.uniform(2-eps,2+eps) for i in range(0,yref.shape[0])]
    box_data = [yobs, yref]
    
    plt.plot(xobs, yobs, 'ro', label=r'$\mathcal{R}_{obs}$')
    plt.plot(xref, yref, 'bo', label=r'$\mathcal{R}_{ref}$')
    xnames = [r'$\mathcal{R}_{obs}$', r'$\mathcal{R}_{ref}$']
    plt.boxplot(box_data,labels=xnames,sym="") #dont show outliers
    from matplotlib import rcParams
    labelsize = 24
    rcParams['xtick.labelsize'] = labelsize
    #plt.yticks([])
    plt.yticks(np.arange(0.1, 0.4, 0.05))
    plt.yticks([])
    plt.legend(loc='center left', shadow=True, facecolor='white', framealpha=1, prop={'size': 16})
    plt.savefig(fig_name, bbox_inches='tight', dpi =800, pad_inches=0)
    plt.show() # render pipeline
    plt.close()


# In[150]:


yobs, yref = compute_exp('legacy/sgan/monte_carlo_sampling_10m_celebahq/monte_carlo_sampling')
print(yobs)
plot_box_plt(yobs, yref, 'boxplot_SGAN_CelebAHQ_1024.pdf')


# In[177]:


yobs, yref = compute_exp('legacy/sgan/monte_carlo_sampling_10m_ffhq/monte_carlo_sampling')
print(yobs)
yref = list(yref)
yref.remove(max(yref))
yref.remove(max(yref))
plot_box_plt(yobs, np.asarray(yref), 'boxplot_SGAN_FFHQ_1024.pdf')


# In[152]:


yobs, yref = compute_exp('legacy/pggan/monte_carlo_sampling_10m_ffhq/monte_carlo_sampling')
print(yobs)
plot_box_plt(yobs, yref, 'boxplot_PGGAN_FFHQ_1024.pdf')


# In[154]:


yobs, yref = compute_exp('legacy/pggan/monte_carlo_sampling_10m_celebahq/monte_carlo_sampling')
print(yobs)
plot_box_plt(yobs, yref, 'boxplot_PGGAN_CelebAHQ_1024.pdf')


# In[ ]:


import random


# In[169]:


yobs, yref = compute_exp('legacy/sgan/monte_carlo_sampling_1m_finetune/monte_carlo_sampling')

plot_box_plt(yobs[random.sample(range(50), 8)], yref, 'boxplot_SGAN_Finetune_128.pdf')


# In[173]:


yobs, yref = compute_exp('legacy/sgan/monte_carlo_sampling_1m_randomness/monte_carlo_sampling')

plot_box_plt(yobs[random.sample(range(100), 5)], yref, 'boxplot_SGAN_Randomness_128.pdf')


# In[159]:


yobs, yref = compute_exp('legacy/sgan/monte_carlo_sampling_1m_sgan_architecture/monte_carlo_sampling')
print(yobs)
plot_box_plt(yobs[random.sample(range(100), 4)], yref, 'boxplot_SGAN_Architecture_128.pdf')


# In[ ]:


yobs, yref = compute_exp('legacy/sgan/monte_carlo_sampling_1m_pggan_architecture/monte_carlo_sampling')
print(yobs)
yref = list(yref)
yref.remove(max(yref))

plot_box_plt(yobs[random.sample(range(100), 4)], np.asarray(yref), 'boxplot_PGGAN_Architecture_128.pdf')


# In[236]:


def plot_2box_plt(yobs1, yref1, yobs2, yref2, fig_name):
    d_lst = list(pl.frange(0.1,0.5,0.01))
    eps = 0.05 #controls amount of jitter
    xobs1 = [random.uniform(-0.4-eps,-0.4+eps) for i in range(0,yobs1.shape[0])]
    xref1 = [random.uniform(0.4-eps,0.4+eps) for i in range(0,yref1.shape[0])]
    box_data1 = [yobs1, yobs2]
    
    xobs2 = [random.uniform(1.6-eps,1.6+eps) for i in range(0,yobs2.shape[0])]
    xref2 = [random.uniform(2.4-eps,2.4+eps) for i in range(0,yref2.shape[0])]
    box_data2 = [yref1, yref2]
    
    plt.plot(xobs1, yobs1, 'ro', label=r'$\mathcal{R}_{obs}$')
    plt.plot(xref1, yref1, 'bo', label=r'$\mathcal{R}_{ref}$')
    
    plt.plot(xobs2, yobs2, 'ro')
    plt.plot(xref2, yref2, 'bo')
    
    xnames = ['Before Calibration', 'After Calibration']
    #plt.boxplot(box_data1,positions = [-0.4, 1.6],labels=xnames,sym="", widths=0.6)
    #plt.boxplot(box_data2,positions = [0.4, 2.4],labels=xnames,sym="", widths=0.6)
    plt.boxplot(box_data1,positions = [-0.4, 1.6],sym="", widths=0.6)
    plt.boxplot(box_data2,positions = [0.4, 2.4],sym="", widths=0.6)
    
    plt.xticks(range(0, len(xnames) * 2, 2), xnames)
    plt.xlim(-2, len(xnames)*2)
    #plt.ylim(0, 8)
    plt.tight_layout()
    from matplotlib import rcParams
    labelsize = 12
    rcParams['xtick.labelsize'] = labelsize
    plt.yticks(np.arange(0.1, 0.4, 0.05))
    #plt.yticks([])
    plt.legend(loc='center left', shadow=True, facecolor='white', framealpha=1, prop={'size': 16})
    plt.savefig(fig_name, bbox_inches='tight', dpi =800, pad_inches=0)
    plt.show() # render pipeline
    plt.close()


# In[260]:


yobs1, yref1 = compute_exp('legacy/pggan/monte_carlo_sampling_10m_celebahq/monte_carlo_sampling')
yobs2, yref2 = compute_exp('legacy/pggan/monte_carlo_sampling_10m_ffhq/monte_carlo_sampling')
print(yobs)
plot_2box_plt(np.asarray([np.min(yobs2)]), yref2, np.asarray([np.max(yobs1)]), yref1, 'boxplot_GMM_reshaping.pdf')


# In[261]:


yobs1, yref1 = compute_exp('legacy/pggan/monte_carlo_sampling_10m_ffhq/monte_carlo_sampling')
yobs2, yref2 = compute_exp('legacy/pggan/monte_carlo_sampling_10m_ffhq/monte_carlo_sampling_old')
print(yobs)
plot_2box_plt(np.asarray([np.min(yobs2)]), yref2, np.asarray([np.mean(yobs1)]), yref1, 'boxplot_IS_reshaping.pdf')

