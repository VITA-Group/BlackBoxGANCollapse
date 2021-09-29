#!/usr/bin/env python
# coding: utf-8

# In[192]:


import pickle
import os
import numpy as np


# In[193]:


def grep(pat, txt):
    r = re.search(pat, txt)
    return int(r.group(1)) * 10 + int(r.group(2))


# In[194]:


pkls = [os.path.join('embds', pkl) for pkl in os.listdir('embds')]
#pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt))
pkls.sort()
obs_embds_lst = []
for pkl in pkls:
    embd = pickle.load(open(pkl, 'rb'))
    obs_embds_lst.append(embd)
    #print(embd)

obs_embds = np.asarray(obs_embds_lst)
print(obs_embds.shape)


# In[195]:


# pkls = [os.path.join('embds_ref', pkl) for pkl in os.listdir('embds_ref')]
# #pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt))
# pkls.sort()
# print(pkls)
# ref_embds_lst = []
# for pkl in pkls:
#     embd = list(pickle.load(open(pkl, 'rb')).values())[0]
#     ref_embds_lst.append(embd)
#     #print(embd)

# ref_embds = np.asarray(ref_embds_lst)
# print(ref_embds.shape)
ref_embds = pickle.load(open('embds.pkl', 'rb'))

keys = list(ref_embds.keys())
#keys.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.png", txt))
keys.sort()
print(keys)
ref_embds_lst = []
for k in keys:
    ref_embds_lst.append(ref_embds[k])
ref_embds = np.asarray(ref_embds_lst)


# In[196]:


print(obs_embds[0])
print(ref_embds[0])


# In[197]:


print(np.dot(obs_embds[0], obs_embds[0]))
print(np.dot(ref_embds[0], ref_embds[0]))


# In[198]:


import math
print(obs_embds.shape)
print(ref_embds.shape)
print(np.arccos(np.matmul(obs_embds, np.transpose(ref_embds))) / math.pi)

