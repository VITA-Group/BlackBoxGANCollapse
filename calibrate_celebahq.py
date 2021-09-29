#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pickle

def get_idx_mapping():
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
    
    return imh_idx_dict


# In[10]:


imh_idx_dict = get_idx_mapping()
from collections import Counter
counter = Counter(list(imh_idx_dict.values()))
#print(imh_idx_dict)
print(max(list(counter.values())))
print(min(list(counter.values())))


# In[ ]:


fields = dict()
fields['identity'] = []
for i in range(30000):
    fields['identity'].append(imh_idx_dict[i])


# In[118]:


print(len(set(imh_idx_dict.values())))


# In[87]:


with open('Age-Gender-Estimate-TF/fname_gender_dict.pkl', 'rb') as handle:
    fname_gender_dict = pickle.load(handle)
with open('Age-Gender-Estimate-TF/fname_age_dict.pkl', 'rb') as handle:
    fname_age_dict = pickle.load(handle)
with open('face-classification/fname_race_dict.pkl', 'rb') as handle:
    fname_race_dict = pickle.load(handle)


# In[89]:


import collections
race_list = list(fname_race_dict.values())
race_counter=collections.Counter(race_list)
print(race_counter)
gender_list = list(fname_gender_dict.values())
gender_counter=collections.Counter(gender_list)
print(gender_counter)
age_list = list(fname_age_dict.values())
age_counter=collections.Counter(age_list)
print(age_counter)


# In[90]:


fields['age'] = []
fields['gender'] = []
fields['race'] = []
white_count, black_count, asian_count = 0, 0, 0
for i in range(30000):
    fields['age'].append(fname_age_dict['../celebahq_resized/{:06d}.png'.format(i)])
    fields['gender'].append(fname_gender_dict['../celebahq_resized/{:06d}.png'.format(i)])
    fields['race'].append(fname_race_dict['../celebahq_resized/{:06d}.png'.format(i)])


# In[ ]:


with open(os.path.join('celeba', 'image_list.txt'), 'rt') as file:
    lines = [line.split() for line in file]
    for idx, field in enumerate(lines[0]):
        print(idx)
        print(field)
        type = int if field.endswith('idx') else str
        fields[field] = [type(line[idx]) for line in lines[1:]]
indices = np.array(fields['idx'])


# In[110]:


female_white_count, female_black_count, female_asian_count = 0,0,0
female_white_ids, female_black_ids, female_asian_ids = [],[],[]
male_white_count, male_black_count, male_asian_count = 0,0,0
male_white_ids, male_black_ids, male_asian_ids = [],[],[]
for i in range(30000):
    if fields['gender'][i] == 'female':
        if fields['race'][i] == 'White':
            female_white_count += 1
            female_white_ids.append(fields['identity'][i])
        elif fields['race'][i] == 'Asian':
            female_asian_count += 1
            female_asian_ids.append(fields['identity'][i])
        else:
            female_black_count += 1
            female_black_ids.append(fields['identity'][i])
    else:
        if fields['race'][i] == 'White':
            male_white_count += 1
            male_white_ids.append(fields['identity'][i])
        elif fields['race'][i] == 'Asian':
            male_asian_count += 1
            male_asian_ids.append(fields['identity'][i])
        else:
            male_black_count += 1
            male_black_ids.append(fields['identity'][i])
print(female_white_count, female_black_count, female_asian_count)
print(len(set(female_white_ids)), len(set(female_black_ids)), len(set(female_asian_ids)))
print(male_white_count, male_black_count, male_asian_count)
print(len(set(male_white_ids)), len(set(male_black_ids)), len(set(male_asian_ids)))


# In[111]:


print(female_white_count, male_white_count)
print(female_black_count, male_black_count)
print(female_asian_count, male_asian_count)


# In[109]:





# In[98]:


white_count, asian_count, black_count = 0, 0, 0
with open('image_list_with_fairness_attributes_race_balanced.txt', 'w') as f:
    f.write('{:<8s} {:<8s} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s}\n'.format(
            'idx', 'orig_idx', 'orig_file', 'proc_md5', 'final_md5', 'age', 'gender', 'race'))
    for i in range(30000):
        if fields['race'][i] == 'White':
            white_count += 1
            if white_count <= 1730:
                f.write('{:<8d} {:<8d} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8d}\n'.format(
                    fields['idx'][i], fields['orig_idx'][i], fields['orig_file'][i], fields['proc_md5'][i], fields['final_md5'][i], fields['age'][i], fields['gender'][i], fields['race'][i], fields['identity'][i]))
        elif fields['race'][i] == 'Asian':
            asian_count += 1
            if asian_count <= 1730:
                f.write('{:<8d} {:<8d} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8d}\n'.format(
                    fields['idx'][i], fields['orig_idx'][i], fields['orig_file'][i], fields['proc_md5'][i], fields['final_md5'][i], fields['age'][i], fields['gender'][i], fields['race'][i], fields['identity'][i]))
        else:
            black_count += 1
            if black_count <= 1730:
                f.write('{:<8d} {:<8d} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8d}\n'.format(
                    fields['idx'][i], fields['orig_idx'][i], fields['orig_file'][i], fields['proc_md5'][i], fields['final_md5'][i], fields['age'][i], fields['gender'][i], fields['race'][i], fields['identity'][i]))


# In[102]:


young_count, old_count = 0, 0
with open('image_list_with_fairness_attributes_age_balanced.txt', 'w') as f:
    f.write('{:<8s} {:<8s} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s}\n'.format(
            'idx', 'orig_idx', 'orig_file', 'proc_md5', 'final_md5', 'age', 'gender', 'race'))
    for i in range(30000):
        age = fields['age'][i]
        if age == '0-10' or age == '10-20' or age == '20-30':
            young_count += 1
            if young_count <= 13122:
                f.write('{:<8d} {:<8d} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8d}\n'.format(
                    fields['idx'][i], fields['orig_idx'][i], fields['orig_file'][i], fields['proc_md5'][i], fields['final_md5'][i], fields['age'][i], fields['gender'][i], fields['race'][i], fields['identity'][i]))
        else:
            old_count += 1
            if old_count <= 13122:
                f.write('{:<8d} {:<8d} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8d}\n'.format(
                    fields['idx'][i], fields['orig_idx'][i], fields['orig_file'][i], fields['proc_md5'][i], fields['final_md5'][i], fields['age'][i], fields['gender'][i], fields['race'][i], fields['identity'][i]))


# In[99]:


male_count, female_count = 0, 0
with open('image_list_with_fairness_attributes_gender_balanced.txt', 'w') as f:
    f.write('{:<8s} {:<8s} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s}\n'.format(
            'idx', 'orig_idx', 'orig_file', 'proc_md5', 'final_md5', 'age', 'gender', 'race'))
    for i in range(30000):
        if fields['gender'][i] == 'male':
            male_count += 1
            if male_count <= 10660:
                f.write('{:<8d} {:<8d} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8d}\n'.format(
                    fields['idx'][i], fields['orig_idx'][i], fields['orig_file'][i], fields['proc_md5'][i], fields['final_md5'][i], fields['age'][i], fields['gender'][i], fields['race'][i], fields['identity'][i]))
        else:
            female_count += 1
            if female_count <= 10660:
                f.write('{:<8d} {:<8d} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8d}\n'.format(
                    fields['idx'][i], fields['orig_idx'][i], fields['orig_file'][i], fields['proc_md5'][i], fields['final_md5'][i], fields['age'][i], fields['gender'][i], fields['race'][i], fields['identity'][i]))


# In[103]:


imh_idx_set = set(imh_idx_dict.values())
print(len(imh_idx_set))


# In[ ]:


with open('image_list_with_fairness_attributes_identity_balanced.txt', 'w') as f:
    f.write('{:<8s} {:<8s} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8s}\n'.format(
            'idx', 'orig_idx', 'orig_file', 'proc_md5', 'final_md5', 'age', 'gender', 'race', 'identity'))
    for i in range(30000):
        if fields['identity'][i] in imh_idx_set:
            f.write('{:<8d} {:<8d} {:<10s} {:<32s} {:<32s} {:<8s} {:<8s} {:<8s} {:<8d}\n'.format(
                fields['idx'][i], fields['orig_idx'][i], fields['orig_file'][i], fields['proc_md5'][i], fields['final_md5'][i], fields['age'][i], fields['gender'][i], fields['race'][i], fields['identity'][i]))
            imh_idx_set.remove(fields['identity'][i])

