import pickle
with open('training_final_neighbors_count_lstoflst_{}.pkl'.format(0.2), 'rb') as fp:
    training_neighbors_count = pickle.load(fp)[2]
with open ('training_sampling/training_sampling_neighbors_count_final_{}.pkl'.format(0.2), 'rb') as fp:
    training_samping_neighbors_count = pickle.load(fp)
with open ('sampling/final_neighbors_count_lstoflst_{}.pkl'.format(0.2), 'rb') as fp:
    samping_neighbors_count = pickle.load(fp)[99]
training_neighbors_count = [count/1e6 for count in training_neighbors_count]
training_samping_neighbors_count = [count/1e6 for count in training_samping_neighbors_count]
samping_neighbors_count = [count/1e6 for count in samping_neighbors_count]
print(min(training_neighbors_count))
print(max(training_neighbors_count))
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
#plt.style.use('seaborn-deep')
bins = np.linspace(0, 0.00005, 10000)
plt.hist([training_neighbors_count,samping_neighbors_count,training_samping_neighbors_count], normed=True,
         histtype='step', cumulative=True, bins=bins, color=['r','g','b'], label=['$T_T$', '$S_S$', '$T_S$'])
plt.legend(loc='upper right')
plt.ylabel('Prob');
plt.xlabel('Count of neighbors ' +'$(\epsilon = 0.2)$' + ' / Population')
plt.savefig('neighbors_count_comp.png', dpi=600)
