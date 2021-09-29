import pickle
import os
import numpy as np
import argparse

from utils import compute_embds_matrix



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize nearest neighbors')
    #parser.add_argument('--start', required=True, help='Start of the distance threshold for neighbors', type=float)
    #parser.add_argument('--end', required=True, help='End of the distance threshold for neighbors', type=float)
    #parser.add_argument('--step_size', required=True, help='Step size of the epsilon', type=float)
    parser.add_argument('--path', required=True, help='The path for reading embeddings', type=str)
    parser.add_argument('--epsilon', required=True, help='The epsilon value for neighbors', type=float)

    args, other_args = parser.parse_known_args()

    M = 10000
    N = 5
    #path = os.path.join(args.path, str(args.resolution))
    path = args.path
    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, N)

    with open(os.path.join(path, 'neighbors', 'final_neighbors_count_lstoflst_{}.pkl'.format(args.epsilon)), 'rb') as fp:
        final_neighbors_count_lstoflst = pickle.load(fp)

    final_neighbors_count_lst = final_neighbors_count_lstoflst[N - 1]
    print(max(final_neighbors_count_lst))
    final_neighbors_count_lst = np.asarray(final_neighbors_count_lst)
    #for epsilon in list(np.arange(args.start, args.end, args.step_size)):

    percentile_lst = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for percentile in percentile_lst:
       pos = int(-1 * M * N * percentile)
       print(pos)
       indices = np.argpartition(final_neighbors_count_lst, -1)[pos].tolist()
       print(indices)
       with open(os.path.join(args.path, 'neighbors', 'clustered_indices_{}.pkl'.format(percentile)), 'wb') as handle:
           pickle.dump(indices,handle)
       print(final_neighbors_count_lst[indices])


    # indices = np.argpartition(final_neighbors_count_lst, -1)[0].tolist()
    # print(indices)
    # with open(os.path.join(args.path, 'neighbors', 'clustered_indices_{}.pkl'.format(0)), 'wb') as handle:
    #     pickle.dump(indices,handle)
    # print(final_neighbors_count_lst[indices])