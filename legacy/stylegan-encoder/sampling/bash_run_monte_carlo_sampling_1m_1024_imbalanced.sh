#!/bin/bash
python compute_neighbors.py --start 0.25 --end 0.25 --step_size 0.01 --path monte_carlo_sampling_1m_1024_imbalanced
python visualize_neighbors.py --start 0.25 --end 0.25 --step_size 0.01 --path monte_carlo_sampling_1m_1024_imbalanced
python ripley.py --start 0.2 --end 0.5 --step_size 0.01 --job_id "collapsed" --sampling_path monte_carlo_sampling_1m_1024_imbalanced --random_anchor False
for ((i=0; i<2; i++))
do
	id=$(($i))
	echo "$id"
	python ripley.py --start 0.2 --end 0.5 --step_size 0.01 --job_id "${id}" --sampling_path monte_carlo_sampling_1m_1024_imbalanced --random_anchor True
done
