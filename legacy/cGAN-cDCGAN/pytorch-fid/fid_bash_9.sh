#!/bin/bash
for((i=9; i<=9; i++))
do
	echo "$i"
	CUDA_VISIBLE_DEVICES=3 ./fid_score.py /mnt/ilcompf5d1/user/zwu/pytorch-MNIST-CelebA-cGAN-cDCGAN/MNIST_cDCGAN_eval_results/baseline/$i /mnt/ilcompf5d1/user/zwu/pytorch-MNIST-CelebA-cGAN-cDCGAN/MNIST_cDCGAN_eval_results/remove_untilted_1_balanced/$i
	CUDA_VISIBLE_DEVICES=3 ./fid_score.py /mnt/ilcompf5d1/user/zwu/pytorch-MNIST-CelebA-cGAN-cDCGAN/MNIST_cDCGAN_eval_results/baseline/$i /mnt/ilcompf5d1/user/zwu/pytorch-MNIST-CelebA-cGAN-cDCGAN/MNIST_cDCGAN_eval_results/remove_tilted_1_balanced/$i
	CUDA_VISIBLE_DEVICES=3 ./fid_score.py /mnt/ilcompf5d1/user/zwu/pytorch-MNIST-CelebA-cGAN-cDCGAN/MNIST_cDCGAN_eval_results/baseline/$i /mnt/ilcompf5d1/user/zwu/pytorch-MNIST-CelebA-cGAN-cDCGAN/MNIST_cDCGAN_eval_results/whole_dataset/$i
done