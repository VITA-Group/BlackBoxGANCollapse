#!/bin/bash
for ((i=0; i<1; i++))
do
	start=$(($i*10000+50000))
	end=$(($i*10000+10000+50000))
	echo "$start"
	echo "$end"
cd /mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling
CUDA_VISIBLE_DEVICES=0 python random_sample_images.py --start=${start} --end=${end} --resolution 128
cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow
CUDA_VISIBLE_DEVICES=0 python get_embd.py --config_path="configs/config_ms1m_100.yaml" --model_path="pretrained/config_ms1m_100_334k/best-m-334000" --read_path="/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/128/images/${start}_${end}" --save_path="/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/128/embds_pkls/${start}_${end}.pkl"
done
#!/bin/bash
for ((i=0; i<1; i++))
do
	start=$(($i*10000+50000))
	end=$(($i*10000+10000+50000))
	echo "$start"
	echo "$end"
cd /mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling
CUDA_VISIBLE_DEVICES=0 python random_sample_images.py --start=${start} --end=${end} --resolution 256
cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow
CUDA_VISIBLE_DEVICES=0 python get_embd.py --config_path="configs/config_ms1m_100.yaml" --model_path="pretrained/config_ms1m_100_334k/best-m-334000" --read_path="/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/128/images/${start}_${end}" --save_path="/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/128/embds_pkls/${start}_${end}.pkl"
done
#!/bin/bash
for ((i=0; i<1; i++))
do
	start=$(($i*10000+50000))
	end=$(($i*10000+10000+50000))
	echo "$start"
	echo "$end"
cd /mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling
CUDA_VISIBLE_DEVICES=0 python random_sample_images.py --start=${start} --end=${end} --resolution 512
cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow
CUDA_VISIBLE_DEVICES=0 python get_embd.py --config_path="configs/config_ms1m_100.yaml" --model_path="pretrained/config_ms1m_100_334k/best-m-334000" --read_path="/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/128/images/${start}_${end}" --save_path="/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/128/embds_pkls/${start}_${end}.pkl"
done
#!/bin/bash
for ((i=0; i<1; i++))
do
	start=$(($i*10000+50000))
	end=$(($i*10000+10000+50000))
	echo "$start"
	echo "$end"
cd /mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling
CUDA_VISIBLE_DEVICES=0 python random_sample_images.py --start=${start} --end=${end} --resolution 128
cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow
CUDA_VISIBLE_DEVICES=0 python get_embd.py --config_path="configs/config_ms1m_100.yaml" --model_path="pretrained/config_ms1m_100_334k/best-m-334000" --read_path="/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/128/images/${start}_${end}" --save_path="/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/128/embds_pkls/${start}_${end}.pkl"
done
#!/bin/bash
for ((i=0; i<1; i++))
do
	start=$(($i*10000+50000))
	end=$(($i*10000+10000+50000))
	echo "$start"
	echo "$end"
cd /mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling
CUDA_VISIBLE_DEVICES=0 python random_sample_images.py --start=${start} --end=${end} --resolution 256
cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow
CUDA_VISIBLE_DEVICES=0 python get_embd.py --config_path="configs/config_ms1m_100.yaml" --model_path="pretrained/config_ms1m_100_334k/best-m-334000" --read_path="/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/256/images/${start}_${end}" --save_path="/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/256/embds_pkls/${start}_${end}.pkl"
done
#!/bin/bash
for ((i=0; i<1; i++))
do
	start=$(($i*10000+50000))
	end=$(($i*10000+10000+50000))
	echo "$start"
	echo "$end"
cd /mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling
CUDA_VISIBLE_DEVICES=0 python random_sample_images.py --start=${start} --end=${end} --resolution 512
cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow
CUDA_VISIBLE_DEVICES=0 python get_embd.py --config_path="configs/config_ms1m_100.yaml" --model_path="pretrained/config_ms1m_100_334k/best-m-334000" --read_path="/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/512/images/${start}_${end}" --save_path="/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/512/embds_pkls/${start}_${end}.pkl"
done
