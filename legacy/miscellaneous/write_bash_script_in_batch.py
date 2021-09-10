import os
node_gpu_map = {'5m':'3','5n':'0123','5x':'023','72':'1','76':'2'}
#{'70':'34','72':'134','74':'134','76':'24','77':'13'}
node_gpu_lst = []
for k,v in node_gpu_map.items():
    for gpu in v:
        node_gpu_lst.append((k,int(gpu)))

print(node_gpu_lst)

for i in range(10):
    with open('bash_run_{}_{}.sh'.format(node_gpu_lst[i][0],node_gpu_lst[i][1]), 'w') as fp:
        fp.write('#!/bin/bash\n')
        fp.write('for ((i=0; i<1; i++))\n')
        fp.write('do\n')
        fp.write('\tstart=$(($i*10000+{}))\n'.format(i*10000))
        fp.write('\tend=$(($i*10000+10000+{}))\n'.format(i*10000))
        fp.write('\techo \"$start\"\n')
        fp.write('\techo \"$end\"\n')
        fp.write('cd /mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python random_sample_images.py --start=${start} --end=${end} --resolution 128\n" % node_gpu_lst[i][1])
        fp.write('cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python get_embd.py --config_path=\"configs/config_ms1m_100.yaml\" "\
                 "--model_path=\"pretrained/config_ms1m_100_334k/best-m-334000\" "\
                 "--read_path=\"/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/128/images/${start}_${end}\" "\
                 "--save_path=\"/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/128/embds_pkls/${start}_${end}.pkl\"\n" % node_gpu_lst[i][1])
        fp.write('done\n')

        fp.write('#!/bin/bash\n')
        fp.write('for ((i=0; i<1; i++))\n')
        fp.write('do\n')
        fp.write('\tstart=$(($i*10000+{}))\n'.format(i*10000))
        fp.write('\tend=$(($i*10000+10000+{}))\n'.format(i*10000))
        fp.write('\techo \"$start\"\n')
        fp.write('\techo \"$end\"\n')
        fp.write('cd /mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python random_sample_images.py --start=${start} --end=${end} --resolution 256\n" % node_gpu_lst[i][1])
        fp.write('cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python get_embd.py --config_path=\"configs/config_ms1m_100.yaml\" "\
                 "--model_path=\"pretrained/config_ms1m_100_334k/best-m-334000\" "\
                 "--read_path=\"/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/256/images/${start}_${end}\" "\
                 "--save_path=\"/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/256/embds_pkls/${start}_${end}.pkl\"\n" % node_gpu_lst[i][1])
        fp.write('done\n')

        fp.write('#!/bin/bash\n')
        fp.write('for ((i=0; i<1; i++))\n')
        fp.write('do\n')
        fp.write('\tstart=$(($i*10000+{}))\n'.format(i*10000))
        fp.write('\tend=$(($i*10000+10000+{}))\n'.format(i*10000))
        fp.write('\techo \"$start\"\n')
        fp.write('\techo \"$end\"\n')
        fp.write('cd /mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python random_sample_images.py --start=${start} --end=${end} --resolution 512\n" % node_gpu_lst[i][1])
        fp.write('cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python get_embd.py --config_path=\"configs/config_ms1m_100.yaml\" "\
                 "--model_path=\"pretrained/config_ms1m_100_334k/best-m-334000\" "\
                 "--read_path=\"/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/512/images/${start}_${end}\" "\
                 "--save_path=\"/mnt/ilcompf5d1/user/zwu/progressive_growing_of_gans/sampling/monte_carlo_sampling_100k/512/embds_pkls/${start}_${end}.pkl\"\n" % node_gpu_lst[i][1])
        fp.write('done\n')

        fp.write('#!/bin/bash\n')
        fp.write('for ((i=0; i<1; i++))\n')
        fp.write('do\n')
        fp.write('\tstart=$(($i*10000+{}))\n'.format(i*10000))
        fp.write('\tend=$(($i*10000+10000+{}))\n'.format(i*10000))
        fp.write('\techo \"$start\"\n')
        fp.write('\techo \"$end\"\n')
        fp.write('cd /mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python random_sample_images.py --start=${start} --end=${end} --resolution 128\n" % node_gpu_lst[i][1])
        fp.write('cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python get_embd.py --config_path=\"configs/config_ms1m_100.yaml\" "\
                 "--model_path=\"pretrained/config_ms1m_100_334k/best-m-334000\" "\
                 "--read_path=\"/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/128/images/${start}_${end}\" "\
                 "--save_path=\"/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/128/embds_pkls/${start}_${end}.pkl\"\n" % node_gpu_lst[i][1])
        fp.write('done\n')

        fp.write('#!/bin/bash\n')
        fp.write('for ((i=0; i<1; i++))\n')
        fp.write('do\n')
        fp.write('\tstart=$(($i*10000+{}))\n'.format(i*10000))
        fp.write('\tend=$(($i*10000+10000+{}))\n'.format(i*10000))
        fp.write('\techo \"$start\"\n')
        fp.write('\techo \"$end\"\n')
        fp.write('cd /mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python random_sample_images.py --start=${start} --end=${end} --resolution 256\n" % node_gpu_lst[i][1])
        fp.write('cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python get_embd.py --config_path=\"configs/config_ms1m_100.yaml\" "\
                 "--model_path=\"pretrained/config_ms1m_100_334k/best-m-334000\" "\
                 "--read_path=\"/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/256/images/${start}_${end}\" "\
                 "--save_path=\"/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/256/embds_pkls/${start}_${end}.pkl\"\n" % node_gpu_lst[i][1])
        fp.write('done\n')

        fp.write('#!/bin/bash\n')
        fp.write('for ((i=0; i<1; i++))\n')
        fp.write('do\n')
        fp.write('\tstart=$(($i*10000+{}))\n'.format(i*10000))
        fp.write('\tend=$(($i*10000+10000+{}))\n'.format(i*10000))
        fp.write('\techo \"$start\"\n')
        fp.write('\techo \"$end\"\n')
        fp.write('cd /mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python random_sample_images.py --start=${start} --end=${end} --resolution 512\n" % node_gpu_lst[i][1])
        fp.write('cd /mnt/ilcompf5d1/user/zwu/InsightFace-tensorflow\n')
        fp.write("CUDA_VISIBLE_DEVICES=%d python get_embd.py --config_path=\"configs/config_ms1m_100.yaml\" "\
                 "--model_path=\"pretrained/config_ms1m_100_334k/best-m-334000\" "\
                 "--read_path=\"/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/512/images/${start}_${end}\" "\
                 "--save_path=\"/mnt/ilcompf5d1/user/zwu/stylegan-encoder/sampling/monte_carlo_sampling_100k/512/embds_pkls/${start}_${end}.pkl\"\n" % node_gpu_lst[i][1])
        fp.write('done\n')
