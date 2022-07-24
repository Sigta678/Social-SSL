import os
import numpy as np
from tqdm import tqdm
from parse_utils_homo_leave1out_scale_perm import HomoParser, create_dataset, scale_npz, merge_npz, calculate_neighbor_distance

dataset_dir = '/home/mrfish/CVPR2021/scale_perm_dataset/datasets/raw'

for subset in os.listdir(dataset_dir):
    if subset == 'train' or subset == 'val' or subset == 'test' or subset == 'all_data':
        print('Processing - {}_{}'.format(dataset_dir.split('/')[-1], subset))
        for dataset in tqdm(os.listdir(os.path.join(dataset_dir, subset))):
            if '.txt' not in dataset or 'log.txt' in dataset:
                continue
            annot_file = os.path.join(dataset_dir, subset, dataset)
            npz_out_file = os.path.join(dataset_dir, subset, dataset.replace('.txt', '.npz'))

            parser = HomoParser()
            parser.load(annot_file)

            obsvs, preds, times, batches = create_dataset(parser.p_data,
                                                        parser.t_data,
                                                        range(parser.t_data[0][0], parser.t_data[-1][-1], parser.interval),
                                                        )

            np.savez(npz_out_file, obsvs=obsvs, preds=preds, times=times, batches=batches)
    
        # Combine each npz dataset into one
        merge_dir = os.path.join(dataset_dir, subset)
        dir_split = merge_dir.split('/')
        merge_output = os.path.join(merge_dir, dir_split[-2] + '_dist_scale_' + dir_split[-1] + '.npz')
        calculate_neighbor_distance(merge_dir)
        scale_npz(merge_dir)
        merge_npz(merge_dir, merge_output)


