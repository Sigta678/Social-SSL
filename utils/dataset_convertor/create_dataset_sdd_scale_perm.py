import os
import numpy as np
from tqdm import tqdm
from parse_utils_sdd_scale_perm import SDDParser, create_dataset, scale_npz, merge_npz, calculate_neighbor_distance

dataset_dir = '/home/mrfish/CVPR2021/scale_perm_dataset/sdd_dataset/deathCircle1'


print('Processing - sdd_{}'.format(dataset_dir.split('/')[-1]))
for dataset in tqdm(os.listdir(dataset_dir)):
    if '.txt' not in dataset or 'log.txt' in dataset:
        continue
    annot_file = os.path.join(dataset_dir, dataset)
    npz_out_file = os.path.join(dataset_dir, dataset.replace('.txt', '.npz'))

    parser = SDDParser()
    parser.load(annot_file)

    obsvs, preds, times, batches = create_dataset(parser.p_data,
                                                parser.t_data,
                                                range(parser.t_data[0][0], parser.t_data[-1][-1], parser.interval),
                                                )

    np.savez(npz_out_file, obsvs=obsvs, preds=preds, times=times, batches=batches)


calculate_neighbor_distance(dataset_dir)
scale_npz(dataset_dir)


