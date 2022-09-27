import os
import csv
import math
import sys
import numpy as np
from tqdm import tqdm
from builtins import ValueError
from operator import itemgetter
from itertools import permutations, combinations
from sklearn.preprocessing import MinMaxScaler




class Scale(object):
    '''
    Given max and min of a rectangle it computes the scale and shift values to normalize data to [0,1]
    '''

    def __init__(self):
        self.min_x = +math.inf
        self.max_x = -math.inf
        self.min_y = +math.inf
        self.max_y = -math.inf
        self.sx, self.sy = 1, 1

    def calc_scale(self, keep_ratio=True):
        self.sx = 1 / (self.max_x - self.min_x)
        self.sy = 1 / (self.max_y - self.min_y)
        if keep_ratio:
            if self.sx > self.sy:
                self.sx = self.sy
            else:
                self.sy = self.sx

    def normalize(self, data, shift=True, inPlace=True):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        if data.ndim == 1:
            data_copy[0] = (data[0] - self.min_x * shift) * self.sx
            data_copy[1] = (data[1] - self.min_y * shift) * self.sy
        elif data.ndim == 2:
            data_copy[:, 0] = (data[:, 0] - self.min_x * shift) * self.sx
            data_copy[:, 1] = (data[:, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 3:
            data_copy[:, :, 0] = (data[:, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, 1] = (data[:, :, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 4:
            data_copy[:, :, :, 0] = (data[:, :, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, :, 1] = (data[:, :, :, 1] - self.min_y * shift) * self.sy
        else:
            return False
        return data_copy

    def denormalize(self, data, shift=True, inPlace=False):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        ndim = data.ndim
        if ndim == 1:
            data_copy[0] = data[0] / self.sx + self.min_x * shift
            data_copy[1] = data[1] / self.sy + self.min_y * shift
        elif ndim == 2:
            data_copy[:, 0] = data[:, 0] / self.sx + self.min_x * shift
            data_copy[:, 1] = data[:, 1] / self.sy + self.min_y * shift
        elif ndim == 3:
            data_copy[:, :, 0] = data[:, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, 1] = data[:, :, 1] / self.sy + self.min_y * shift
        elif ndim == 4:
            data_copy[:, :, :, 0] = data[:, :, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, :, 1] = data[:, :, :, 1] / self.sy + self.min_y * shift
        else:
            return False

        return data_copy




class HomoParser:
    def __init__(self):
        self.scale = Scale()
        self.all_ids = list()
        self.all_ts = list()
        self.delimit = '\t'
        self.p_data = []
        self.v_data = []
        self.t_data = []
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = 10

    def load(self, filename, down_sample=1):
        pos_data_dict = dict()
        vel_data_dict = dict()
        time_data_dict = dict()
        self.all_ids.clear()
        self.all_ts.clear()

        file_names = list()
        # If the filename contains *, it will include all files in the directory
        # e.g. ./datasets/eth/train/*
        if '*' in filename:
            files_path = filename[:filename.index('*')]
            extension = filename[filename.index('*') + 1:]
            for file in os.listdir(files_path):
                if file.endswith(extension):
                    file_names.append(files_path + file)
        else:
            file_names.append(filename)

        for file in file_names:
            if not os.path.exists(file):
                raise ValueError("No such file or directory:", file)
            with open(file, 'r') as data_file:
                content = data_file.readlines()
                id_list = list()
                for i, row in enumerate(content):
                    row = row.split(self.delimit)
                    while '' in row: row.remove('')
                    if len(row) < 4: continue

                    ts = float(row[0])
                    self.all_ts.append(ts)

                    id = round(float(row[1]))

                    if ts < self.min_t: self.min_t = ts
                    if ts > self.max_t: self.max_t = ts


                    px = float(row[2])
                    py = float(row[3])

                    if id not in id_list:
                        id_list.append(id)
                        pos_data_dict[id] = list()
                        vel_data_dict[id] = list()
                        time_data_dict[id] = np.empty(0, dtype=int)
                        last_t = ts
                    pos_data_dict[id].append(np.array([px, py, 0, id]))
                    time_data_dict[id] = np.hstack((time_data_dict[id], np.array([ts])))
            self.all_ids += id_list
        self.all_ts = np.unique(self.all_ts)


        for key, value in pos_data_dict.items():
            poss_i = np.array(value)
            self.p_data.append(poss_i)
            self.t_data.append(np.array(time_data_dict[key]).astype(np.int32))

        # calc scale
        for i in range(len(self.p_data)):
            poss_i = np.array(self.p_data[i])
            self.scale.min_x = min(self.scale.min_x, min(poss_i[:, 0]))
            self.scale.max_x = max(self.scale.max_x, max(poss_i[:, 0]))
            self.scale.min_y = min(self.scale.min_y, min(poss_i[:, 1]))
            self.scale.max_y = max(self.scale.max_y, max(poss_i[:, 1]))
        self.scale.calc_scale()




def create_dataset(p_data, t_data, t_range, n_past=8, n_next=12):
    dataset_t0 = []
    dataset_x = []
    dataset_y = []

    for t in range(t_range.start, t_range.stop, 1):
        for i in range(len(t_data)):
            t0_ind = (np.where(t_data[i] == t))[0]
            tP_ind = (np.where(t_data[i] == t - t_range.step * n_past))[0]
            tF_ind = (np.where(t_data[i] == t + t_range.step * (n_next - 1)))[0]

            if t0_ind.shape[0] == 0 or tP_ind.shape[0] == 0 or tF_ind.shape[0] == 0:
                continue

            t0_ind = t0_ind[0]
            tP_ind = tP_ind[0]
            tF_ind = tF_ind[0]

            dataset_t0.append(t)
            dataset_x.append(p_data[i][tP_ind:t0_ind])
            dataset_y.append(p_data[i][t0_ind:tF_ind + 1])


    sub_batches = []
    last_included_t = -1000
    min_interval = 1
    for i, t in enumerate(dataset_t0):
        if t > last_included_t + min_interval:
            sub_batches.append([i, i+1])
            last_included_t = t

        if t == last_included_t:
            sub_batches[-1][1] = i + 1

    sub_batches = np.array(sub_batches).astype(np.int32)
    dataset_x_ = []
    dataset_y_ = []
    last_ind = 0
    for ii, sb in enumerate(sub_batches):
        dataset_x_.append(dataset_x[sb[0]:sb[1]])
        dataset_y_.append(dataset_y[sb[0]:sb[1]])
        sb[1] = sb[1] - sb[0] + last_ind
        sb[0] = last_ind
        last_ind = sb[1]

    dataset_x = np.concatenate(dataset_x_)
    dataset_y = np.concatenate(dataset_y_)

    sub_batches = np.array(sub_batches).astype(np.int32)
    dataset_x = np.array(dataset_x).astype(np.float32)
    dataset_y = np.array(dataset_y).astype(np.float32)

    return dataset_x, dataset_y, dataset_t0, sub_batches




def calculate_neighbor_distance(dataset_dir):
    for dataset in os.listdir(dataset_dir):
        if '.npz' in dataset and 'dist' not in dataset:
            data = np.load(os.path.join(dataset_dir, dataset), allow_pickle=True)
            obsv, pred, batches, time = data['obsvs'], data['preds'], data['batches'], data['times']

            idx_and_dist = []
            for bs in batches:
                frame_obsv = obsv[bs[0]:bs[1]]
                if (bs[1]-bs[0]) == 1:
                    idx_and_dist.append(None)
                    continue
                n = len(frame_obsv)

                frame_dist = []
                for i in range(n):
                    dist_list = []
                    for j in range(n):
                        if i == j:
                            continue
                        dist = np.linalg.norm(frame_obsv[i,-1,:2] - frame_obsv[j,-1,:2])
                        dist_list.append((j, dist))
                        dist_list.sort(key=lambda x:x[1])

                    frame_dist.append(dist_list)
                    
                idx_and_dist.append(np.array(frame_dist).reshape(n, n-1, 2))

            np.savez(os.path.join(dataset_dir, dataset.replace('.npz', '_dist.npz')), 
                    obsvs = obsv, 
                    preds = pred, 
                    times = time, 
                    batches = batches,
                    idx_and_dist = idx_and_dist)




def scale_npz(dataset_dir):
    for dataset in os.listdir(dataset_dir):
        if '_dist.npz' in dataset and 'scale' not in dataset:
            data = np.load(os.path.join(dataset_dir, dataset), allow_pickle=True)
            obsv, pred, batches, time, idx_and_dist = data['obsvs'], data['preds'], data['batches'], data['times'], data['idx_and_dist']

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(obsv[:,:,:2].reshape(-1,).reshape(-1,2)) 
            obsv[:,:,:2] = scaler.transform(obsv[:,:,:2].reshape(-1,2)).reshape(-1,8,2)
            pred[:,:,:2] = scaler.transform(pred[:,:,:2].reshape(-1,2)).reshape(-1,12,2)

            scale_and_min = np.full((len(obsv), 2, 2), [scaler.scale_, scaler.data_min_])

            np.savez(os.path.join(dataset_dir, dataset.replace('.npz', '_scale.npz')), 
                     obsvs = obsv, 
                     preds = pred, 
                     times = time, 
                     batches = batches,
                     idx_and_dist = idx_and_dist,
                     scale_and_min = scale_and_min)




def merge_npz(dataset_dir, npz_out_file):
    dataset_obsv = list()
    dataset_pred = list()
    the_batches = list()
    data_time = list()
    source_and_bias = list()
    scale_and_min = list()
    idx_and_dist = list()
    list_empty = True
    index_bias = 0
    timestamp_bias = 0


    logtxt = open(os.path.join(dataset_dir, 'log.txt'), 'w')
    logtxt.writelines('dataset_name\t' + 'start_idx\t' + 'end_idx\t' + 'start_ts\t' + 'end_ts\t' + 'scale\t' + 'data_min\n')

    datasets = os.listdir(dataset_dir)
    datasets.sort()
    for dataset in datasets:
        if '_dist_scale.npz' in dataset:

            # If the merged npz already exist, then doesn't include it.
            if npz_out_file.split('/')[-1] == dataset:    
                continue

            data = np.load(os.path.join(dataset_dir, dataset), allow_pickle=True)
            obsv, pred, batches, time, id_dist, s_m = \
                    data['obsvs'], data['preds'], data['batches'], data['times'], data['idx_and_dist'], data['scale_and_min']

            batches = batches.astype(np.int32)
            time = time.astype(np.int32)

            if not list_empty:
                index_bias = the_batches[-1][-1][-1]      # last obsv index of previous dataset
                timestamp_bias = data_time[-1][-1] + 100  # last time stamp of previous dataset
                batches += index_bias
                time += timestamp_bias   

            dataset_obsv.append(obsv)
            dataset_pred.append(pred)
            the_batches.append(batches)
            data_time.append(time)
            idx_and_dist.append(id_dist)
            scale_and_min.append(s_m)
            list_empty = False
            
            #==================== Recovery log ====================
            source_and_bias.append(np.full((len(obsv), 3), [dataset, index_bias, timestamp_bias]))
            info = [str(dataset), '\t', str(batches[0][0]), '\t', str(batches[-1][-1]), '\t',
                    str(time[0]), '\t', str(time[-1]), '\t', str(s_m[0][0]), '\t', str(s_m[0][1]), '\n']
            logtxt.writelines(info)
            #======================================================
            
    dataset_obsv = np.concatenate(dataset_obsv)
    dataset_pred = np.concatenate(dataset_pred)
    the_batches = np.concatenate(the_batches)
    data_time = np.concatenate(data_time)
    idx_and_dist = np.concatenate(idx_and_dist)
    scale_and_min = np.concatenate(scale_and_min)
    source_and_bias = np.concatenate(source_and_bias)

    logtxt.close()


    np.savez(npz_out_file, 
             obsvs=dataset_obsv, 
             preds=dataset_pred, 
             times=data_time, 
             batches=the_batches,
             idx_and_dist = idx_and_dist,
             scale_and_min = scale_and_min,
             source_and_bias = source_and_bias)
