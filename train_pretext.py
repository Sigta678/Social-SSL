import torch
import torch.nn as nn
import numpy as np
import math
import os
import random
import copy
from itertools import permutations
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = 'data/pedestrian/'
dataset_name = 'hotel/' 
train_file_name = dataset_name+'train/hotel_dist_scale_train.npz'  # leave-one-out pre-train 
val_file_name = dataset_name+'val/hotel_dist_scale_val.npz' 
file_path = data_path + train_file_name
val_file_path = data_path + val_file_name

pred_data = np.load(file_path, allow_pickle=True)
dataset_obsv, dataset_pred, the_batches, data_time, pairs = pred_data['obsvs'], \
    pred_data['preds'], pred_data['batches'], pred_data['times'], pred_data['idx_and_dist']
val_data = np.load(val_file_path, allow_pickle=True)   
val_dataset_obsv, val_dataset_pred, val_the_batches, val_data_time, val_pairs = val_data['obsvs'], \
    val_data['preds'], val_data['batches'], val_data['times'], val_data['idx_and_dist'] 

n_past = dataset_obsv.shape[1]
n_next = dataset_pred.shape[1]
obsv = dataset_obsv[:,:,:2]
val_obsv = val_dataset_obsv[:,:,:2]
pred = dataset_pred[:,:,:2]

def create_inout_sequences(obsv_t): 
    real_seq = copy.deepcopy(obsv_t)
    mask_idx = np.zeros((obsv_t.shape[0], obsv_t.shape[1], 1)) 
    # <CLS> 
    zeros = np.zeros((obsv_t.shape[0], 1, 2)) 
    zeros[:,:,0] = 1 
    zeros[:,:,1] = 0
    enc_in = np.concatenate((zeros, obsv_t), axis=1)
    # <SEP> 
    zeros[:,:,0] = 1
    zeros[:,:,1] = 1
    enc_in = np.concatenate((enc_in, zeros), axis=1)

    mask = np.zeros((obsv_t.shape[0], 1, 2))
    dec_in = copy.deepcopy(obsv_t[:,:-1])
    dec_in = np.concatenate((mask, dec_in), axis=1)
    
    L = obsv_t.shape[0] 
    for i in range(L):     
        start_idx = random.randint(1,3) 
        mask_idx[i,start_idx:start_idx+4] = 1 
        enc_in[i,start_idx:start_idx+4] = [0,0]
        dec_in[i,:start_idx+1] = [0,0] 
        dec_in[i,start_idx+4:] = [0,0]

    return torch.FloatTensor(enc_in), torch.FloatTensor(dec_in), torch.FloatTensor(mask_idx), torch.FloatTensor(real_seq)

def get_data(obsv_d):
    enc_in, dec_in, mask_idx, real_seq = create_inout_sequences(obsv_d)
    return enc_in, dec_in, mask_idx, real_seq

def get_batch(enc_in, dec_in, mask_idx, real_seq, idx, batch_pairs):
    
    enc_in_s = None
    dec_in_s = None
    mask_in_s = None
    real_in_s = None
    
    perm = []
    if len(batch_pairs) > a_nn:
        for j, other_agents in enumerate(batch_pairs):
            top_k_agents = other_agents[:a_nn,0]
            for k in top_k_agents:
                perm.append((j,int(k)))
    else:
        perm = list(permutations(np.arange(len(batch_pairs)), 2))

    sparse_distance_score = np.zeros(len(perm))
    distance_score = np.zeros(len(perm)) 
    distance_score_2 = np.zeros(len(perm)) 
    distance_score_3 = np.zeros(len(perm)) 
    distance_score_4 = np.zeros(len(perm)) 
    distance_score_5 = np.zeros(len(perm)) 
    distance_score_6 = np.zeros(len(perm)) 
    distance_score_7 = np.zeros(len(perm)) 
    distance_score_8 = np.zeros(len(perm)) 
    dynamic_label = np.zeros(len(perm))
    
    skip_count = 0
    for i, (j,k) in enumerate(perm): 
        sentiment_count = 0
        static_count = 0
        sparse_interact_tag = 0
        enc_r = enc_in[idx+j].unsqueeze(0)
        dec_r = dec_in[idx+j].unsqueeze(0)
        mask_r = mask_idx[idx+j].unsqueeze(0)
        real_r = real_seq[idx+j].unsqueeze(0)

        seg_r = np.zeros((enc_r.shape[0], enc_r.shape[1], 1)) 
        dec_r = np.concatenate((dec_r, np.zeros((dec_r.shape[0], dec_r.shape[1], 1))), axis=2)
        agent_j = real_seq[idx+j] 
        agent_k = real_seq[idx+k]
        
        diff_arr = agent_j - agent_k 
        diff = np.linalg.norm(diff_arr, axis=1)  
        for t in diff[4:]:
            if t < 0.15:
                sparse_interact_tag = 1
        for e,t in enumerate(diff):
            interact_tag = 0
            # Dist thres
            if t < 0.15:
                interact_tag = 1
            elif e == 0:
                distance_score[int(i)] = interact_tag 
            elif e == 1:
                distance_score_2[int(i)] = interact_tag
            elif e == 2:
                distance_score_3[int(i)] = interact_tag
            elif e == 3:
                distance_score_4[int(i)] = interact_tag
            elif e == 4:
                distance_score_5[int(i)] = interact_tag 
            elif e == 5:
                distance_score_6[int(i)] = interact_tag
            elif e == 6:
                distance_score_7[int(i)] = interact_tag
            elif e == 7:
                distance_score_8[int(i)] = interact_tag

        social_agent_d = agent_k[4:] - agent_k[3:7]
        social_agent_dist_change = np.linalg.norm(social_agent_d, axis=1)
        if True:
            sentiment = diff[4:] - diff[3:7] 
            for s in sentiment:
                if s > 0: 
                    sentiment_count += 1
                elif s < 0: 
                    sentiment_count -= 1

            if sparse_interact_tag == 1:
                sparse_distance_score[int(i)] = 1 
                
            if sentiment_count > 0:
                dynamic_label[int(i)] = 0
            elif sentiment_count == 0:
                dynamic_label[int(i)] = 1
            elif sentiment_count < 0:
                dynamic_label[int(i)] = 2     

        enc_seq = np.concatenate((enc_r, enc_in[idx+k,1:].unsqueeze(0)), axis=1)
        seg_seq = np.concatenate((seg_r, np.ones((enc_in[idx+k,1:].unsqueeze(0).shape[0], enc_in[idx+k,1:].unsqueeze(0).shape[1], 1))), axis=1)
        enc_seq = np.concatenate((enc_seq, seg_seq), axis=2)

        if i == 0:
            enc_in_s = enc_seq 
            dec_in_s = dec_r
            mask_in_s = mask_r
            real_in_s = real_r
        else:
            enc_in_s = np.concatenate((enc_in_s, enc_seq), axis=0)
            dec_in_s = np.concatenate((dec_in_s, dec_r), axis=0)
            mask_in_s = np.concatenate((mask_in_s, mask_r), axis=0)
            real_in_s = np.concatenate((real_in_s, real_r), axis=0)

    enc_in_s = np.transpose(enc_in_s, (1,0,2))
    dec_in_s = np.transpose(dec_in_s, (1,0,2))
    mask_idx_s = np.transpose(mask_in_s, (1,0,2))
    real_seq_s = np.transpose(real_in_s, (1,0,2))
    
    return torch.FloatTensor(enc_in_s), torch.FloatTensor(dec_in_s), torch.FloatTensor(mask_idx_s), torch.FloatTensor(real_seq_s), \
           torch.LongTensor(sparse_distance_score), torch.LongTensor(distance_score), torch.LongTensor(distance_score_2), torch.LongTensor(distance_score_3), torch.LongTensor(distance_score_4), \
           torch.LongTensor(distance_score_5), torch.LongTensor(distance_score_6), torch.LongTensor(distance_score_7), torch.LongTensor(distance_score_8), torch.LongTensor(dynamic_label)


def train():
    model.train() 
    total_loss, total_mse, total_ce_i, total_ce_d = 0., 0., 0., 0.
    
    batches_index = np.arange(len(prep_train_batch))
    np.random.shuffle(batches_index)
    for i, index in enumerate(tqdm(batches_index), 1):
        sb = prep_train_batch[index]
        enc_in, dec_in, mask_idx, real, sparse_interact, interact_1, interact_2, interact_3, interact_4, interact_5, interact_6, interact_7, interact_8, dynamic_ = sb
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        mask_idx = mask_idx.to(device)
        real = real.to(device)
        sparse_interact = sparse_interact.to(device)
        interact_1 = interact_1.to(device)
        interact_2 = interact_2.to(device)
        interact_3 = interact_3.to(device)
        interact_4 = interact_4.to(device)
        interact_5 = interact_5.to(device)
        interact_6 = interact_6.to(device)
        interact_7 = interact_7.to(device)
        interact_8 = interact_8.to(device)
        dynamic_ = dynamic_.to(device)
        
        optimizer.zero_grad()

        out_xy, out_sparse_interact, out_interact_1, out_interact_2, out_interact_3, out_interact_4, out_interact_5, out_interact_6, out_interact_7, out_interact_8, out_dynamic = model(enc_in, dec_in)  # , out_interact, out_dynamic
        out_xy[1:] = out_xy[1:] + real[:-1] 
        output_ = out_xy*mask_idx 
        true_ = real*mask_idx 
        MSE = mse(output_, true_)
        ce_i_s = ceLoss_dist_thres(out_sparse_interact, sparse_interact)
        ce_i_1 = ceLoss_dist_thres(out_interact_1, interact_1)
        ce_i_2 = ceLoss_dist_thres(out_interact_2, interact_2)
        ce_i_3 = ceLoss_dist_thres(out_interact_3, interact_3)
        ce_i_4 = ceLoss_dist_thres(out_interact_4, interact_4)
        ce_i_5 = ceLoss_dist_thres(out_interact_5, interact_5)
        ce_i_6 = ceLoss_dist_thres(out_interact_6, interact_6)
        ce_i_7 = ceLoss_dist_thres(out_interact_7, interact_7)
        ce_i_8 = ceLoss_dist_thres(out_interact_8, interact_8)
        ce_i = ce_i_s + ce_i_1 + ce_i_2 + ce_i_3 + ce_i_4 + ce_i_5 + ce_i_6 + ce_i_7 + ce_i_8
        ce_d = ceLoss_dynamic(out_dynamic, dynamic_)

        lambda_i = 1 
        lambda_d = 1
        loss = MSE + lambda_d*ce_d + lambda_i*ce_i 
        loss.backward()
        total_loss += loss.item()     
        total_mse += MSE.item()
        total_ce_i += ce_i.item()
        total_ce_d += ce_d.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if i == len(prep_train_batch):
            print('| epoch {:3d} | '
                  'lr {:02.7f} | '
                  'mse {:5.11f} | distance {:5.11f} | dynamic {:5.11f}'.format(
                    epoch, scheduler.get_lr()[0],
                    total_mse, total_ce_i/len(prep_train_batch), total_ce_d/len(prep_train_batch)) ) # *8/total_mask

def plot_and_loss(eval_model, test_agent, file):
    eval_model.eval() 
    total_loss, total_mse, total_ce_i, total_ce_d = 0., 0., 0., 0.
    start_idx = None
    
    with torch.no_grad():
        for i, sb in enumerate(prep_test_batch, 1):
            
            enc_in, dec_in, mask_idx, real, sparse_interact, interact_1, interact_2, interact_3, interact_4, interact_5, interact_6, interact_7, interact_8, dynamic_ = sb 
            enc_in = enc_in.to(device)
            dec_in = dec_in.to(device)
            mask_idx = mask_idx.to(device)
            real = real.to(device)
            sparse_interact = sparse_interact.to(device)
            interact_1 = interact_1.to(device)
            interact_2 = interact_2.to(device)
            interact_3 = interact_3.to(device)
            interact_4 = interact_4.to(device)
            interact_5 = interact_5.to(device)
            interact_6 = interact_6.to(device)
            interact_7 = interact_7.to(device)
            interact_8 = interact_8.to(device)
            dynamic_ = dynamic_.to(device)
            
            out_xy, out_sparse_interact, out_interact_1, out_interact_2, out_interact_3, out_interact_4, out_interact_5, out_interact_6, out_interact_7, out_interact_8, out_dynamic = eval_model(enc_in, dec_in) # 
            out_xy[1:] = out_xy[1:] + real[:-1] 
            output_ = out_xy*mask_idx 
            true_ = real*mask_idx 

            MSE = mse(output_, true_)
            ce_i_s = ceLoss_dist_thres(out_sparse_interact, sparse_interact)
            ce_i_1 = ceLoss_dist_thres(out_interact_1, interact_1)
            ce_i_2 = ceLoss_dist_thres(out_interact_2, interact_2)
            ce_i_3 = ceLoss_dist_thres(out_interact_3, interact_3)
            ce_i_4 = ceLoss_dist_thres(out_interact_4, interact_4)
            ce_i_5 = ceLoss_dist_thres(out_interact_5, interact_5)
            ce_i_6 = ceLoss_dist_thres(out_interact_6, interact_6)
            ce_i_7 = ceLoss_dist_thres(out_interact_7, interact_7)
            ce_i_8 = ceLoss_dist_thres(out_interact_8, interact_8)
            ce_i = ce_i_s + ce_i_1 + ce_i_2 + ce_i_3 + ce_i_4 + ce_i_5 + ce_i_6 + ce_i_7 + ce_i_8
            ce_d = ceLoss_dynamic(out_dynamic, dynamic_)
    
            loss = MSE + ce_i + ce_d 
            total_loss += loss.item() 
            
            total_mse += MSE.item()
            total_ce_i += ce_i.item()
            total_ce_d += ce_d.item()
            
            if i == len(prep_test_batch):
                print('| epoch {:3d} | '
                    'lr {:02.7f} | '
                    'mse {:5.11f} | distance {:5.11f} | dynamic {:5.11f}'.format(
                        epoch, scheduler.get_lr()[0], total_mse, total_ce_i/len(prep_test_batch), total_ce_d/len(prep_test_batch)) ) 

                file.write('| {:3d} | '
                  ' {:02.7f} | '
                  ' {:5.11f} | {:5.11f} | {:5.11f}\n'.format(
                    epoch, scheduler.get_lr()[0],
                    total_mse, total_ce_i/len(prep_test_batch), total_ce_d/len(prep_test_batch)) )

    return total_mse / test_agent

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model): 
        super(PositionalEncoding, self).__init__()   
        self.xy_emb = nn.Linear(2, d_model)
        self.segment_emb = nn.Linear(1, d_model) 
        
        max_len = 256 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) 
        pe.requires_grad = False 
        self.register_buffer('pe', pe)
        

    def forward(self, x):
        tmp_pe = copy.deepcopy(self.pe)
        for i in range(x.shape[1]-1): 
            tmp_pe = torch.cat([self.pe, tmp_pe], dim=1)
        x_xy = self.xy_emb(x[:,:,:2])
        x_seg = self.segment_emb(x[:,:,-1].unsqueeze(2))

        return x_xy + tmp_pe[:x.shape[0]] + x_seg 
    
class TransMTM(nn.Module):
    def __init__(self,feature_size=256,num_layers=4,dropout=0.1): 
        super(TransMTM, self).__init__()
        self.model_type = 'Transformer'
        self.tgt_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=16, dropout=dropout) 
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)
        
        self.nn = nn.Linear(feature_size, feature_size) 
        self.xy_rec = nn.Linear(feature_size,2) 
        
        self.Relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.nn_cls = nn.Linear(feature_size, feature_size) 
        self.nn_sparse_interact = nn.Linear(feature_size, 2)
 
        self.nn_interact = nn.Linear(feature_size, 2) 
        self.nn_dynamic = nn.Linear(feature_size, 3)

    def forward(self,enc, dec):    
        if self.tgt_mask is None or self.tgt_mask.size(0) != (len(dec)): 
            device = dec.device
            mask = self._generate_square_subsequent_mask(len(dec)).to(device)
            self.tgt_mask = copy.deepcopy(mask)
           
        encoder_emb = self.pos_encoder(enc) 
        decoder_emb = self.pos_encoder(dec)
        
        encoder_output = self.transformer_encoder(encoder_emb)

        output_interact_1 = self.softmax(self.nn_interact(encoder_output[1]-encoder_output[10]))
        output_interact_2 = self.softmax(self.nn_interact(encoder_output[2]-encoder_output[11]))
        output_interact_3 = self.softmax(self.nn_interact(encoder_output[3]-encoder_output[12]))
        output_interact_4 = self.softmax(self.nn_interact(encoder_output[4]-encoder_output[13]))
        output_interact_5 = self.softmax(self.nn_interact(encoder_output[5]-encoder_output[14]))
        output_interact_6 = self.softmax(self.nn_interact(encoder_output[6]-encoder_output[15]))
        output_interact_7 = self.softmax(self.nn_interact(encoder_output[7]-encoder_output[16]))
        output_interact_8 = self.softmax(self.nn_interact(encoder_output[8]-encoder_output[17]))
        decoder_output = self.transformer_decoder(tgt=decoder_emb, tgt_mask=self.tgt_mask, memory=encoder_output) 
        
        nn_out = torch.tanh(self.nn(decoder_output))
        output_xy = self.xy_rec(nn_out)
        
        nn_cls = self.Relu(self.nn_cls(encoder_output[0]))
        output_sparse_interact = self.softmax(self.nn_sparse_interact(nn_cls)) 
        output_dynamic = self.softmax(self.nn_dynamic(nn_cls))

        return output_xy , output_sparse_interact, output_interact_1, output_interact_2, output_interact_3, output_interact_4, output_interact_5, output_interact_6, output_interact_7, output_interact_8, output_dynamic
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


d_enc_in, d_dec_in, d_mask, d_real = get_data(obsv)
val_d_enc_in, val_d_dec_in, val_d_mask, val_d_real = get_data(val_obsv)
a_nn = 8 
prep_train_batch = []
prep_test_batch = []
test_agent = 0
for idx, sb in enumerate(tqdm(the_batches)):
    batch_size = sb[1] - sb[0]
    if batch_size == 1:
        continue
    batch_pair = pairs[idx]
    enc_in, dec_in, mask_idx, real, sparse_interact, interact_1, interact_2, interact_3, interact_4, interact_5, interact_6, interact_7, interact_8, dynamic_ = get_batch(d_enc_in, d_dec_in, d_mask, d_real, sb[0], batch_pair)
    prep_train_batch.append((enc_in, dec_in, mask_idx, real, sparse_interact, interact_1, interact_2, interact_3, interact_4, interact_5, interact_6, interact_7, interact_8, dynamic_))
    
for idx, sb in enumerate(tqdm(val_the_batches)):
    batch_size = sb[1] - sb[0]
    if batch_size == 1:
        continue
    batch_pair = val_pairs[idx]

    enc_in, dec_in, mask_idx, real, sparse_interact, interact_1, interact_2, interact_3, interact_4, interact_5, interact_6, interact_7, interact_8, dynamic_ = get_batch(val_d_enc_in, val_d_dec_in, val_d_mask, val_d_real, sb[0], batch_pair)
    prep_test_batch.append((enc_in, dec_in, mask_idx, real, sparse_interact, interact_1, interact_2, interact_3, interact_4, interact_5, interact_6, interact_7, interact_8, dynamic_))
    test_agent += batch_size

model = TransMTM().to(device)
'''
model_dict = TransMTM().state_dict() 
PATH = 'best/pretext_hotel.pt' 
pretrained_dict = torch.load(PATH)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
model.load_state_dict(model_dict)
'''

mse = nn.MSELoss()
ceLoss_dist_thres = nn.CrossEntropyLoss()
ceLoss_dynamic = nn.CrossEntropyLoss()

lr = 3e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 300.0, gamma=0.3) 

best_val_loss = float("inf")
best_model = None
epochs = 700  
with open("pretext_hotel_3e-6.txt", "w") as f: 
    f.write('| epoch | lr | mse | interact | dynamic |\n')
    for epoch in range(1, epochs + 1):
        train() 
        if epoch % 10 == 1: 
            val_loss = plot_and_loss(model, test_agent, f)
            if val_loss < best_val_loss:
                best_val_loss = copy.deepcopy(val_loss)
                best_model = copy.deepcopy(model)
                PATH = 'weight/pretext_hotel_3e-6.pt' 
                torch.save(best_model.state_dict(), PATH) 
            
