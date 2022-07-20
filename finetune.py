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
train_file_name = dataset_name+'train/hotel_dist_scale_train.npz' 
val_file_name = dataset_name+'val/hotel_dist_scale_val.npz'
train_file_path = data_path + train_file_name
val_file_path = data_path + val_file_name

train_data = np.load(train_file_path, allow_pickle=True)
train_obsv, train_pred, train_batches , train_time, train_scale, train_pairs = train_data['obsvs'], train_data['preds'], train_data['batches'], train_data['times'], train_data['scale_and_min'], train_data['idx_and_dist'] 
val_data = np.load(val_file_path, allow_pickle=True)
val_obsv, val_pred, val_batches , val_time, val_scale, val_pairs = val_data['obsvs'], val_data['preds'], val_data['batches'], val_data['times'], val_data['scale_and_min'] , val_data['idx_and_dist']

n_past = train_obsv.shape[1]
n_next = train_pred.shape[1]
t_obsv = train_obsv[:,:,:2].reshape(-1,8,2)
t_pred = train_pred[:,:,:2].reshape(-1,12,2)
v_obsv = val_obsv[:,:,:2].reshape(-1,8,2)
v_pred = val_pred[:,:,:2].reshape(-1,12,2)

def create_inout_sequences(obsv_t, pred_t):
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
    dec_in = copy.deepcopy(pred_t)
    dec_in = np.concatenate((mask, dec_in), axis=1)
    
    return torch.FloatTensor(enc_in), torch.FloatTensor(dec_in)

def get_data(obsv, pred):
    enc_in, dec_in = create_inout_sequences(obsv, pred)
    
    return enc_in, dec_in

def get_batch(enc_in, dec_in, idx, batch_pairs):
    enc_in_s = None
    dec_in_s = None
    
    perm = [] 
    if len(batch_pairs) > a_nn:
        for j, other_agents in enumerate(batch_pairs):
            top_k_agents = other_agents[:a_nn,0]
            for k in top_k_agents:
                perm.append((j,int(k)))
    else:
        perm = list(permutations(np.arange(len(batch_pairs)), 2))

    for i, (j,k) in enumerate(perm): 
        enc_r = enc_in[idx+j].unsqueeze(0)
        dec_r = dec_in[idx+j].unsqueeze(0)
        seg_r = np.zeros((enc_r.shape[0], enc_r.shape[1], 1)) 
        dec_r = np.concatenate((dec_r, np.zeros((dec_r.shape[0], dec_r.shape[1], 1))), axis=2)
        
        enc_seq = np.concatenate((enc_r, enc_in[idx+k, 1:].unsqueeze(0)), axis=1)
        seg_seq = np.concatenate((seg_r, np.ones((enc_in[idx+k,1:].unsqueeze(0).shape[0], enc_in[idx+k,1:].unsqueeze(0).shape[1], 1))), axis=1)
        
        enc_seq = np.concatenate((enc_seq, seg_seq), axis=2)
        if i == 0:
            enc_in_s = enc_seq 
            dec_in_s = dec_r
        else:
            enc_in_s = np.concatenate((enc_in_s, enc_seq), axis=0)
            dec_in_s = np.concatenate((dec_in_s, dec_r), axis=0)

    enc_in_s = np.transpose(enc_in_s, (1,0,2))
    dec_in_s = np.transpose(dec_in_s, (1,0,2))
    
    return torch.FloatTensor(enc_in_s).to(device), torch.FloatTensor(dec_in_s).to(device)

def train():
    model.train()  
    total_loss = 0.

    for i, sb in enumerate(tqdm(prep_train_batch), 1):
        enc_in, dec_in = sb
        in_data = enc_in.to(device)
        out_data = dec_in.to(device)

        optimizer.zero_grad()

        output = model(in_data, out_data[:-1]) 
        output[0] = output[0] + in_data[8,:,:2]
        output[1:] = output[1:] + out_data[1:-1,:,:2]
        loss = criterion(output , out_data[1:,:,:2]) 
        loss.backward()
        total_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if i == len(prep_train_batch):
            print('| epoch {:3d} | '
                  'lr {:02.8f} | '
                  'loss {:5.9f} '.format(
                    epoch, scheduler.get_lr()[0],
                    total_loss) ) 
         
def plot_and_loss(eval_model, total_agents, file):
    eval_model.eval() 
    total_loss, total_ade, total_fde = 0., 0., 0.
    
    with torch.no_grad():
        for e, sb in enumerate(tqdm(prep_test_batch), 1):

            enc_in, dec_in, batch_size = sb 
            in_data = enc_in.to(device)
            out_data = dec_in.to(device)

            dec_inp = out_data[0].unsqueeze(0)
            
            for i in range(12):
                out = eval_model(in_data, dec_inp) 
                if i == 0:
                    out = out + in_data[8,:,:2] 
                    out = torch.cat((out, torch.FloatTensor(np.zeros((out.shape[0], out.shape[1], 1))).to(device)), dim=2)
                else:
                    out[-1:,:,:2] = out[-1:,:,:2] + dec_inp[-1:,:,:2]
                    out = torch.cat((out, torch.FloatTensor(np.zeros((out.shape[0], out.shape[1], 1))).to(device)), dim=2)
                dec_inp = torch.cat((dec_inp, out[-1:]))
                
            loss = criterion(dec_inp[1:,:,:2], out_data[1:,:,:2])
            total_loss += loss.item()

            if batch_size < (a_nn+1):
                for j in range(batch_size):
                    tmp_ade = []
                    tmp_fde = []
                    for k in range(batch_size-1): 
                        output_arr = dec_inp[1:,j*(batch_size-1)+k,:2].cpu().detach().numpy()/val_scale[0,0] + val_scale[0,1] # (batch_size-1) -> (K)
                        target_arr = out_data[1:,j*(batch_size-1)+k,:2].cpu().detach().numpy()/val_scale[0,0] + val_scale[0,1]
                        diff_arr = np.linalg.norm((output_arr - target_arr), axis=1)
                        tmp_ade.append(diff_arr.sum()/12)
                        tmp_fde.append(diff_arr[11])
                    total_ade += min(tmp_ade)
                    total_fde += min(tmp_fde)   

            else:
                for j in range(batch_size):
                    tmp_ade = []
                    tmp_fde = []
                    for k in range(a_nn): 
                        output_arr = dec_inp[1:,j*(a_nn)+k,:2].cpu().detach().numpy()/val_scale[0,0] + val_scale[0,1] 
                        target_arr = out_data[1:,j*(a_nn)+k,:2].cpu().detach().numpy()/val_scale[0,0] + val_scale[0,1]
                        diff_arr = np.linalg.norm((output_arr - target_arr), axis=1)
                        tmp_ade.append(diff_arr.sum()/12)
                        tmp_fde.append(diff_arr[11])
                    total_ade += min(tmp_ade)
                    total_fde += min(tmp_fde)

            if e == len(prep_test_batch):
                print('| epoch {:3d} | '
                    'lr {:02.8f} | '
                    'loss {:5.9f} '.format(
                        epoch, scheduler.get_lr()[0],
                        total_loss) )
 
        print("ADE")
        print(total_ade/total_agents)
        print("FDE")
        print(total_fde/total_agents)
        file.write('| {:3d} | {:02.8f} | {:5.9f} | {:02.3f} | {:02.3f} |\n'.format(epoch, scheduler.get_lr()[0],
                        total_loss, total_ade/total_agents, total_fde/total_agents))
    return total_ade / total_agents

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
    def __init__(self,feature_size=256,num_layers=4,dropout=0.1):  # (256,4)
        super(TransMTM, self).__init__()
        self.model_type = 'Transformer'
        self.tgt_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=16, dropout=dropout) 

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)
        
        self.nn = nn.Linear(feature_size, feature_size) 
        self.Relu = nn.ReLU()
        self.xy_rec = nn.Linear(feature_size,2) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,enc,dec):          
        if self.tgt_mask is None or self.tgt_mask.size(0) != (len(dec)): 
            device = dec.device
            mask = self._generate_square_subsequent_mask(len(dec)).to(device)
            self.tgt_mask = copy.deepcopy(mask)   
            
        encoder_emb = self.pos_encoder(enc) 
        decoder_emb = self.pos_encoder(dec)
        
        with torch.no_grad():    
            encoder_output = self.transformer_encoder(encoder_emb) 
        
        decoder_output = self.transformer_decoder(tgt=decoder_emb, tgt_mask=self.tgt_mask, memory=encoder_output) 

        nn_out = torch.tanh(self.nn(decoder_output)) 

        output_xy = self.xy_rec(nn_out) 
        
        return output_xy
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
train_enc_in, train_dec_in = get_data(t_obsv, t_pred)
val_enc_in, val_dec_in = get_data(v_obsv, v_pred)
a_nn = 8 
prep_train_batch = []
prep_test_batch = []
total_agents = 0
for batch_idx, sb in enumerate(tqdm(train_batches)): 
    batch_size = sb[1] - sb[0]
    if batch_size == 1:
        continue
    prob = np.random.randint(100) 
    if prob > 1: 
        continue
    total_agents += batch_size
    batch_pair = train_pairs[batch_idx]
    enc_in, dec_in = get_batch(train_enc_in, train_dec_in, sb[0], batch_pair) 
    prep_train_batch.append((enc_in, dec_in))
    
    # For hotel dataset
    cls_tok, sep_tok = copy.deepcopy(enc_in[0,:,:]), copy.deepcopy(enc_in[9,:,:])
    sos_tok = copy.deepcopy(dec_in[0,:,:])
    x_enc_in, y_enc_in = copy.deepcopy(enc_in[:,:,0]), copy.deepcopy(enc_in[:,:,1])
    x_dec_in, y_dec_in = copy.deepcopy(dec_in[:,:,0]), copy.deepcopy(dec_in[:,:,1])
    enc_in[:,:,0], enc_in[:,:,1] = 1-y_enc_in, 1-x_enc_in 
    enc_in[0,:,:], enc_in[9,:,:], enc_in[-1,:,:] = copy.deepcopy(cls_tok), copy.deepcopy(sep_tok), copy.deepcopy(sep_tok)
    dec_in[:,:,0], dec_in[:,:,1] = 1-y_dec_in, 1-x_dec_in 
    dec_in[0,:,:] = copy.deepcopy(sos_tok)
    prep_train_batch.append((enc_in, dec_in))
    
    
max_ = 0
total_agents_test = 0 
for batch_idx, sb in enumerate(tqdm(val_batches)): 
    batch_size = sb[1] - sb[0]
    if batch_size == 1:
        continue
    if batch_size > max_:
        max_ = batch_size
    total_agents_test += batch_size
    batch_pair = val_pairs[batch_idx] 
    enc_in, dec_in = get_batch(val_enc_in, val_dec_in, sb[0]-val_batches[0][0], batch_pair) 
    prep_test_batch.append((enc_in, dec_in, batch_size))

model = TransMTM().to(device)

model_dict = TransMTM().state_dict() 
PATH = 'weight/hotel_pretext.pt'
pretrained_dict = torch.load(PATH, map_location='cuda:0')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
model.load_state_dict(model_dict)

criterion = nn.MSELoss()
lr = 3e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99) 

best_val_ade = float("inf")
best_model = None
epochs = 31
with open("hotel_finetune_3e-6.txt", "w") as f: 
    f.write('| epoch | lr | loss | ade | fde |\n')
    for epoch in range(1, epochs + 1):
        train() 
        if epoch % 2 == 1:
            val_ade = plot_and_loss(model, total_agents_test, f)
            if val_ade < best_val_ade:
                best_val_ade = copy.deepcopy(val_ade)
                best_model = copy.deepcopy(model)
                PATH = 'weight/hotel_finetune_3e-6.pt' 
                torch.save(best_model.state_dict(), PATH) 

    f.close()
    
