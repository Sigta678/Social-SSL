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
test_file_name = dataset_name+'test/hotel_dist_scale_test.npz'
test_file_path = data_path + test_file_name

test_data = np.load(test_file_path, allow_pickle=True)
test_obsv, test_pred, test_batches , test_time, test_scale, test_pairs = test_data['obsvs'], test_data['preds'], test_data['batches'], test_data['times'], test_data['scale_and_min'] , test_data['idx_and_dist']

test_obsv = test_obsv[:,:,:2].reshape(-1,8,2)
test_pred = test_pred[:,:,:2].reshape(-1,12,2)
future_timestamps = test_pred.shape[1]
print('Obsv data shape: test {}'.format(test_obsv.shape))
print('Pred data shape: test {}'.format(test_pred.shape))

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
    if len(batch_pairs) > a_scene:
        for j, other_agents in enumerate(batch_pairs):
            top_k_agents = other_agents[:a_scene,0]
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

         
def plot_and_loss(eval_model, total_agents, file):
    eval_model.eval() 
    total_loss, total_ade, total_fde = 0., 0., 0.
    with torch.no_grad():
        for e, sb in enumerate(tqdm(prep_test_batch), 1):

            enc_in, dec_in, batch_size = sb 
            in_data = enc_in.to(device)
            out_data = dec_in.to(device)
            dec_inp = out_data[0].unsqueeze(0)
            
            for i in range(future_timestamps):
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

            for j in range(batch_size):
                tmp_ade = []
                tmp_fde = []
                for k in range(batch_size-1): 
                    output_arr = dec_inp[1:,j*(batch_size-1)+k,:2].cpu().detach().numpy()/test_scale[0,0] + test_scale[0,1] # (batch_size-1) -> (K)
                    target_arr = out_data[1:,j*(batch_size-1)+k,:2].cpu().detach().numpy()/test_scale[0,0] + test_scale[0,1]
                    diff_arr = np.linalg.norm((output_arr - target_arr), axis=1)
                    tmp_ade.append(diff_arr.sum()/12)
                    tmp_fde.append(diff_arr[11])
                total_ade += min(tmp_ade) 
                total_fde += min(tmp_fde)    

        print("avgADE")
        print(total_ade/total_agents)
        print("avgFDE")
        print(total_fde/total_agents)
        file.write('| {:5.9f} | {:02.3f} | {:02.3f} |\n'.format(total_loss, total_ade/total_agents, total_fde/total_agents))

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
    
test_enc_in, test_dec_in = get_data(test_obsv, test_pred)

prep_test_batch = []
a_scene = 0 
total_agents_test = 0 
for batch_idx, sb in enumerate(test_batches): 
    batch_size = sb[1] - sb[0]
    if batch_size == 1:
        continue
    if batch_size > a_scene:
        a_scene = batch_size
    total_agents_test += batch_size
    batch_pair = test_pairs[batch_idx] 
    enc_in, dec_in = get_batch(test_enc_in, test_dec_in, sb[0]-test_batches[0][0], batch_pair) 
    prep_test_batch.append((enc_in, dec_in, batch_size))

model = TransMTM().to(device)
model_dict = TransMTM().state_dict()
PATH = 'weight/hotel_finetune.pt'
pretrained_dict = torch.load(PATH, map_location='cuda:0')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
model.load_state_dict(model_dict)

criterion = nn.MSELoss()

with open("hotel_eval.txt", "w") as f: 
    f.write('| loss | ade | fde |\n')
    plot_and_loss(model, total_agents_test, f) 
    f.close()
    
