import os,argparse
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional 
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
from mamba_ssm import Mamba
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import numpy as np
from torch_geometric.nn import GINEConv
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from torch_geometric.nn import GATConv  
from torch_geometric.data import Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score,adjusted_rand_score, normalized_mutual_info_score,silhouette_score,precision_score,recall_score,pairwise_distances_argmin_min,pairwise_distances,silhouette_samples,rand_score
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')



def set_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    # np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed_num)


set_seed(114514)
def create_path(root_dir, dataset_name):
    output_directory = root_dir +  dataset_name + '/' 
    if os.path.exists(output_directory):
        print("exist")
    else:
        os.makedirs(output_directory)
    return output_directory 

def load_data(fname):
    path = '/PG-Mamba/dataset/'
    data_train = pd.read_csv(path + fname+'/'+fname+'_TRAIN.tsv', sep='\t',header=None)
    data_test = pd.read_csv(path + fname+'/'+fname+'_TEST.tsv', sep='\t',header=None)
    
    data_test = np.nan_to_num(data_test)
    data_train = np.nan_to_num(data_train)
    
    train_X = data_train[:,1:]
    train_X = train_X.astype('float32')
    label_train = data_train[:,0].astype('float32')
    
    test_X = data_test[:,1:]
    test_X = test_X.astype('float32')
    label_test = data_test[:,0].astype('float32')
    judge = check_dataset(fname)
    if judge:
        print("{} in adjust_datasets".format(fname))
    
    x = np.concatenate((train_X, test_X))
    y = np.concatenate((label_train, label_test))
  
	  
    return x,y,judge
    
def check_dataset(dataset_name):
    adjust_datasets=['ACSF1','BME','BirdChicken','Chinatown','Coffee','DistalPhalanxOutlineCorrect','DodgerLoopGame','ECG5000','FordA','FordB','Fungi','GunPoint''GunPointAgeSpan','Ham','HouseTwenty','InsectEPGRegularTrain','InsectEPGSmallTrain','Meat','MiddlePhalanxOutlineCorrect','MixedShapesSmallTrain','PLAID','Phoneme','ProximalPhalanxOutlineAgeGroup','RefrigerationDevices','ScreenType','SemgHandGenderCh2','SemgHandMovementCh2','SemgHandSubjectCh2','ShapeletSim','SmoothSubspace','UMD','Wafer','WormsTwoClass']
    return 1 if dataset_name in adjust_datasets else 0  

def create_data_object_list(time_series_data, k=8, time_window=3,sigma=1.0):
    data_list = []
    #print('patch',time_series_data.shape)
    for sample_idx in range(len(time_series_data)):
        sample = time_series_data[sample_idx]  # (T, F)
        T, F = sample.shape
        
       
        df = pd.DataFrame(sample, columns=[f'f_{i}' for i in range(F)])
        df['time_step'] = range(T)
        
        # time_edges
        time_edges = []
        for t in range(T):
            for dt in range(-time_window, time_window+1):
                if t + dt >=0 and t + dt < T and dt !=0:
                    time_edges.append((t, t+dt))
        
        #feature_edges
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(sample)
        _, knn_indices = knn.kneighbors(sample)
        feature_edges = []
        for i in range(T):
            for j in knn_indices[i]:
                feature_edges.append((i, j))
        
        
        all_edges = list(set(time_edges + feature_edges))
        edge_index = torch.tensor(list(zip(*all_edges)), dtype=torch.long)
        #print(edge_index)
        edge_attr = []
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            dist = np.linalg.norm(sample[src] - sample[dst])
            dist = torch.exp(-torch.tensor(dist) / sigma)
            #print("dist",dist)
            edge_attr.append(dist)  
        edge_attr = torch.tensor(edge_attr).unsqueeze(1).float()
        #print(edge_attr)
        node_features = torch.tensor(sample, dtype=torch.float32)
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=T,
            seq_len=T 
        )
        data_list.append(data)
    
    return data_list
class MSGLayer(nn.Module):

    def __init__(self, q_len,patch_len,stride,padding_patch,buildA_true, gcn_depth, num_nodes, dropout, subgraph_size, node_dim, propalpha, tanhalpha,conv_channels, residual_channels):
        super(MSGLayer,self).__init__()
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.q_len = q_len
        self.padding_patch = padding_patch
        
        
        self.MSPatch = MSPatch(self.q_len,patch_len, stride, self.padding_patch)
        #self.gc = graph_constructor(num_nodes,subgraph_size, node_dim,alpha=tanhalpha)
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gconv1.append(mixprop(patch_len, patch_len, gcn_depth, dropout, propalpha))
        self.gconv2.append(mixprop(patch_len, patch_len, gcn_depth, dropout, propalpha))
        self.perm = np.random.permutation(range(num_nodes))
    def forward(self, x):
        
        
        z = self.MSPatch(x).to(device)
        
        if self.buildA_true:
             
            adp_tc = torch.tensor(adj.to_numpy()).float().to(device)
            
            x_tc = (self.gconv1[0](z, adp_tc) + self.gconv2[0](z, adp_tc.transpose(1, 0))).to(device)
            
            return x_tc
        return z  
        
class MSPatch(nn.Module):
    def __init__(self, q_len, patch_len, stride, padding_patch):
        super(MSPatch,self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
    
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) #
            q_len += 1 # q_len is patch number


    def forward(self, x:Tensor, scores:Optional[Tensor]=None,key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        attn_bias=None
        
        # do patching
        if self.padding_patch == 'end':
           
            x = self.padding_patch_layer(x).to(device)
        
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [bs x patch_num x patch_len]

        return x
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nwl,wv->nwl',(x,A))
        return x.contiguous()
        
class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)  
        
class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
   

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to('cuda:3')
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        
        ho = torch.cat(out,dim=3).to('cuda:3')
      
        ho=ho.permute(0,3,2,1)
        
        ho = self.mlp(ho)
        return ho
        
class RevIN(nn.Module):
   
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Mmamba(nn.Module):
    def __init__(self,context_window,n1,n2,dropout1,ch_ind,d_state,d_conv,e_fact,pred_len,residual,c_in):
        super().__init__()
        
        act_layer = nn.SiLU
        self.seq_len= context_window
        self.n1= n1
        self.n2= n2
        self.dropout1= dropout1
        self.ch_ind= ch_ind
        self.d_state= d_state
        self.d_conv= d_conv
        self.e_fact= e_fact
        self.pred_len= pred_len
        self.residual= residual
        self.enc_in= c_in
        self.context_window=context_window
        self.lin_before = nn.Linear(self.context_window,self.context_window*2)
        self.lin1=nn.Linear(self.context_window,self.n1)
        self.dropout1=nn.Dropout(dropout1)
        
        self.weight_param = 0.02
        self.lin2=nn.Linear(self.n1,self.n2)
        self.dropout2=nn.Dropout(dropout1)
       
        
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.pred_len, out_channels=self.n1,bias = True, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(self.n1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
         )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.n2, out_channels=self.n1,bias = True, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(self.n1),
            nn.Dropout2d(0.2))
        
        self.Norm1d= nn.BatchNorm1d(self.n1)
        #self.act: nn.Module = act_layer()
        #self.act = nn.LeakyReLU(0.2)
        if self.ch_ind==1:
            self.d_model_param1=1
            self.d_model_param2=1

        else:
            self.d_model_param1=self.n2
            self.d_model_param2=self.n1
        
        
        self.pool3d = nn.MaxPool3d((2,2,self.pred_len//2),stride=(4,4,self.pred_len//2),padding=0)
        self.flatten = nn.Flatten()
       
        self.mamba1=Mamba(d_model=self.d_model_param1//4,d_state=self.d_state,d_conv=self.d_conv,expand=self.e_fact) 
      
        self.mamba3=Mamba(d_model=self.n1//4,d_state=self.d_state,d_conv=self.d_conv,expand=self.e_fact)
       
        torch.compile(self.mamba1, mode="reduce-overhead")
        #torch.compile(self.mamba2, mode="reduce-overhead")
        torch.compile(self.mamba3, mode="reduce-overhead")
        #torch.compile(self.mamba4, mode="reduce-overhead")
        self.lin3=torch.nn.Linear(self.n2,self.n1)
        self.lin3_=torch.nn.Linear(self.n1,self.pred_len)
        self.lin4=torch.nn.Linear(2*self.n1,self.pred_len)
        self.revin_layer_ts = RevIN(c_in, affine=True, subtract_last=False)
        self.revin_layer = RevIN(c_in*2, affine=True, subtract_last=False)
        self.SE_layer = SEModule(c_in*2)
        self.SE_layer_ts = SEModule(c_in)
       
        
        self.mamba1_conv2d = nn.Conv2d(
            in_channels=self.pred_len, out_channels=self.n1,
            bias = True, kernel_size=(1,1))
        self.mamba2_conv2d = nn.Conv2d(
            in_channels=self.n1, out_channels=self.n2,
            bias = True, kernel_size=(1,1))
        self.mamba3_conv2d = nn.Conv2d(
            in_channels=self.n2, out_channels=self.n1,
            bias = True, kernel_size=(1,1))
        self.mamba4_conv2d = nn.Conv2d(
            in_channels=2*self.n1, out_channels=self.pred_len,
            bias = True, kernel_size=(1,1))    
        
        
        
    def forward(self, x, higuchi_fd=None):
        
         weight_decay = 0.005
         kmax = 4
         fd_values = []
         fd_values_2 = []
         groups = 2
         
         x= x.permute(0,2,1)
         x=self.revin_layer_ts(x,'norm')
         x= x.permute(0,2,1)
         
         x_inite = x
         x_conv = self.conv2d_1(x_inite.permute(2,1,0).unsqueeze(0))
         x_conv = x_conv.squeeze(0).permute(2,1,0)
         
         if self.ch_ind==1:
             x=torch.reshape(x,(x.shape[0]*x.shape[1],1,x.shape[2]))
         
         x = self.mamba1_conv2d(x.permute(2,1,0))
         x = x.permute(2,1,0)
         
         x_res1=x + x_conv
         
         x_1, x_2, x_3, x_4 = torch.chunk(x, 4, dim=2)
         x_mamba1 = self.mamba3(x_1) 
         x_mamba2 = self.mamba3(x_2) 
         x_mamba3 = self.mamba3(x_3) 
         x_mamba4 = self.mamba3(x_4) 
         x3 = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
         
         x = self.mamba2_conv2d(x.permute(2,1,0))
         x = x.permute(2,1,0)
         
         x_res2=x
         x=self.dropout2(x)
         
         if self.ch_ind==1:
             x1=torch.permute(x,(0,2,1))
         else:
             x1=x
        
         
         x_1, x_2, x_3, x_4 = torch.chunk(x1, 4, dim=2)
         x_mamba1 = self.mamba1(x_1) 
         x_mamba2 = self.mamba1(x_2) 
         x_mamba3 = self.mamba1(x_3) 
         x_mamba4 = self.mamba1(x_4) 
         x1 = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
        
         if self.residual==1:
             x=x1+x_res2
         else:
             x=x1
        
         x = self.mamba3_conv2d(x.permute(2,1,0))
         x = x.permute(2,1,0)
         x = self.SE_layer_ts(x.unsqueeze(3)).squeeze(3) 
         
         if self.residual==1:
             x=x+x_res1

         
         x=torch.cat([x,x3],dim=2)
         
       
         x = self.mamba4_conv2d(x.permute(2,1,0))
         x = x.permute(2,1,0)
         
         if self.ch_ind==1:
             x=torch.reshape(x,(-1,self.enc_in,self.pred_len))
         
         
         x = x.permute(0,2,1)
         x = self.revin_layer_ts(x, 'denorm')
         x = x.permute(0,2,1)
        
       
         return x
         
#Feature Representation Learning Module
class TemporalMamba(nn.Module):
    def __init__(self, input_dim, d_model=128,patch_num=10, n_layers=2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, d_model)
        self.mamba_layers = nn.ModuleList([
            Mmamba(d_model,128,64,0.2,0,256,2,1,d_model,1,patch_num) for _ in range(n_layers)
        ])
        self.d_model = d_model
        self.patch_num = patch_num
    def forward(self, x):
        # x: (B, m, d) -> (B, m, D)
        x = self.proj_in(x)
        x = x.reshape(-1,self.patch_num,self.d_model)
        
        for layer in self.mamba_layers:
            x = F.relu(layer(x))
        x = x.reshape(-1,self.d_model)
       
        return x
        
class MambaGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,batch_size = 16,patch_num=10,gnn_dropout=0.3):
        super().__init__()
        #print(input_dim)
        self.temporal_encoder = TemporalMamba(input_dim,hidden_dim,patch_num)
        self.gnn = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        ))
        self.gat = GATConv(hidden_dim, hidden_dim, heads = 1, dropout = gnn_dropout)
         
        self.hidden_dim =hidden_dim
        self.batch_size =batch_size
        self.edge_encoder = nn.Linear(1, hidden_dim)
        self.patch_num = patch_num
        self.decoder = nn.Sequential(
           nn.Linear(2* hidden_dim,  hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, 1))
    def forward(self, data):
        
        x_seq = data.x.unsqueeze(0)  # (B, m, d)
        #print("before",x_seq.shape)
        temporal_feat = self.temporal_encoder(x_seq)  # (B, m, D)
        #print("after",temporal_feat.shape)
         
        x_graph = temporal_feat.squeeze(0)  
        x = F.silu(x_graph)
        
        edge_index = data.edge_index
        
        edge_attr = self.edge_encoder(data.edge_attr)
        #the reconstruction of the adjacency matrix
        edge_emb = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]]], dim=1)
        adj_recon = self.decoder(edge_emb)
        
        graph_embed = global_mean_pool(x, batch=data.batch)
        
        return graph_embed,adj_recon

        
class DeepClustering(nn.Module):
    def __init__(self, encoder, n_clusters, alpha=1.0):
        super().__init__()
        self.encoder = encoder
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, encoder.hidden_dim))
        nn.init.xavier_normal_(self.cluster_layer)
        self.alpha = alpha
        self.n_clusters = n_clusters
        
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        p = p / p.sum(1, keepdim=True)
        return p
    
    def forward(self, data):
        #print(data.edge_index.shape,)
        z,recon = self.encoder(data)  
        #print(z.shape)
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer)**2, dim=2) / self.alpha)
        q = q**((self.alpha +1.0)/2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return z, q,recon

#data augmentation
def temporal_augmentation(x, mask_ratio=0.2, jitter_scale=0.1):
   
    T, F = x.shape
    augmented = x.clone()
    
   
    mask = torch.rand(T) < mask_ratio
    augmented[mask] = 0
    
   
    noise = torch.randn_like(augmented) * jitter_scale
    augmented += noise
    
    return augmented
def feature_dropout(x, drop_prob=0.3):
    
    mask = torch.rand_like(x) > drop_prob
    out = x * mask.float()
    
    return out

    
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        data_0 = Data(
            x=data.x,
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() 
        )
        data_aug1 = Data(
            x=temporal_augmentation(data.x),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() 
        )
        
       
        return data_0,data_aug1

#Lc
def contrastive_loss(z1, z2, temperature):
    
    batch_size = z1.size(0)
    
   
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
   
    logits = torch.mm(z1, z2.t()) / temperature
    
   
    labels = torch.arange(batch_size).to(device)
    
   
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_j) / 2

#Feature Representation Learning Pre-training
def pretrain_contrastive(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    
    for data in dataloader:
        aug1_batch = data[1].to(device)
        aug2_batch = data[0].to(device)
        
        optimizer.zero_grad()
        
        
        z1,_ = model(aug1_batch)
        z2,_ = model(aug2_batch)
        
        
        loss = contrastive_loss(z1, z2, temperature=0.1)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

#Clustering Module    
def train_cluster(model, dataloader, optimizer, n_epochs=50):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            z,_ = model.encoder(data[0].to(device))
            embeddings.append(z.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    early_stopping = EarlyStopping(patience=10) 
    
    kmeans = KMeans(n_clusters=model.n_clusters).fit(embeddings.numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    
    previous_labels = None
    for epoch in range(n_epochs):
        
        total_loss = 0
        for data in dataloader:
            data = data[0].to(device)
            optimizer.zero_grad()
            
            z, q,recon= model(data)
            p = model.target_distribution(q.detach())
            
           
            cluster_loss = F.kl_div(q.log(), p, reduction='batchmean')
            recon_loss = F.mse_loss(recon, data.edge_attr)
           
            loss = cluster_loss + recon_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        early_stopping(total_loss)
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        scheduler.step(total_loss/len(dataloader))
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
        
    return model 
    
def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
        
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=0,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe  
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:  # no
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # yes
            self.best_score = val_loss
            self.counter = 0  
def calculate_metrics(all_labels, all_predicted_labels):
   
    all_labels = all_labels.reshape(-1)
    all_predicted_labels = all_predicted_labels.reshape(-1)
    
   
    res = pd.DataFrame({
        'all_labels': all_labels,
        'all_predicted_labels': all_predicted_labels,
        
    })
    
    
    return res
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description="DeepVQ - CIFAR10")
    parser.add_argument("--dataset", "-d", required=True, help='Dataset')
    parser.add_argument("--resume", "-r", default='', help="Checkpoint file for resume training")
    parser.add_argument("--code_length", "-L", required=True, type=int, help="Hash code length")
    parser.add_argument("--batch_size", "-b", default=70, type=int, help="Batch size")
    parser.add_argument("--batch_size1", "-i", default=70, type=int, help="test Batch size")
    parser.add_argument("--num_epochs", "-e", default=5, type=int, help="Number of epochs")
    parser.add_argument("--gamma", "-g", default=0.5, type=float, help="Penalty parameter")
    parser.add_argument("--seq_len",  default=24, type=int, help="window parameter")
    parser.add_argument("--pred_len", default=10, type=float, help="predict parameter")
    parser.add_argument("--n1", default=128, type=float, help="dimension change")
    parser.add_argument("--n2", default=64, type=float, help="dimension change")
    parser.add_argument("--dropout1", default=0.3, type=float, help="dropout")
    parser.add_argument("--ch_ind", default=0, type=float, help="Channel Independence; True 1 False 0")
    parser.add_argument("--d_state", default=256, type=float, help="d_state parameter of Mamba")
    parser.add_argument("--d_conv", default=2, type=float, help="d_conv parameter of Mamba")
    parser.add_argument("--e_fact", default=1, type=float, help="expand factor parameter of Mamba ")
    parser.add_argument("--residual", default=1, type=float, help="residual")
    parser.add_argument("--c_in", default=12, type=float, help="featuer demension")
    parser.add_argument("--patch_len", default='6,12', type=str, help="patch size")
    parser.add_argument("--stride", default='2,4', type=str, help="stride length")
    parser.add_argument("--n_branches", default=2, type=float, help="classes of input")
    parser.add_argument("--padding_patch", default='s', type=str, help="padding")
    parser.add_argument("--numComponents", default=10, type=int, help="pca dimension")
    
    parser.add_argument('--use_pca', default=False, type=bool,
                        help='use stft transform - if absent, use delay embedding')  # can be base
    parser.add_argument('--n_fft',default = 2, type=int, help='n_fft, only needed if using stft')
    parser.add_argument('--hop_length',default =2, type=int, help='hop_length, only needed if using stft')
    
    parser.add_argument("--model", default='LCVAEv11', type=str, help="model name")
    args = parser.parse_args()
    x,y,judge= load_data(args.dataset)
    x_init = x = torch.Tensor(x)
    
    # Load training data
    root = '/PG-Mamba/code/cluster/'
    out_path = create_path(root,args.dataset)
    print(sum((y == 0) * 1.0))
    print(sum((y == 1) * 1.0))
    print(sum((y == 2) * 1.0))
    print(sum((y == 3) * 1.0))
    print(sum((y == 4) * 1.0))
    print(sum((y == 5) * 1.0))
    print(sum((y == 6) * 1.0))
    print(sum((y == 7) * 1.0))
    print(sum((y == 8) * 1.0))
    print(sum((y == 9) * 1.0))


    print('| Training data')
    print("| Data shape: {}".format(x.shape))
    print("| Data range: {}/{}".format(x.min(), x.max()))
    print("| Label range: {}/{}".format(y.min(), y.max()))
    
    
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(x.detach().cpu().numpy()))
    #test_X = torch.tensor(scaler.fit_transform(test_X.detach().cpu().numpy()))
    
    if isinstance(args.patch_len, str):
        patch_len = args.patch_len.split(',')
        patch_len = [int(i) for i in patch_len]
    if isinstance(args.stride, str):
        stride = args.stride.split(',')
        stride = [int(i) for i in stride]
        #print(self.seq_len.dtype)
    patch_num = [int((args.seq_len - patch_len[j]) / stride[j] + 1) for j in range(args.n_branches)]
    if args.padding_patch == 'end':
        patch_num = [p_n + 1 for p_n in args.patch_num]
    MSmodel = MSGLayer(q_len=patch_num[0],patch_len=patch_len[0], stride=stride[0], padding_patch=args.padding_patch,
                                  buildA_true=False, gcn_depth=2, num_nodes= args.seq_len, dropout=0.3, subgraph_size=5, node_dim=40, propalpha=0.05, tanhalpha=3,conv_channels=32,     residual_channels=32) 
    x= MSmodel(x).detach().cpu().numpy()
    
    train_graph_dataset = create_data_object_list(x)
    
    aug_dataset = AugmentedDataset(train_graph_dataset)


    train_dataloader = DataLoader(aug_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    
    encoder = MambaGNN(input_dim=patch_len[0], hidden_dim=256,batch_size=args.batch_size,patch_num=patch_num[0]).to(device)
    cluster_model = DeepClustering(encoder, n_clusters=args.code_length).to(device)
    import os

    model_path = out_path +'model.pth'
    start_time = time.time()
    if os.path.exists(model_path):
        cluster_model.load_state_dict(torch.load(model_path,map_location=device))
    
        cluster_model.eval()
    else:
    
        if judge:
            
            opt_pretrain = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-4)
        else:
            opt_pretrain = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_pretrain, patience=10, verbose=True)
        early_stopping = EarlyStopping(patience=10) 
        
        #pretrain MambaGNN
        for epoch in range(args.num_epochs):
            loss = pretrain_contrastive(encoder, train_dataloader, opt_pretrain)
            print(f"Pretrain Epoch {epoch}, Loss: {loss:.4f}")
            early_stopping(loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
            
            
        
        
        opt_cluster = torch.optim.Adam(cluster_model.parameters(), lr=1e-4)
        cluster_model = train_cluster(cluster_model, train_dataloader, opt_cluster,args.num_epochs)
        
        torch.save(cluster_model.state_dict(), out_path +'model.pth')
    
    with torch.no_grad():
        embeddings = []
        for data in train_dataloader:
            z,q,_ = cluster_model(data[0].to(device))
            embeddings.append(z.cpu())
        embeddings = torch.cat(embeddings)
    
    clf = fit_lr(embeddings.numpy(), y)
    y_pred = clf.predict(embeddings.numpy())  
    end_time = time.time()
    duration = end_time - start_time
    
    
    ari = adjusted_rand_score(y, y_pred)
    ri = rand_score(y, y_pred)
    nmi = normalized_mutual_info_score(y, y_pred)
    print(f"{ari:.4f} {ri:.4f} {nmi:.4f} {duration:.4f}")
    
    