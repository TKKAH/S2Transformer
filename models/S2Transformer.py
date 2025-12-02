import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from models.Attention import Attention
import torch.nn.functional as F
class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.fuse_lin = nn.Linear(2 * hidden_size, hidden_size)
        
    def forward(self, x,patch,attn_mask,attn_bias,patch_attn_bias):
        B,T,_,D = x.shape

        zero_tensor = torch.zeros((x.shape[0], T,1, x.shape[-1])).to(x.device)
        x = torch.cat((x, zero_tensor), dim=2)
        patch_x = x[:,:,patch,:]
        B,T,P,N,D=patch_x.shape

        qkv = self.snorm1(patch_x.reshape(B*T*P,N,D))    
        torch.diagonal(attn_mask, dim1=1, dim2=2).fill_(1)
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1,-1, -1).reshape(B * P, N,N).unsqueeze(1).bool()
        attn_bias= attn_bias.unsqueeze(0).expand(B, -1,-1, -1).reshape(B * P, N,N).unsqueeze(1).float()
        patch_attn_bias = patch_attn_bias.unsqueeze(0).expand(B, -1,-1).reshape(B, P,P).unsqueeze(1).float()

        patch_x = patch_x + self.sattn(qkv,mask=attn_mask,bias=attn_bias).reshape(B,T,P,N,D)
        patch_x = patch_x + self.smlp(self.snorm2(patch_x))
        p=self.nnorm1(patch_x.mean(dim=-2, keepdim=False).squeeze(1))
        p=p+self.nattn(p,mask=None,bias=patch_attn_bias)
        p=p+self.nmlp(self.nnorm2(p))
        p = p.unsqueeze(-2).repeat(1, 1,N, 1).unsqueeze(1)
        z = torch.cat([patch_x, p], dim=4)
        patch_x = F.relu(self.fuse_lin(z)) + patch_x

        x[:,:,patch,:] = patch_x
        x=x[:, :,:-1, :]
        return x


class Model(nn.Module):
    def __init__(self, config,node_feature,device,patch,intra_adj_mx,inter_adj_mx,tod=25, dow=8,moy=13,dom=32,layers=2,input_dims=768, 
                 node_dims=48, tod_dims=32, dow_dims=32,moy_dims=32,dom_dims=32,node_feature_learning=False):
        super(Model, self).__init__()
        self.node_num = 5672
        self.horizon= config.pred_len
        self.seq_len = config.seq_len
        self.device = device
        self.tod, self.dow = tod, dow
        
        # input_emb
        self.input_st_fc = nn.Conv2d(in_channels=1, out_channels=input_dims, kernel_size=(1, self.seq_len), stride=(1, self.seq_len), bias=True)
        # tem_emb
        self.month_in_year_emb = nn.Parameter(
                torch.empty(moy, moy_dims))
        nn.init.xavier_uniform_(self.month_in_year_emb)
        self.day_in_month_emb = nn.Parameter(
                torch.empty(dom, dom_dims))
        nn.init.xavier_uniform_(self.day_in_month_emb)
        self.time_in_day_emb = nn.Parameter(
                torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        # spa_emb
        if node_feature_learning:
            self.node_emb = nn.Parameter(
                    torch.empty(self.node_num, node_dims))
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb=torch.tensor(node_feature, dtype=torch.float32).to(self.device)
            node_dims= self.node_emb.shape[-1]
        # lonlat_emb
        self.lonlat_weight = nn.Parameter(torch.zeros(16),requires_grad=True)
        embedded_lonlat_np = np.load('dataset/embedded_lonlat.npy')
        self.lonlat_emb = torch.tensor(embedded_lonlat_np, dtype=torch.float32)

        dims = input_dims + node_dims+moy_dims+dom_dims+tod_dims+dow_dims+16
        
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 8, mlp_ratio=1) for _ in range(layers)
        ])

        self.patch= patch
        self.intra_adj_mx = intra_adj_mx
        self.inter_adj_mx = inter_adj_mx

        self.learnable_scalar=nn.Parameter(torch.zeros(self.node_num+1),requires_grad=True)
        self.patch_learnable_scalar=nn.Parameter(torch.zeros(patch[0].shape[0]),requires_grad=True)
        
        self.regression_conv = nn.Conv2d(in_channels=dims, out_channels=self.horizon, kernel_size=(1, 1), bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc=x_enc.unsqueeze(-1)
        x_mark_enc=x_mark_enc.unsqueeze(2).expand(-1, -1, x_enc.shape[2],-1).type(torch.int32)
        rex = self.embedding(x_enc,x_mark_enc)
        self.patch=[p.to(x_enc.device) for p in self.patch]
        self.intra_adj_mx = self.intra_adj_mx.to(x_enc.device)
        self.inter_adj_mx = self.inter_adj_mx.to(x_enc.device)


        for i, block in enumerate(self.spa_encoder):
            patch_mask = (self.patch[i] != self.node_num).float().unsqueeze(-1)
            attn_mask = torch.matmul(patch_mask, patch_mask.transpose(-1, -2)).int() 
            rex = block(rex,
                        self.patch[i],
                        attn_mask,
                        self.intra_adj_mx*(self.learnable_scalar[self.patch[i]].unsqueeze(-1)),
                        self.inter_adj_mx*(self.patch_learnable_scalar.unsqueeze(-1).to(rex.device)))

        pred_y = self.regression_conv(rex.permute(0,3,2,1))
        pred_y=pred_y.squeeze(-1)
        pred_y=pred_y.reshape(pred_y.shape[0],-1,1,pred_y.shape[2]).permute(0,3,1,2)
        pred_y=pred_y.squeeze(-1).permute(0,2,1)
        return pred_y

    def embedding(self, x, te):
        b,t,n,_ = x.shape
        input_data = self.input_st_fc(x.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]    

        m_i_y_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.month_in_year_emb[(m_i_y_data).type(torch.LongTensor)]], -1)

        d_i_m_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_month_emb[(d_i_m_data).type(torch.LongTensor)]], -1)

        # # cat time of day embedding
        t_i_d_data = te[:, -input_data.shape[1]:, :, 3]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # # # cat day of week embedding
        d_i_w_data = te[:, -input_data.shape[1]:, :, 2]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        # cat spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1).to(input_data.device)
        input_data = torch.cat([input_data, node_emb], -1)

        # cat lonlat embedding
        lonlat_emb = (self.lonlat_emb*self.lonlat_weight.type(torch.FloatTensor)).unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1).to(input_data.device)
        input_data = torch.cat([input_data, lonlat_emb], -1)
        return input_data