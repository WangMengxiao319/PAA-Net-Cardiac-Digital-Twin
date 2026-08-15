from tkinter import Y
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu
from torchinfo import summary
from models.utils.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
# from utils.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


    
# ------------- Cross-Attention Block ------------- #
class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.attn1 = CrossAttention(query_dim, heads=heads, dim_head=dim_head, dropout=dropout) # self-attention
        self.attn2 = CrossAttention(query_dim, context_dim=context_dim,heads=heads, dim_head=dim_head, dropout=dropout) # cross-attention
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        # self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        return x
    
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # Arrange the heads for parallel computation.

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)  

# ------------- Cross-Attention Block---- End ------------- #

# ------------- FiLM (Feature-wise Linear Modulation) Block ------------- #

class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(FiLM, self).__init__()
        
        # Fully connected layers that generate the gamma and beta parameters.
        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)
        
    def forward(self, x, condition):
        # Compute the scale and shift (gamma and beta) from the conditioning features.
        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)
        
        # Scale and shift x so the conditioning features modulate the input features.
        y = gamma * x + beta 
        return y
    
    


class SphericalLeadEncoding(nn.Module):
    def __init__(self, embed_dim=64, num_freqs=8):
        """
        Convert (theta, phi) angles into a high-dimensional spatial embedding.
        :param embed_dim: Final output dimension; it must match the temporal feature dimension or be aligned by a linear layer.
        :param num_freqs: Number of frequency bands used for high-frequency encoding.
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.embed_dim = embed_dim
        
        # Define the standard physical angles for the 12 leads (theta: frontal plane, phi: horizontal plane), in degrees.
        # Order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
        angles_deg = [
            [0, 0],      # I
            [60, 0],     # II
            # [120, 0],    # III
            # [-150, 0],   # aVR
            # [-30, 0],    # aVL
            # [90, 0],     # aVF
            [0, 115],    # V1 (approximate physical mapping)
            [0, 90],     # V2
            [0, 75],     # V3
            [0, 60],     # V4
            [0, 30],     # V5
            [0, 0]       # V6 (approximately directly leftward)
        ]
        
        angles_rad = torch.tensor(angles_deg, dtype=torch.float32) * (np.pi / 180.0)

        self.register_buffer('lead_angles', angles_rad) 
    
        self.projection = nn.Linear(4 * num_freqs, embed_dim)

    def forward(self):

        theta = self.lead_angles[:, 0:1] # (8, 1)
        phi = self.lead_angles[:, 1:2]   # (8, 1)
        
        encodings = []
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            encodings.append(torch.sin(freq * theta))
            encodings.append(torch.cos(freq * theta))
            encodings.append(torch.sin(freq * phi))
            encodings.append(torch.cos(freq * phi))
            
        encoded_angles = torch.cat(encodings, dim=-1)
        
        spatial_embed = self.projection(encoded_angles)
        return spatial_embed


class AnatomicalSEBlock(nn.Module):
    def __init__(self, pe_dim=16):
        super().__init__()
        # Obtain the 3D coordinate encoding for the eight leads.
        self.pe = SphericalLeadEncoding(embed_dim=pe_dim)
        
        # Gating MLP input = signal energy (1) + spatial coordinates (pe_dim).
        self.mlp = nn.Sequential(
            nn.Linear(1 + pe_dim, pe_dim),
            nn.ReLU(),
            nn.Linear(pe_dim, 1),
            nn.Sigmoid() # Output weights in [0, 1] to scale the corresponding lead signals.
        )

    def forward(self, x):
        # x shape: [b, 8, seq_len].
        b, n_lead, seq_len = x.size()
        
        # 1. Squeeze: extract waveform energy using the mean absolute value as the lead's signal strength.
        # Shape: [b, 8, 1].
        squeeze = torch.mean(torch.abs(x), dim=2, keepdim=True)
        
        # 2. Obtain the anatomical spatial prior and expand it across the batch dimension.
        # Shape: [b, 8, pe_dim].
        spatial_pe = self.pe().unsqueeze(0).expand(b, -1, -1)
        
        # 3. Fuse the current signal features with the physical-location prior.
        # Shape: [b, 8, 1 + pe_dim].
        se_input = torch.cat([squeeze, spatial_pe], dim=2)
        
        # 4. Excitation: compute attention weights constrained by anatomical coordinates.
        # Shape: [b, 8, 1].
        attn_weights = self.mlp(se_input)
        
        # 5. Modulate the original signal by injecting a 3D spatial-coordinate bias into the temporal waveform.
        return x * attn_weights
    
    
class CRNN_v2(nn.Module):
    '''
    nh: default=256, 'size of the LSTM hidden state'
    imgH: default=8, 'the height of the input image to network'
    imgW: default=256, 'the width of the input image to network'

    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''

    def __init__(self, n_lead=8, z_dims=16):
        super(CRNN_v2, self).__init__()

        n_out = 128
        self.z_dims = z_dims

        # >>> Add the anatomical-position gating module. <<<
        self.anatomical_gate = AnatomicalSEBlock(pe_dim=16)

        self.cnn = nn.Sequential(
            nn.Conv1d(n_lead, n_out, kernel_size=16, stride=2, padding=2),
            nn.BatchNorm1d(n_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(n_out, n_out*2, kernel_size=16, stride=2, padding=2),
            nn.BatchNorm1d(n_out*2),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.rnn = BidirectionalLSTM(256, z_dims*4, z_dims*2)

    def forward(self, input):
        # input: [b, 8, seq_len]
        
        # >>> Directionally modulate the input with spatial coordinates before CNN mixing. <<<
        input = self.anatomical_gate(input)

        # Convolutional features (original logic unchanged).
        conv = self.cnn(input) # [b, c, w] 
        b, c, w = conv.size()
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # RNN features (original logic unchanged).
        output = self.rnn(conv).permute(1, 0, 2) 
        features = torch.max(output, 1)[0]   # Temporal Global Max Pooling [b, z_dims*2]
    
        return features
    
class PAA_Net(nn.Module):
    def __init__(self, in_ch=3+4, out_ch=3, num_input=1024, z_dims=16, fusion_method = 'concat'):
        super(PAA_Net, self).__init__()
        self.fusion_method = fusion_method
        
        self.z_dims = z_dims

        # PointNet++ Encoder
        self.sa1 = PointNetSetAbstraction(npoint=num_input, radius=0.2, nsample=64, in_channel=in_ch, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 512, 1024], False)
        self.fc11 = nn.Linear(1024*16, z_dims*2)
        
        # FUSION LAYER  
        self.cross_attn_block = CrossAttentionBlock(query_dim=self.z_dims*2, context_dim=self.z_dims*2, heads=8, dim_head=16, dropout=0.1)
        self.film_layer = FiLM(z_dims*2, z_dims*2)

        # PointNet++ Decoder
        if fusion_method == 'concat':
            self.fc12 = nn.Linear(z_dims*4, 1024)
        else:
            self.fc12 = nn.Linear(z_dims*2, 1024)
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, out_ch, 1) 

        # self.decoder_geometry = BetaVAE_Decoder(num_input, num_input//4, in_ch, z_dims) # in_ch -> out_ch*3
        
        self.encoder_signal = CRNN_v2()

        # decode for signal
        # self.elu = nn.ELU(inplace=True)
        # self.fc1 = nn.Linear(z_dims, 256*2)
        # self.fc2 = nn.Linear(256*2, 512*2)
        # self.up = nn.Upsample(size=(8, 512), mode='bilinear')
        # self.deconv = DoubleDeConv(1, 1)

    def decode_signal(self, latent_z): # P(x|z, c)
        '''
        z: (bs, latent_size)
        '''
        inputs = latent_z
        f = self.elu(self.fc1(inputs))
        f = self.elu(self.fc2(f))
        u = self.up(f.reshape(f.shape[0], 1, 8, -1))
        dc = self.deconv(u)

        return dc
    
    def forward(self, partial_input, signal_input):  
        num_points = partial_input.shape[-1]
        # extract ecg features
        features_ECG = self.encoder_signal(signal_input) # [b, z_dims*2]
        
        # extract point cloud features      
        l0_xyz = partial_input[:,:3,:] 
        l0_points = partial_input[:,3:,:] 
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        features_PC = self.fc11(l3_points.view(-1, 1024*16)) # [b, z_dims*2]
        
        if self.fusion_method == 'concat':
            latent_z = torch.concat((features_PC, features_ECG), dim=1)  # fusion latent space
        elif self.fusion_method == 'cross_attention':
            # cross-attention fusion
            features_PC_expanded = features_PC.unsqueeze(1)
            features_ECG_expanded = features_ECG.unsqueeze(1)
            latent_z = self.cross_attn_block(features_PC_expanded, context=features_ECG_expanded).squeeze(1)
        elif self.fusion_method == 'cross_attention_2':
            # cross-attention fusion
            features_PC_expanded = features_PC.unsqueeze(1)
            features_ECG_expanded = features_ECG.unsqueeze(1)
            latent_z = self.cross_attn_block(features_ECG_expanded, context=features_PC_expanded).squeeze(1)
        elif self.fusion_method == 'film':
            # FiLM fusion
            latent_z = self.film_layer(features_PC, features_ECG)
            
        # segment point cloud
        anatomy_signal_feat = F.relu(self.fc12(latent_z))
        anatomy_signal_feat = anatomy_signal_feat.view(-1, 1024, 1).repeat(1, 1, num_points)      
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, anatomy_signal_feat)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        y_seg = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        y_seg = self.conv2(y_seg)   
        y_seg = nn.Softmax(dim=1)(y_seg)     

        return y_seg
    
if __name__ == '__main__':
    model = PAA_Net()
    pc = torch.randn(2, 7, 4096)
    ecg = torch.randn(2, 8, 512)
    out = model(pc, ecg)
    print(out.shape)
    # summary(model, input_size=[(2, 7, 1024), (2, 8, 512)])
