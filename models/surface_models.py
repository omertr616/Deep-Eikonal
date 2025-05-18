import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import ring_size_mapping_sphere


def preprocess_input(x, point_xyz, mask):
    # 1 - subtract the minimal distance from the real negibors
    x = x.clone()
    mask = mask.clone()
    mask = mask.to(x.device) 
    masked_distances = x[:, 3, :].masked_fill(mask == 0, float('inf'))  # [B, N]
    min_distances = masked_distances.min(dim=1, keepdim=True)[0]       # [B, 1]
    min_distances = torch.where(min_distances == float('inf'), torch.zeros_like(min_distances), min_distances)  # [B, 1]
    x[:, 3, :] -= min_distances 
    
    # 2 - make the predicted point_xyz be the center
    coords = x[:, :3, :]  # [B, 3, 90]
    coords_centered = coords - point_xyz.unsqueeze(2)  # [B, 3, 90]

    # 3 make for every point so that the mean norm of it is 1 (not sure why - in the paper)
    l2_norms = torch.norm(coords_centered, dim=1)  # [B, 90]
    mean_l2_norm = l2_norms.mean(dim=1, keepdim=True).unsqueeze(1)  # [B, 1, 1]
    coords_scaled = coords_centered / (mean_l2_norm + 1e-8)

    # 4 Apply SO(3) rotation matrix - not sure bout this part
    # coords_t = coords_scaled.transpose(1, 2)  # [B, 90, 3]
    # mean = coords_t.mean(dim=1, keepdim=True)  # [B, 1, 3]
    # coords_t_centered = coords_t - mean
    # cov = torch.matmul(coords_t_centered.transpose(1, 2), coords_t_centered)  # [B, 3, 3]
    # eigvals = []
    # eigvecs = []
    # for b in range(cov.size(0)):
    #     e_val, e_vec = torch.linalg.eigh(cov[b].cpu())  # [3], [3, 3]
    #     eigvecs.append(e_vec.T)  # Transpose to use as rotation matrix
    # R = torch.stack(eigvecs).to(x.device)  # [B, 3, 3]
    
    # coords_aligned = torch.matmul(R, coords_scaled)  # [B, 3, 90]

    # x[:, :3, :] = coords_aligned

    x[:, :3, :] = coords_scaled  # [B, 3, 90]
    return x, min_distances


def postprocess_output(output, min_distances):
    output = output.clone()
    output += min_distances  
    return output



def preprocess_input_for_attention(x, point_xyz, mask):
    # 1 - subtract the minimal distance from the real negibors
    x = x.clone()
    mask = mask.clone()
    mask = mask.to(x.device) 
    masked_distances = x[:, 3, :].masked_fill(mask == 0, float('inf'))  # [B, N]
    min_distances = masked_distances.min(dim=1, keepdim=True)[0]       # [B, 1]
    min_distances = torch.where(min_distances == float('inf'), torch.zeros_like(min_distances), min_distances)  # [B, 1]
    x[:, 3, :] -= min_distances 
    # 2 - make the predicted point_xyz be the center
    coords = x[:, :3, :]  # [B, 3, 90]
    coords_centered = coords - point_xyz.unsqueeze(2)  # [B, 3, 90]
    x[:, :3, :] = coords_centered  # [B, 3, 90]
    # 3 - zero out all channels for masked (inactive) neighbors
    mask_expanded = mask.unsqueeze(1)  # [B, 1, 90]
    x = x * mask_expanded  # zero out everything where mask == 0

    return x, min_distances




    #------------------------------------------------------------------------------------
    #------------------Forth attempt : attention + convolution---------------------------
    #------------------------------------------------------------------------------------


class TransformerEncoderBlock2(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        #self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        #self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, input_dim]  - raw or embedded input
        attn_mask: [B, N] - optional mask (True = keep, False = pad)
        """
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~attn_mask if attn_mask is not None else None)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x



class surface_skips(nn.Module):
    def __init__(self, ring=3, input_channels=4):
        super(surface_skips, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels + 2, 16, 1)
        self.attn_block2 = TransformerEncoderBlock2(input_dim=16, embed_dim=32, num_heads=16)
        
        self.conv3 = nn.Conv1d(16, 64, 1)
        self.attn_block4 = TransformerEncoderBlock2(input_dim=64, embed_dim=128, num_heads=16)

        # Skip 1: conv1 -> after attn_block4
        self.conv_merge1 = nn.Conv1d(16 + 64, 128, 1)

        self.attn_block6 = TransformerEncoderBlock2(input_dim=128, embed_dim=256, num_heads=16)
        self.conv7 = nn.Conv1d(128, 256, 1)

        # Skip 2: conv3 -> after conv7
        self.conv_merge2 = nn.Conv1d(64 + 256, 256, 1)

        self.conv8 = nn.Conv1d(256, 512, 1)
        self.conv9 = nn.Conv1d(512, 1024, 1)

        # Skip 3: conv_merge1 -> after conv9
        self.conv_merge3 = nn.Conv1d(128 + 1024, 1024, 1)

        self.max_pool = nn.MaxPool1d(ring_size_mapping_sphere[ring])
        
        self.fc0 = nn.Linear(1024 + 3, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, point_xyz, mask):
        x, min_distances = preprocess_input_for_attention(x, point_xyz, mask)
        mask_for_attn = mask.clone().bool().to(x.device)
        all_false = ~mask_for_attn.any(dim=1)
        mask_for_attn[all_false, 0] = True

        xyz = x[:, :3, :]
        euclid = torch.norm(xyz - point_xyz.unsqueeze(2), dim=1, keepdim=True)
        norm_xyz = F.normalize(xyz, dim=1)
        norm_center = F.normalize(point_xyz.unsqueeze(2), dim=1)
        cos_angle = torch.sum(norm_xyz * norm_center, dim=1, keepdim=True)
        x = torch.cat([x, euclid, cos_angle], dim=1)  # [B, C+2, N]

        # Conv1
        x = self.relu(self.conv1(x))     # [B, 16, N]
        x_skip1 = x.clone()

        x = x.permute(0, 2, 1)
        x = self.attn_block2(x, attn_mask=mask_for_attn)
        x = x.permute(0, 2, 1)

        # Conv3
        x = self.relu(self.conv3(x))     # [B, 64, N]
        x_skip2 = x.clone()

        x = x.permute(0, 2, 1)
        x = self.attn_block4(x, attn_mask=mask_for_attn)
        x = x.permute(0, 2, 1)

        # Skip1: x + x_skip1 → conv_merge1
        x = torch.cat([x, x_skip1], dim=1)   # [B, 64 + 16, N]
        x = self.relu(self.conv_merge1(x))  # [B, 128, N]
        x_skip3 = x.clone()

        x = x.permute(0, 2, 1)
        x = self.attn_block6(x, attn_mask=mask_for_attn)
        x = x.permute(0, 2, 1)

        # Conv7 + Skip2
        x = self.relu(self.conv7(x))        # [B, 256, N]
        x = torch.cat([x, x_skip2], dim=1)  # [B, 256 + 64, N]
        x = self.relu(self.conv_merge2(x))  # [B, 256, N]

        # Conv8 and Conv9
        x = self.relu(self.conv8(x))        # [B, 512, N]
        x = self.relu(self.conv9(x))        # [B, 1024, N]

        # Skip3: x + x_skip3
        x = torch.cat([x, x_skip3], dim=1)  # [B, 1024 + 128, N]
        x = self.relu(self.conv_merge3(x))  # [B, 1024, N]

        # Pool
        x = x.masked_fill(mask.unsqueeze(1).to(x.device) == 0, float('-inf'))
        x = self.max_pool(x)
        x = torch.where(x == float('-inf'), torch.zeros_like(x), x)
        x = x.squeeze(-1)

        # Fully connected
        x = torch.cat([x, point_xyz], dim=1)
        x = self.dropout(x)
        x = self.relu(self.fc0(x))
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return postprocess_output(x, min_distances)
