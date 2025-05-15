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

class SpherePointNetRing(nn.Module):
    def __init__(self, ring=3, input_channels=4):
        super(SpherePointNetRing, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.max_pool = nn.MaxPool1d(ring_size_mapping_sphere[ring])
        self.fc1 = nn.Linear(1024 + 3, 512)  # +3 for the prdicted point (x, y, z)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, point_xyz, mask):
        """
        x: [B, 4, 90] - neighbor features
        point_xyz: [B, 3] - the predicted point coordinates
        """
        x, min_distances = preprocess_input(x, point_xyz, mask)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        #masking out before max pool:
        mask = mask.unsqueeze(1)  # [B, 1, 90]
        mask = mask.to(x.device)
        x = x.masked_fill(mask == 0, float('-inf'))
        x = self.max_pool(x)         # [B, 1024, 1]
        x = torch.where(x == float('-inf'), torch.zeros_like(x), x)
        #---------------------------------  
        x = x.squeeze(-1)            # [B, 1024]
        x = torch.cat([x, point_xyz], dim=1)  # [B, 1027]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        output = postprocess_output(x, min_distances)
        return output
    


#---------------------------------------------------
#------------------Second attempt-------------------
#---------------------------------------------------
def preprocess_input_for_FeatureExtractionFirstRing(x, point_xyz, mask):
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
    x[:, :3, :] = coords_scaled  # [B, 3, 90]

    # 4 Sort the nighbors by distance + sort the masking accordingly
    distances = x[:, 3, :]  # [B, 90]
    distances_masked = distances.masked_fill(mask == 0, float('inf'))  # [B, 90]
    sorted_indices = distances_masked.argsort(dim=1)  # [B, 90], smallest to largest
    B = x.shape[0]
    idx_expanded = sorted_indices.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, 4, 90]
    x_sorted = torch.gather(x, dim=2, index=idx_expanded)
    mask_sorted = torch.gather(mask, dim=1, index=sorted_indices)
    return x_sorted, min_distances, mask_sorted

class SpherePointNetRingFeatureExtractionFirstRing(nn.Module):
    def __init__(self, ring=3, input_channels=4):
        super(SpherePointNetRingFeatureExtractionFirstRing, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 2, padding='same')
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.max_pool = nn.MaxPool1d(ring_size_mapping_sphere[ring])
        self.fc1 = nn.Linear(1024 + 3, 512)  # +3 for the prdicted point (x, y, z)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, point_xyz, mask):
        """
        x: [B, 4, 90] - neighbor features
        point_xyz: [B, 3] - the predicted point coordinates
        """
        x, min_distances, mask = preprocess_input_for_FeatureExtractionFirstRing(x, point_xyz, mask)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        #masking out before max pool:
        mask = mask.unsqueeze(1)  # [B, 1, 90]
        mask = mask.to(x.device)
        x = x.masked_fill(mask == 0, float('-inf'))
        x = self.max_pool(x)         # [B, 1024, 1]
        x = torch.where(x == float('-inf'), torch.zeros_like(x), x)
        
        #---------------------------------  
        x = x.squeeze(-1)            # [B, 1024]
        x = torch.cat([x, point_xyz], dim=1)  # [B, 1027]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        output = postprocess_output(x, min_distances)
        return output
    

#---------------------------------------------------
#-------------Third attempt : attention-------------
#---------------------------------------------------
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

class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ffn_hidden, dropout=0.1):
        super().__init__()

        self.project_input = (
            nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        )

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        #self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        #self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, input_dim]  - raw or embedded input
        attn_mask: [B, N] - optional mask (True = keep, False = pad)
        """
        x = self.project_input(x)  # now [B, N, embed_dim]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~attn_mask if attn_mask is not None else None)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SpherePointNetRingAttention(nn.Module):   #Amphrically BAD!
    def __init__(self, ring=3, input_channels=4):
        super(SpherePointNetRingAttention, self).__init__()
        self.attn_block1 = TransformerEncoderBlock(input_dim=input_channels, embed_dim=64, num_heads=4, ffn_hidden=64, dropout=0.1)
        self.attn_block2 = TransformerEncoderBlock(input_dim=64, embed_dim=128, num_heads=4, ffn_hidden=256, dropout=0.1)
        self.attn_block3 = TransformerEncoderBlock(input_dim=128, embed_dim=256, num_heads=4, ffn_hidden=512, dropout=0.1)
        self.attn_block4 = TransformerEncoderBlock(input_dim=256, embed_dim=512, num_heads=4, ffn_hidden=512, dropout=0.1)

        self.max_pool = nn.MaxPool1d(ring_size_mapping_sphere[ring])
        self.fc1 = nn.Linear(512 + 3, 256)  # +3 for the prdicted point (x, y, z)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, point_xyz, mask):
        """
        x: [B, 4, 90] - neighbor features
        point_xyz: [B, 3] - the predicted point coordinates
        """
        mask_for_attn = mask.clone().bool().to(x.device)  
        all_false = ~mask_for_attn.any(dim=1) #Checks if there is a batch where all values are masked out which is a problem..
        mask_for_attn[all_false, 0] = True  #for that batch, set the first value to True (keep it)
        x, min_distances = preprocess_input_for_attention(x, point_xyz, mask)
        x = x.permute(0, 2, 1)
        x = self.attn_block1(x, attn_mask=mask_for_attn)
        x = self.attn_block2(x, attn_mask=mask_for_attn)
        x = self.attn_block3(x, attn_mask=mask_for_attn)
        x = self.attn_block4(x, attn_mask=mask_for_attn)
        x = x.permute(0, 2, 1) 
        #masking out before max pool:
        mask = mask.unsqueeze(1)  # [B, 1, 90]
        mask = mask.to(x.device)
        x = x.masked_fill(mask == 0, float('-inf'))
        x = self.max_pool(x)   #[B, 512, 90] -> [B, 512, 1]
        x = torch.where(x == float('-inf'), torch.zeros_like(x), x)
        #---------------------------------  
        x = x.squeeze(-1)           
        x = torch.cat([x, point_xyz], dim=1)  
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        output = postprocess_output(x, min_distances)
        return output
    

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

class SpherePointNetRingAttentionAndConvolution(nn.Module):
    def __init__(self, ring=3, input_channels=4):
        super(SpherePointNetRingAttentionAndConvolution, self).__init__()
        # self.attn_block1 = TransformerEncoderBlock(input_dim=input_channels, embed_dim=64, num_heads=4, ffn_hidden=64, dropout=0.1)
        # self.attn_block2 = TransformerEncoderBlock(input_dim=64, embed_dim=128, num_heads=4, ffn_hidden=256, dropout=0.1)
        # self.conv3 = nn.Conv1d(128, 256, 3, padding='same')
        # self.conv4 = nn.Conv1d(256, 512, 3, padding='same')

        self.conv1 = nn.Conv1d(input_channels + 2, 16, 1)
        self.attn_block2 = TransformerEncoderBlock2(input_dim=16, embed_dim=32, num_heads=16)# changed num heads to 16
        self.conv3 = nn.Conv1d(16, 64, 1)
        self.attn_block4 = TransformerEncoderBlock2(input_dim=64, embed_dim=128, num_heads=16)
        self.conv5 = nn.Conv1d(64, 128, 1)
        self.attn_block6 = TransformerEncoderBlock2(input_dim=128, embed_dim=256, num_heads=16)
        self.conv7 = nn.Conv1d(128, 256, 1)
        self.conv8 = nn.Conv1d(256, 512, 1)
        self.conv9 = nn.Conv1d(512, 1024, 1)

        self.max_pool = nn.MaxPool1d(ring_size_mapping_sphere[ring])
        
        self.fc0 = nn.Linear(1024 + 3, 512)  # +3 for the prdicted point (x, y, z)
        self.fc1 = nn.Linear(512, 256)  # +3 for the prdicted point (x, y, z)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, point_xyz, mask):
        """
        x: [B, 4, 90] - neighbor features
        point_xyz: [B, 3] - the predicted point coordinates
        """
        # x, min_distances, mask_sorted = preprocess_input_for_FeatureExtractionFirstRing(x, point_xyz, mask)
        # mask_for_attn = mask_sorted.clone().bool().to(x.device)
        # all_false = ~mask_for_attn.any(dim=1) #Checks if there is a batch where all values are masked out which is a problem..
        # mask_for_attn[all_false, 0] = True  #for that batch, set the first value to True (keep it)
        
        ##############################################################
        # Try adding noise to the input
        # x = x + 0.001 * torch.randn_like(x)
        # point_xyz = point_xyz + 0.001 * torch.randn_like(point_xyz)
        ##############################################################

        x, min_distances = preprocess_input_for_attention(x, point_xyz, mask)
        mask_for_attn = mask.clone().bool().to(x.device)
        all_false = ~mask_for_attn.any(dim=1)
        mask_for_attn[all_false, 0] = True

        #--------------------manual feature extraction:--------------------
        # coords = x[:, :3, :]  # [B, 3, 90]
        # distances = x[:, 3:, :]  # [B, 1, 90]
        # batch_size, _,  _ = x.shape
        # real_neighbor_counts = mask.sum(dim=1).long().tolist()
        # curvatures = torch.zeros((batch_size, 1, 90), device=x.device)
        # for b in range(batch_size):
        #     for i in range(real_neighbor_counts[b]):
        #         neighbors_coords = fitst_ring_neighbors[b][i]  # shape [k_i, 3]
        #         neighbors_tensor = torch.tensor(neighbors_coords, dtype=torch.float32, device=x.device)  # [k_i, 3]
        #         center = coords[b, :, i].unsqueeze(0)  # [1, 3]
        #         deltas = neighbors_tensor - center  # [k_i, 3]
        #         mean_delta = deltas.mean(dim=0, keepdim=True)  # [1, 3]
        #         H = torch.norm(mean_delta)  # scalar
        #         curvatures[b, 0, i] = H
        # x = torch.cat([coords, distances, curvatures], dim=1)
        # Tensor after manual feature extraction: [B, 5, 90]
        xyz = x[:, :3, :]                          # [B, 3, 90]
        euclid = torch.norm(xyz - point_xyz.unsqueeze(2), dim=1, keepdim=True)  # [B, 1, 90]
        norm_xyz = F.normalize(xyz, dim=1)
        norm_center = F.normalize(point_xyz.unsqueeze(2), dim=1)
        cos_angle = torch.sum(norm_xyz * norm_center, dim=1, keepdim=True)  # [B, 1, 90]
        x = torch.cat([x, euclid, cos_angle], dim=1)
        #------------------------------------------------------------------

        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x = self.attn_block2(x, attn_mask=mask_for_attn)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        x = self.attn_block4(x, attn_mask=mask_for_attn)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv5(x))
        x = x.permute(0, 2, 1)
        x = self.attn_block6(x, attn_mask=mask_for_attn)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        #masking out before max pool:
        mask = mask.unsqueeze(1)  # [B, 1, 90]
        mask = mask.to(x.device)
        x = x.masked_fill(mask == 0, float('-inf'))
        x = self.max_pool(x)   #[B, 512, 90] -> [B, 512, 1]
        x = torch.where(x == float('-inf'), torch.zeros_like(x), x)
        #---------------------------------  
        x = x.squeeze(-1)           
        x = torch.cat([x, point_xyz], dim=1)  
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        output = postprocess_output(x, min_distances)
        return output