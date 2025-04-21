import torch
import torch.nn as nn
from utils.utils import ring_size_mapping_sphere
from utils.utils import torch_

# def preprocess_input(x):
#     x = x.clone()
#     min_distances = x[:, 3, :].min(dim=1, keepdim=True)[0]  # shape: [B, 1]
#     x[:, 3, :] -= min_distances 
#     return x, min_distances

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
    


class SpherePointNetRingCos(nn.Module):
    def __init__(self, ring=3, input_channels=94):
        super(SpherePointNetRingCos, self).__init__()
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

        # add arccos between each point to each other:
        # x shape: [B, 4, 90]
        xyz = x[:, :3, :]  # [B, 3, 90]
        norm = torch.norm(xyz, dim=1, keepdim=True) + 1e-8  # [B, 1, 90]
        unit_xyz = xyz / norm  # [B, 3, 90]
        dot_products = torch.einsum('bci, bcj -> bij', unit_xyz, unit_xyz)
        angles = torch.arccos(torch.clamp(dot_products, -1 + 1e-6, 1 - 1e-6))
        x_arccos = angles
        x_arccos = x_arccos.permute(0, 2, 1)  # still [B, 90, 90]
        x = torch.cat([x, x_arccos], dim=1) #shape is [B, 4+90, 90]
        #----------------


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
