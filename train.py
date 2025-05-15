import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sphere_models import *
from sphere_data_loader import SphereDataset 
from utils.utils import ring_size_mapping_sphere
import os
torch.set_default_dtype(torch.float64)

# ---------- Config ----------
data_path = "high_resolution/mix"
ring = 3
epochs = 100
batch_size = 128
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_checkpoint = False

# ---------- Dataset ----------
dataset = SphereDataset(data_path, ring_size=ring, precompute_neighbors=True)  
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------- Model ----------
model = SpherePointNetRing(ring=ring, input_channels=4).to(device).double()
model = SpherePointNetRingFeatureExtractionFirstRing(ring=ring, input_channels=4).to(device).double()
model = SpherePointNetRingAttention(ring=ring, input_channels=4).to(device).double()
model = SpherePointNetRingAttentionAndConvolution(ring=ring, input_channels=4).to(device).double()

# ----------Load Pretrained Model ----------
if load_checkpoint:  
    checkpoint_path = "checkpoints/sphere_pointnet_epoch99_loss0.000000.pt"  
    checkpoint = torch.load(checkpoint_path, map_location=device)  
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
# ---------- Loss and Optimizer ----------
criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------- Training Loop ----------
best_loss = float("inf")
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")   
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for x, point_xyz, point_gt, mask in pbar:
        x = x.to(device).double() # [B, 4, 90]
        point_xyz = point_xyz.to(device).double() # [B, 3]
        point_gt = point_gt.to(device).double() # [B, 1]

        optimizer.zero_grad()
        output = model(x, point_xyz, mask)
        loss = criterion(output, point_gt) 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")
    # ---------- Save only if improved ----------
    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, f"checkpoints/sphere_pointnet_epoch{epoch}_loss{avg_loss}.pt")