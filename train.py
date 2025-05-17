import torch
import traceback
from tqdm import tqdm
import torch.nn as nn
#from our_test import *
from our_test_fmm_in_python import *
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sphere_models import *
from sphere_data_loader import SphereDataset 
from utils.utils import ring_size_mapping_sphere, send_email
import os
torch.set_default_dtype(torch.float64)

# ---------- Config ----------
data_path = "generated_mix/" 
data_path = "DeepEikonal/generated_spheres/train_spheres/small_spheres/"
ring = 3
epochs = 100
batch_size = 256
lr = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
load_checkpoint = False
to_send_email = True
# ---------- Dataset ----------
dataset = SphereDataset(data_path, ring_size=ring, precompute_neighbors=True)  
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------- Model ----------
model = SpherePointNetRing(ring=ring, input_channels=4).to(device).double() #1,250,000 4 sec to 50%
model = SpherePointNetRingFeatureExtractionFirstRing(ring=ring, input_channels=4).to(device).double()
model = SpherePointNetRingAttention(ring=ring, input_channels=4).to(device).double()
model = SpherePointNetRingAttentionAndConvolution(ring=ring, input_channels=4).to(device).double()
model = SpherePointNetRingAttentionAndConvolution2(ring=ring, input_channels=4).to(device).double() #1,600,000 15 sec to 50%
model = SpherePointNetRingAttentionAndConvolution3(ring=ring, input_channels=4).to(device).double() #449,233 7 sec to 50%
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
print(f"📊 Model has {total_params:,} trainable parameters.")


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
try:
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
            torch.save({"model_state_dict": model.state_dict()}, f"DeepEikonal/checkpoints/sphere_pointnet_epoch{epoch}_loss{avg_loss}.pt")
            #Test
            plot_convergence_comparison_sphere(saar_model_path="DeepEikonal/checkpoints/Good_isosphere_98_epochs.pt", our_model_path=f"DeepEikonal/checkpoints/sphere_pointnet_epoch{epoch}_loss{avg_loss}.pt")
            #test_on_surface(saar_model_path="checkpoints/Good_isosphere_98_epochs.pt", our_model_path=f"checkpoints/sphere_pointnet_epoch{epoch}_loss{avg_loss}.pt", report_path="plots/")
            #-----
            if to_send_email:
                send_email(
                    subject="✅ New Best Model Saved",
                    body=f"Epoch {epoch + 1}, loss: {avg_loss:.6f}",
                    to_email="kizner.study@gmail.com",
                    from_email="kizner.gil@gmail.com",
                    app_password="fzov pshb ruvn brau",
                    attachment_paths=["plots/convergence_plot.png", "plots/surface_eval_results.txt"]
                )
            
except Exception as e:
    if to_send_email:
        error_message = traceback.format_exc()
        send_email(
            subject="❌ Training Crashed",
            body=f"The training script crashed.\n\nError:\n{error_message}",
            to_email="kizner.study@gmail.com",
            from_email="kizner.gil@gmail.com",
            app_password="fzov pshb ruvn brau"
        )
        raise  # re-raise the exception to print it in the console too
