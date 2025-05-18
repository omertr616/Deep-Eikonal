import os
import torch
import heapq
import random
import numpy as np
import pyvista as pv
from utils.utils import *
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from models.sphere_models import *
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from sphere_data_loader import SphereDataset
torch.set_default_dtype(torch.float64)

global_local_solver = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------------------------------------------------------------------
#--------------------------Helper functions----------------------------------
#----------------------------------------------------------------------------

def make_sphere(radius=1.0, nsub=3):
    # sphere = pv.Sphere(radius=radius, theta_resolution=theta_resolution//2, phi_resolution=phi_resolution)
    sphere = pv.Icosphere(radius=radius, nsub=nsub)  # More uniform

    points = sphere.points.astype(np.float64)
    faces = sphere.faces.reshape(-1, 4)[:, 1:]

    n_points = len(points)
    edge_set = set()
    rows, cols, dists = [], [], []
    edge_lengths = []

    for tri in faces:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            edge = tuple(sorted((a, b)))
            if edge not in edge_set:
                edge_set.add(edge)
                dist = np.linalg.norm(points[a] - points[b])
                edge_lengths.append(dist)  
                rows.extend([a, b])
                cols.extend([b, a])
                dists.extend([dist, dist])

    graph = csr_matrix((dists, (rows, cols)), shape=(n_points, n_points))
    h = np.mean(edge_lengths)

    return graph, faces, points, sphere, h


def plot_errors(h_values, *error_series, labels=None, save_path="plots/convergence_plot_omer_epoch9.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    for i, errors in enumerate(error_series):
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        plt.loglog(h_values, errors, marker='o', label=label)

    plt.xlabel('Mean edge length h')
    plt.ylabel('L1 Loss')
    plt.title('Convergence Plot (Log-Log)')
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to: {save_path}")



def get_3_ring_visited_neighbors(p, graph, status, u, points, local_solver_model):
    visited = []
    queue = deque([(p, 0)])  # (node, depth)
    seen = set([p])
    
    while queue:
        current, depth = queue.popleft()
        if depth > 3:
            continue
        if status[current] == 'Visited' and current != p:
            point_xyz = tuple(points[current])  # (x, y, z)
            visited.append([point_xyz[0], point_xyz[1], point_xyz[2], u[current]])
        for neighbor in graph[current].nonzero()[1]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))

    return torch.tensor(visited, dtype=torch.float64).T  # Shape: [4, N] 


#----------------------------------------------------------------------------
#---------------------------Local solvers------------------------------------
#----------------------------------------------------------------------------

                          
def local_solver_dijkstra(p, graph, status, u, points, faces, local_solver_model):
    min_val = np.inf
    for neighbor in graph[p].nonzero()[1]:
        if status[neighbor] == 'Visited':
            candidate = u[neighbor] + graph[p, neighbor]
            if candidate < min_val:
                min_val = candidate
    return min_val

                            
def local_solver_model_ring3(p, graph, status, u, points, faces, local_solver_model): #fast
    visited_neighbors = get_3_ring_visited_neighbors(p, graph, status, u, points, local_solver_model).to(device)  # shape: (4, N) with N ≤ 90
    num_valid = visited_neighbors.shape[1]
    visited_neighbors_padded = F.pad(visited_neighbors, (0, 90 - num_valid), "constant", 0).unsqueeze(0)  # (1, 4, 90)
    mask = torch.zeros(90, dtype=torch.float64)
    mask[:num_valid] = 1
    mask = mask.unsqueeze(0)  # (1, 90)
    point_xyz = torch.tensor(points[p], dtype=torch.float64).unsqueeze(0)  # (1, 3)
    visited_neighbors_padded = visited_neighbors_padded.to(device)
    point_xyz = point_xyz.to(device)
    mask = mask.to(device)
    
    distance_estimation = local_solver_model(visited_neighbors_padded, point_xyz, mask).item()
    return distance_estimation


def local_solver_triangle_geometric(p, graph, status, u, points, faces, local_solver_model=None):
    epsilon = 1e-7
    best_u = np.inf
    xp = points[p].astype(np.float64)

    for face in faces:
        if p not in face:
            continue

        a, b = [v for v in face if v != p]
        if status[a] != 'Visited' or status[b] != 'Visited':
            continue

        ua, ub = float(u[a]), float(u[b])
        xa = points[a].astype(np.float64)
        xb = points[b].astype(np.float64)

        va = xa - xp
        vb = xb - xp
        X = np.stack((va, vb), axis=1)

        XtX = X.T @ X
        try:
            Q = np.linalg.pinv(XtX)
        except np.linalg.LinAlgError:
            continue

        ones = np.ones(2)
        t = np.array([ua, ub])
        delta = ones @ Q @ t
        A = ones @ Q @ ones
        B = t @ Q @ t - 1
        discriminant = delta ** 2 - A * B

        if discriminant >= 0:
            p_est = (delta + np.sqrt(discriminant)) / A
            if p_est >= max(ua, ub) - epsilon and p_est < best_u:
                best_u = p_est
        else:
            # Fallback: direct edge-based
            dp0 = ua + np.linalg.norm(va)
            dp1 = ub + np.linalg.norm(vb)
            fallback_u = min(dp0, dp1)
            if fallback_u < best_u:
                best_u = fallback_u

    # Final fallback to Dijkstra-style
    if best_u == np.inf:
        best_u = min(
            u[n] + np.linalg.norm(points[p] - points[n])
            for n in range(len(points))
            if status[n] == 'Visited'
        )

    return best_u


def local_solver_triangle_geometric_nn(p, graph, status, u, points, faces, local_solver_model=None):
    epsilon = 1e-7
    best_u = np.inf
    xp = points[p].astype(np.float64)

    for face in faces:
        if p not in face:
            continue

        a, b = [v for v in face if v != p]
        if status[a] != 'Visited' or status[b] != 'Visited':
            continue

        ua, ub = float(u[a]), float(u[b])
        xa = points[a].astype(np.float64)
        xb = points[b].astype(np.float64)

        va = xa - xp
        vb = xb - xp
        X = np.stack((va, vb), axis=1)

        XtX = X.T @ X
        try:
            Q = np.linalg.pinv(XtX)
        except np.linalg.LinAlgError:
            continue

        ones = np.ones(2)
        t = np.array([ua, ub])
        delta = ones @ Q @ t
        A = ones @ Q @ ones
        B = t @ Q @ t - 1
        discriminant = delta ** 2 - A * B

        if discriminant >= 0:
            p_est = (delta + np.sqrt(discriminant)) / A
            if p_est >= max(ua, ub) - epsilon and p_est < best_u:
                best_u = p_est
        else:
            # Fallback: direct edge-based
            neighbors = get_3_ring_visited_neighbors(p, graph, status, u, points, local_solver_model).to(device)  # shape: (4, N) with N ≤ 90
            num_valid = neighbors.shape[1]
            #if num_valid >= 15:
            dp_model = local_solver_model_ring3(p, graph, status, u, points, faces, local_solver_model)
            if dp_model < best_u:
                print("ggggggggg")
                best_u = dp_model
            #dp0 = ua + np.linalg.norm(va)
            #dp1 = ub + np.linalg.norm(vb)
            #fallback_u = min(dp0, dp1)
            #if fallback_u < best_u:
            #    best_u = fallback_u

    # Final fallback to Dijkstra-style
    if best_u == np.inf:
        best_u = min(
            u[n] + np.linalg.norm(points[p] - points[n])
            for n in range(len(points))
            if status[n] == 'Visited'
        )

    return best_u

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


def FMM_with_local_solver(graph: csr_matrix, points, faces, source_points, local_solver, local_solver_model=None):  #Implementation of the "real" FMM
    n_points = graph.shape[0]
    u = np.full(n_points, np.inf)
    status = np.full(n_points, 'Unvisited', dtype=object)
    heap = []  # (u value, point index)

    for p in source_points:
        u[p] = 0
        status[p] = 'Visited'
        for neighbor in graph[p].nonzero()[1]:
            if status[neighbor] == 'Unvisited':
                status[neighbor] = 'Wavefront'
                heapq.heappush(heap, (u[p] + graph[p, neighbor], neighbor))

    while heap:
        _, p = heapq.heappop(heap)
        if status[p] == 'Visited':
            continue
        u[p] = local_solver(p, graph, status, u, points, faces, local_solver_model)
        status[p] = 'Visited'

        # for neighbor in graph[p].nonzero()[1]:
        #     if status[neighbor] == 'Unvisited':
        #         status[neighbor] = 'Wavefront'
        #         u[neighbor] = local_solver(neighbor, graph, status, u, points, faces, local_solver_model)
        #         heapq.heappush(heap, (u[p] + graph[p, neighbor], neighbor))
        #     elif status[neighbor] == 'Wavefront':
        #         new_distance = local_solver(neighbor, graph, status, u, points, faces, local_solver_model)
        #         if new_distance < u[neighbor]:
        #             u[neighbor] = new_distance
        #             heapq.heappush(heap, (u[p] + graph[p, neighbor], neighbor))
            
        for neighbor in graph[p].nonzero()[1]:
            if status[neighbor] == 'Unvisited' or status[neighbor] == 'Wavefront':
                status[neighbor] = 'Wavefront'
                u[neighbor] = local_solver(neighbor, graph, status, u, points, faces, local_solver_model)
                heapq.heappush(heap, (u[p] + graph[p, neighbor], neighbor))
    return u

def make_spheres_radiuses_set(nsub=3, start_h=0.1, end_h=1.6, step=0.1):   #to calculate desired h and get r,nsub
    graph, faces, points, sphere, h = make_sphere(radius=1, nsub=nsub)
    radiuses = (np.arange(start_h, end_h, step) / h)  #here we change
    nsub_radises = [(r, nsub) for r in radiuses] 
    return nsub_radises

def true_distances(points, radius, source_idx):
    points = points.astype(np.float64)
    norm_points = points / np.linalg.norm(points, axis=1, keepdims=True)
    norm_source = norm_points[source_idx:source_idx+1]
    cosines = np.einsum('ij,ij->i', norm_points, norm_source)
    cosines = np.clip(cosines, -1.0, 1.0)
    dists = radius * np.arccos(cosines)
    dists[source_idx] = 0.0
    return dists

def plot_convergence_comparison_sphere(saar_model_path, our_model_path):
    cache_dir = "plots"
    os.makedirs(cache_dir, exist_ok=True)
    # nsub_radius = make_spheres_radiuses_set(nsub=2, start_h=0.1, end_h=1.0, step=0.1) 
    nsub_radius = [(1.0, 1), (1.0,2), (1.0,3), (1.0,4), (1.0,5)]
    h_vals = []
    l1_errors_dijkstra = []
    l1_errors_FMM=[]
    l1_errors_saar_model = []#
    l1_errors_our_model=[]
    np.random.seed(42)
    for n_r in nsub_radius:
        radius = n_r[0]
        nsub = n_r[1]
        #points = coordinates of each point
        #graph = (number of node a, number of node b, edge length between a and b)
        graph, faces, points, sphere, h = make_sphere(radius=radius, nsub=nsub) #radius=10, nsub=r
        source_idxs = np.random.choice(len(points), size=1, replace=False) #randomly select points from the sphere
        print(f"h: {h}")
        #Load the local solver (the model)
        local_solver_saar = SpherePointNetRing().double()
        #local_solver_saar = SpherePointNetRingAttentionAndConvolution3() #<-------------------------------------
        # local_solver_our = SpherePointNetRingAttentionAndConvolution3() #<-------------------------------------
        local_solver_our = SpherePointNetRingAttentionAndConvolution().double()

        checkpoint_saar = torch.load(saar_model_path, map_location=device)
        checkpoint_our = torch.load(our_model_path, map_location=device)
        local_solver_saar.load_state_dict(checkpoint_saar["model_state_dict"])
        local_solver_saar = local_solver_saar.to(device).eval()
        local_solver_saar.eval()
        local_solver_our.load_state_dict(checkpoint_our["model_state_dict"])
        local_solver_our = local_solver_our.to(device).eval()
        local_solver_our.eval()
    
        l1_losses_dijkstra = []
        l1_losses_FMM = []
        l1_losses_saar_model=[]
        l1_losses_our_model=[]
        for source_idx in source_idxs:
            key = f"r{radius:.4f}_src{source_idx}"
            dijkstra_path = os.path.join(cache_dir, f"{key}_dijkstra.npy")
            fmm_path = os.path.join(cache_dir, f"{key}_fmm.npy")
            
            # cache Dijkstra 
            if os.path.exists(dijkstra_path):
                distances_dijkstra = np.load(dijkstra_path)
            else:
                # distances_dijkstra = FMM_with_local_solver(graph, points, faces, [source_idx], local_solver_dijkstra)
                from make_surface_dataset import fmm
                distances_dijkstra = fmm(V=points, F=faces, source_index=source_idx)
                np.save(dijkstra_path, distances_dijkstra)
            #------------
            # cache fmm
            if os.path.exists(fmm_path):
                distances_FMM = np.load(fmm_path)
            else:
                distances_FMM = FMM_with_local_solver(graph, points, faces, [source_idx], local_solver_triangle_geometric)
                np.save(fmm_path, distances_FMM)
            #------------
            
            # distances_saar_model = FMM_with_local_solver(graph, points, faces, [source_idx], local_solver_model_ring3, local_solver_saar)
            distances_saar_model = [1]*len(points)
            
            #distances_our_model = FMM_with_local_solver(graph, points, faces, [source_idx], local_solver_triangle_geometric_nn, local_solver_our)
            distances_our_model = FMM_with_local_solver(graph, points, faces, [source_idx], local_solver_model_ring3, local_solver_our)
            true_geodesic = true_distances(points, radius, source_idx)
            l1_losses_dijkstra.append(np.mean(np.abs(distances_dijkstra - true_geodesic)))
            l1_losses_FMM.append(np.mean(np.abs(distances_FMM - true_geodesic)))
            l1_losses_saar_model.append(np.mean(np.abs(distances_saar_model - true_geodesic)))
            l1_losses_our_model.append(np.mean(np.abs(distances_our_model - true_geodesic)))
    
        l1_loss_dijkstra = np.mean(l1_losses_dijkstra)
        l1_loss_FMM = np.mean(l1_losses_FMM)
        l1_loss_saar_model = np.mean(l1_losses_saar_model)
        l1_loss_our_model = np.mean(l1_losses_our_model)
        
        l1_errors_dijkstra.append(l1_loss_dijkstra)
        l1_errors_FMM.append(l1_loss_FMM)
        l1_errors_saar_model.append(l1_loss_saar_model)
        l1_errors_our_model.append(l1_loss_our_model)
        h_vals.append(h)

    average_slope_dijkstra = np.mean(np.diff(np.log(l1_errors_dijkstra)) / np.diff(np.log(h_vals)))
    average_slope_FMM = np.mean(np.diff(np.log(l1_errors_FMM)) / np.diff(np.log(h_vals)))
    average_slope_saar_model = np.mean(np.diff(np.log(l1_errors_saar_model)) / np.diff(np.log(h_vals)))
    average_slope_our_model = np.mean(np.diff(np.log(l1_errors_our_model)) / np.diff(np.log(h_vals)))
    print(f"h values: {h_vals}")
    
    print(f"Average slope dijkstra: {average_slope_dijkstra:.3f}")
    print(f"Average slope FMM: {average_slope_FMM:.3f}")
    print(f"Average slope saar_model: {average_slope_saar_model:.3f}")
    print(f"Average slope our_model: {average_slope_our_model:.3f}")

    plot_errors(h_vals, l1_errors_dijkstra, l1_errors_FMM, l1_errors_saar_model, l1_errors_our_model, labels=["Dijkstra", "FMM", "Saar model", "Our model"])


        
if __name__ == "__main__": #With caching!
    plot_convergence_comparison_sphere(
        saar_model_path="checkpoints/Good_isosphere_98_epochs.pt",
        our_model_path="checkpoints/unit_sphere_fine_tune/best_attnep9__epoch8_loss2.6399570543732853e-07.pt")






#sphere["GeodesicDistance"] = true_geodesic
#sphere.plot(scalars="GeodesicDistance", cmap="viridis", show_edges=False)
#sphere["GeodesicDistance"] = distances_regular_FMM
#sphere.plot(scalars="GeodesicDistance", cmap="viridis", show_edges=False)
#sphere["GeodesicDistance"] = distances_saar_model
#sphere.plot(scalars="GeodesicDistance", cmap="viridis", show_edges=False)


