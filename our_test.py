import os
import torch
import heapq
import random
import numpy as np
import pyvista as pv
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from fast_matching_python.fast_marching import fast_marching
from models.sphere_models import SpherePointNetRing, SpherePointNetRingFeatureExtractionFirstRing, SpherePointNetRingAttention, SpherePointNetRingAttentionAndConvolution
import sys
sys.path.append("/home/kizner.gil/ron/project_ron_conv_attention0405/Deep-Eikonal")

global_local_solver = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fmm(V: np.ndarray, F: np.ndarray, source_index=0) -> np.ndarray:
    tmp_off = "/tmp/mesh.off"
    with open(tmp_off, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for vert in V:
            f.write(f"{vert[0]} {vert[1]} {vert[2]}\n")
        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    D = fast_marching.fast_marching(data_dir=tmp_off, source_index=source_index, verbose=True)  # if 'source' is the correct argument name
    return np.asarray(D)

def make_sphere(radius=1.0, nsub=3):
    # sphere = pv.Sphere(radius=radius, theta_resolution=theta_resolution//2, phi_resolution=phi_resolution)
    sphere = pv.Icosphere(radius=radius, nsub=nsub)  # More uniform

    points = sphere.points.astype(np.float32)
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


def plot_errors(h_values, *error_series, labels=None, save_path="plots/convergence_plot.png"):
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

    return torch.tensor(visited, dtype=torch.float32).T  # Shape: [4, N]


def local_solver_dijkstra(p, graph, status, u, points, local_solver_model):
    min_val = np.inf
    for neighbor in graph[p].nonzero()[1]:
        if status[neighbor] == 'Visited':
            candidate = u[neighbor] + graph[p, neighbor]
            if candidate < min_val:
                min_val = candidate
    return min_val


def local_solver_model_ring3(p, graph, status, u, points, local_solver_model): #fast
    visited_neighbors = get_3_ring_visited_neighbors(p, graph, status, u, points, local_solver_model).to(device)  # shape: (4, N) with N ≤ 90
    num_valid = visited_neighbors.shape[1]
    visited_neighbors_padded = F.pad(visited_neighbors, (0, 90 - num_valid), "constant", 0).unsqueeze(0)  # (1, 4, 90)
    mask = torch.zeros(90, dtype=torch.float32)
    mask[:num_valid] = 1
    mask = mask.unsqueeze(0)  # (1, 90)
    point_xyz = torch.tensor(points[p], dtype=torch.float32).unsqueeze(0)  # (1, 3)
    visited_neighbors_padded = visited_neighbors_padded.to(device)
    point_xyz = point_xyz.to(device)
    mask = mask.to(device)
    
    distance_estimation = local_solver_model(visited_neighbors_padded, point_xyz, mask).item()
    return distance_estimation


def FMM_with_local_solver(graph: csr_matrix, points, source_points, local_solver, local_solver_model=None):  #Implementation of the "real" FMM
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

        u[p] = local_solver(p, graph, status, u, points, local_solver_model)
        status[p] = 'Visited'

        for neighbor in graph[p].nonzero()[1]:
            if status[neighbor] == 'Unvisited':
                status[neighbor] = 'Wavefront'
                u[neighbor] = local_solver(neighbor, graph, status, u, points, local_solver_model)
                heapq.heappush(heap, (u[p] + graph[p, neighbor], neighbor))
            elif status[neighbor] == 'Wavefront':
                new_distance = local_solver(neighbor, graph, status, u, points, local_solver_model)
                if new_distance < u[neighbor]:
                    u[neighbor] = new_distance
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
    nsub_radius = make_spheres_radiuses_set(nsub=3, start_h=0.1, end_h=1.2, step=0.2) 
    h_vals = []
    l1_errors_dijkstra = []
    l1_errors_FMM=[]
    l1_errors_saar_model = []#
    l1_errors_our_model=[]
    np.random.seed(0)
    for n_r in nsub_radius:
        radius = n_r[0]
        nsub = n_r[1]
        #points = coordinates of each point
        #graph = (number of node a, number of node b, edge length between a and b)
        graph, faces, points, sphere, h = make_sphere(radius=radius, nsub=nsub) #radius=10, nsub=r
        source_idxs = np.random.choice(len(points), size=3, replace=False) #randomly select points from the sphere
        print(f"h: {h}")
        #Load the local solver (the model)
        local_solver_saar = SpherePointNetRing()
        local_solver_our = SpherePointNetRingAttentionAndConvolution()
        checkpoint_saar = torch.load(saar_model_path, map_location=device)
        checkpoint_our = torch.load(our_model_path, map_location=device)
        local_solver_saar.load_state_dict(checkpoint_saar["model_state_dict"])
        local_solver_saar = local_solver_saar.to(device)
        local_solver_saar.eval()
        local_solver_our.load_state_dict(checkpoint_our["model_state_dict"])
        local_solver_our = local_solver_our.to(device)
        local_solver_our.eval()
    
        l1_losses_dijkstra = []
        l1_losses_FMM = []
        l1_losses_saar_model=[]
        l1_losses_our_model=[]
        for source_idx in source_idxs:
            distances_dijkstra = FMM_with_local_solver(graph, points, [source_idx], local_solver_dijkstra)
            distances_FMM = fmm(V=points, F=faces, source_index=source_idx)
            distances_saar_model = FMM_with_local_solver(graph, points, [source_idx], local_solver_model_ring3, local_solver_saar)
            distances_our_model = FMM_with_local_solver(graph, points, [source_idx], local_solver_model_ring3, local_solver_our)
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



def evaluate_algorithms_on_mesh_with_ground_truth(
    graph, faces, points, h, source_idx, true_geodesic,
    saar_model_path, our_model_path
):
    """
    Evaluate 4 distance solvers against provided true geodesic for one source.

    Parameters:
    - graph, faces, points: Mesh representation
    - h: Mean edge length
    - source_idx: Index of source point
    - true_geodesic: Ground truth distance array (np.ndarray of shape [N])
    - saar_model_path, our_model_path: Paths to model checkpoints
    """
    print(f"▶ Evaluating on mesh with h = {h:.5f}, source_idx = {source_idx}, #points = {len(points)}")

    # Load models
    local_solver_saar = SpherePointNetRing()
    local_solver_our = SpherePointNetRingAttentionAndConvolution()

    checkpoint_saar = torch.load(saar_model_path, map_location=device)
    checkpoint_our = torch.load(our_model_path, map_location=device)

    local_solver_saar.load_state_dict(checkpoint_saar["model_state_dict"])
    local_solver_our.load_state_dict(checkpoint_our["model_state_dict"])

    local_solver_saar = local_solver_saar.to(device).eval()
    local_solver_our = local_solver_our.to(device).eval()

    # Run all solvers
    distances_dijkstra = FMM_with_local_solver(graph, points, [source_idx], local_solver_dijkstra)
    distances_FMM = fmm(V=points, F=faces, source_index=source_idx)
    distances_saar_model = FMM_with_local_solver(graph, points, [source_idx], local_solver_model_ring3, local_solver_saar)
    distances_our_model = FMM_with_local_solver(graph, points, [source_idx], local_solver_model_ring3, local_solver_our)

    # Compute L1 errors
    l1_dijkstra = np.mean(np.abs(distances_dijkstra - true_geodesic))
    l1_FMM = np.mean(np.abs(distances_FMM - true_geodesic))
    l1_saar = np.mean(np.abs(distances_saar_model - true_geodesic))
    l1_our = np.mean(np.abs(distances_our_model - true_geodesic))

    # Report
    print(f"✅ L1 error - Dijkstra:    {l1_dijkstra:.6f}")
    print(f"✅ L1 error - FMM (C++):   {l1_FMM:.6f}")
    print(f"✅ L1 error - Saar Model:  {l1_saar:.6f}")
    print(f"✅ L1 error - Our Model:   {l1_our:.6f}")




def make_uniform_surface_multimesh(f=None, radius=1.0, spacing=0.01, sparse_strides=[10]):
    """
    Generate a fine and coarse uniform triangular mesh over z = f(x,y)
    Returns:
        (fine_graph, fine_faces, fine_points, fine_mesh, fine_h),
        (coarse_graph, coarse_faces, coarse_points, coarse_mesh, coarse_h)
    """
    if f is None:
        f = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        
        
    # === Helper to build graph and h ===
    def build_graph_and_h(points3d, faces):
        edge_set = set()
        rows, cols, dists, edge_lengths = [], [], [], []
        for face in faces:
            for i in range(3):
                a, b = face[i], face[(i+1)%3]
                edge = tuple(sorted((a,b)))
                if edge not in edge_set:
                    edge_set.add(edge)
                    dist = np.linalg.norm(points3d[a] - points3d[b])
                    edge_lengths.append(dist)
                    rows += [a, b]
                    cols += [b, a]
                    dists += [dist, dist]
        graph = csr_matrix((dists, (rows, cols)), shape=(len(points3d), len(points3d)))
        h = np.mean(edge_lengths)
        return graph, h


    # === Fine hex grid ===
    x_range = np.arange(-radius, radius + spacing, spacing)
    y_range = np.arange(-radius, radius + spacing * np.sqrt(3)/2, spacing * np.sqrt(3)/2)
    xx, yy = np.meshgrid(x_range, y_range)
    xx[::2, :] += spacing / 2  # Hexagonal shift

    x_fine = xx.ravel()
    y_fine = yy.ravel()
    z_fine = f(x_fine, y_fine)
    points2d_fine = np.column_stack((x_fine, y_fine))
    points3d_fine = np.column_stack((x_fine, y_fine, z_fine)).astype(np.float32)

    tri_fine = Delaunay(points2d_fine)
    faces_fine = tri_fine.simplices

    mesh_fine = pv.PolyData()
    mesh_fine.points = points3d_fine
    mesh_fine.faces = np.hstack([[3, *face] for face in faces_fine])

    graph_fine, h_fine = build_graph_and_h(points3d_fine, faces_fine)

    # === Coarse mesh: subsample using stride in 2D grid ===
    sub_meshes = []
    for sparse_stride in sparse_strides:
        mask = np.zeros(xx.shape, dtype=bool)
        mask[::sparse_stride, ::sparse_stride] = True
        x_coarse = xx[mask]
        y_coarse = yy[mask]
        z_coarse = f(x_coarse, y_coarse)
        points2d_coarse = np.column_stack((x_coarse, y_coarse))
        points3d_coarse = np.column_stack((x_coarse, y_coarse, z_coarse)).astype(np.float32)

        tri_coarse = Delaunay(points2d_coarse)
        faces_coarse = tri_coarse.simplices

        mesh_coarse = pv.PolyData()
        mesh_coarse.points = points3d_coarse
        mesh_coarse.faces = np.hstack([[3, *face] for face in faces_coarse])

        
        graph_coarse, h_coarse = build_graph_and_h(points3d_coarse, faces_coarse)
        
        sub_meshes.append((graph_coarse, faces_coarse, points3d_coarse, mesh_coarse, h_coarse))


    coordinates_to_fine_point_index = {tuple(coor): idx for idx, coor in enumerate(points3d_fine)}
    return ((graph_fine, faces_fine, points3d_fine, mesh_fine, h_fine), sub_meshes, coordinates_to_fine_point_index)


def test_on_surface(saar_model_path, our_model_path):
    def twist_saddle(x, y):
        a = 1.2
        b = 0.8
        twist = 2.5
        return a * x**2 - b * y**2 + 0.2 * np.sin(twist * (x + y))
    surface_func = twist_saddle
    (graph_high_res, faces_high_res, points3d_high_res, mesh_high_res, h_high_res), sub_meshes, coor_to_index = \
                make_uniform_surface_multimesh(f=surface_func, radius=5, spacing=0.01, sparse_strides=[10, 20])

    for graph_sparse, faces_sparse, points3d_sparse, mesh_sparse, h_sparse in sub_meshes:
        print(f'sparse resolution h = {h_sparse}')
        sparse_source_idx = random.randint(0, len(points3d_sparse) - 1)
        high_res_source_idx = coor_to_index[tuple(points3d_sparse[sparse_source_idx])]
        print("starting FMM")
        distances_high_res = fmm(V=points3d_high_res, F=faces_high_res, source_index=high_res_source_idx)
        sparse_points_tuples = [tuple(p) for p in points3d_sparse]
        mask = [coor_to_index[p] for p in sparse_points_tuples]
        sparse_distances = distances_high_res[mask]
        evaluate_algorithms_on_mesh_with_ground_truth(graph=graph_sparse, faces=faces_sparse, points=points3d_sparse, h=h_sparse, source_idx=sparse_source_idx, true_geodesic=sparse_distances,saar_model_path=saar_model_path, our_model_path=our_model_path)
        

if __name__ == "__main__":
    #plot_convergence_comparison_sphere(saar_model_path="checkpoints/Good_isosphere_98_epochs.pt", our_model_path="checkpoints/conv_atten_best_for_now.pt")
    test_on_surface(saar_model_path="checkpoints/Good_isosphere_98_epochs.pt", our_model_path="checkpoints/conv_atten_best_for_now.pt")







#sphere["GeodesicDistance"] = true_geodesic
#sphere.plot(scalars="GeodesicDistance", cmap="viridis", show_edges=False)
#sphere["GeodesicDistance"] = distances_regular_FMM
#sphere.plot(scalars="GeodesicDistance", cmap="viridis", show_edges=False)
#sphere["GeodesicDistance"] = distances_saar_model
#sphere.plot(scalars="GeodesicDistance", cmap="viridis", show_edges=False)


