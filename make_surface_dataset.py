import numpy as np
import pyvista as pv
import os
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import dijkstra


def make_uniform_surface_multimesh(f=None, radius=1.0, spacing=0.005, sparse_stride=10):
    """
    Generate a fine and coarse uniform triangular mesh over z = f(x,y)
    Returns:
        (fine_graph, fine_faces, fine_points, fine_mesh, fine_h),
        (coarse_graph, coarse_faces, coarse_points, coarse_mesh, coarse_h)
    """
    if f is None:
        f = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)

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

    # === Coarse mesh: subsample using stride in 2D grid ===
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

    # === sparse to high res ===
    points2d_fine_list = points2d_fine.tolist()
    points2d_coarse_list = points2d_coarse.tolist()
    coarse_to_fine_map = {
        coarse_idx: points2d_fine_list.index(pt)
        for coarse_idx, pt in enumerate(points2d_coarse_list)
    }


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

    graph_fine, h_fine = build_graph_and_h(points3d_fine, faces_fine)
    graph_coarse, h_coarse = build_graph_and_h(points3d_coarse, faces_coarse)

    return (
        (graph_fine, faces_fine, points3d_fine, mesh_fine, h_fine),
        (graph_coarse, faces_coarse, points3d_coarse, mesh_coarse, h_coarse),
        coarse_to_fine_map
    )


def generate_function_list(n=100):
    funcs = []

    for i in range(n):
        choice = np.random.choice(['sin', 'gauss', 'poly', 'mix', 'saddle'])

        if choice == 'sin':
            a, b = np.random.uniform(1, 5), np.random.uniform(1, 5)
            phase_x, phase_y = np.random.uniform(0, 2*np.pi, 2)
            funcs.append(lambda x, y, a=a, b=b, phase_x=phase_x, phase_y=phase_y:
                         np.sin(a * x + phase_x) * np.sin(b * y + phase_y))

        elif choice == 'gauss':
            cx, cy = np.random.uniform(-0.5, 0.5, 2)
            sx, sy = np.random.uniform(0.1, 1.0, 2)
            amp = np.random.uniform(0.5, 2.0)
            funcs.append(lambda x, y, cx=cx, cy=cy, sx=sx, sy=sy, amp=amp:
                         amp * np.exp(-((x - cx)**2 / (2 * sx**2) + (y - cy)**2 / (2 * sy**2))))

        elif choice == 'poly':
            a, b, c = np.random.uniform(-2, 2, 3)
            funcs.append(lambda x, y, a=a, b=b, c=c: a*x**2 + b*y**2 + c*x*y)

        elif choice == 'saddle':
            a = np.random.uniform(0.5, 2.0)
            funcs.append(lambda x, y, a=a: a * (x**2 - y**2))

        elif choice == 'mix':
            a, b, c = np.random.uniform(0.5, 2.0, 3)
            funcs.append(lambda x, y, a=a, b=b, c=c:
                         a * np.sin(b * x) + c * y**2)

    return funcs


OUTPUT_DIR = "Deep-Eikonal/generated_surfaces/train_surfaces"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

funcs = generate_function_list(100)
#plot the functions
for i, func in enumerate(funcs):
    for h in tqdm(np.arange(0.03, 0.1, 0.01), desc=f"Func {i}", leave=False):
        (high_res_graph, high_res_faces, high_res_points, high_res_mesh, high_res_h), (sparse_graph, sparse_faces, sparse_points, sparse_mesh, sparse_h), mapping = make_uniform_surface_multimesh(f=func, radius=5.0, spacing=h, sparse_stride=5)
        sparse_source_idx = random.randint(0, len(sparse_points) - 1)
        filename = f"function_{i}_h_{sparse_h:.3f}_source_{sparse_source_idx}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        high_res_source_idx = mapping[sparse_source_idx]
        distances, predecessors = dijkstra(csgraph=high_res_graph, directed=False, indices=high_res_source_idx, return_predecessors=True)
        #distances = FMM_with_local_solver(high_res_graph, high_res_points, [source_idx], local_solver_ron)
        distances = np.array([distances[mapping[index]] for index in range(len(sparse_points))])
                                
        with open(filepath, "w") as f:
            f.write(f"source_idx: {sparse_source_idx}\n")
            f.write(f"h_value: {sparse_h:.6f}\n\n")

            f.write("points:\n")
            for p, d in zip(sparse_points, distances):
                f.write(f"{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}, {d:.6f}\n")

            f.write("\ngraph:\n")
            coo = sparse_graph.tocoo()
            for a, b, w in zip(coo.row, coo.col, coo.data):
                if a < b:  # write each edge once
                    f.write(f"{a}, {b}, {w:.6f}\n")

        print(f"✓ Saved {filename} (h = {sparse_h:.3f}, points = {len(sparse_points)})")


