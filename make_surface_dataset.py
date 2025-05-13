import numpy as np
import pyvista as pv
import os
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
import numpy as np
from fast_matching_python.fast_marching import fast_marching

# params:
# V - verteices
# F - faces
# source_index - source point index
# return distances, such that: distances[i] = source_index->...->V[i]
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
    points3d_fine = np.column_stack((x_fine, y_fine, z_fine)).astype(np.float64)

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
        points3d_coarse = np.column_stack((x_coarse, y_coarse, z_coarse)).astype(np.float64)

        tri_coarse = Delaunay(points2d_coarse)
        faces_coarse = tri_coarse.simplices

        mesh_coarse = pv.PolyData()
        mesh_coarse.points = points3d_coarse
        mesh_coarse.faces = np.hstack([[3, *face] for face in faces_coarse])

        
        graph_coarse, h_coarse = build_graph_and_h(points3d_coarse, faces_coarse)
        
        sub_meshes.append((graph_coarse, faces_coarse, points3d_coarse, mesh_coarse, h_coarse))


    coordinates_to_fine_point_index = {tuple(coor): idx for idx, coor in enumerate(points3d_fine)}
    return ((graph_fine, faces_fine, points3d_fine, mesh_fine, h_fine), sub_meshes, coordinates_to_fine_point_index)

if __name__ == "__main__":
    # 1. Radial sine ripple
    def ripple(x, y, A=1.0, k=2):
        r = np.sqrt(x**2 + y**2)
        return A * np.sin(k * r)

    # 2. Gaussian “bump” or hill
    def gaussian_hill(x, y, A=3.0, sigma=1.5):
        return A * np.exp(-((x**2 + y**2) / (2 * sigma**2)))

    # 3. Mexican hat (Ricker) wavelet
    def mexican_hat(x, y, A=3.0, sigma=1.5):
        r2 = x**2 + y**2
        s2 = sigma**2
        return A * (1 - r2/s2) * np.exp(-r2/(2*s2))

    # 4. Saddle surface (hyperbolic paraboloid)
    def saddle(x, y, a=2.0, b=2.0):
        return (x**2 / a**2) - (y**2 / b**2)

    # 5. Elliptic paraboloid (bowl)
    def paraboloid(x, y, a=2.0, b=2.0):
        return (x**2 / a**2) + (y**2 / b**2)

    # 6. Sinusoidal grid waves
    def grid_waves(x, y, A=0.5, kx=3.0, ky=3.0):
        return A * (np.sin(kx * x) + np.sin(ky * y))

    # 7. Checkerboard of Gaussian bumps
    def bump_grid(x, y, A=1.0, sigma=0.5, spacing=3):
        # place little Gaussians at grid centers
        xi = np.round(x/spacing) * spacing
        yi = np.round(y/spacing) * spacing
        return A * np.exp(-(((x-xi)**2 + (y-yi)**2) / (2*sigma**2)))

    # 11. User‐combination: mix two surfaces
    def mix_surfaces(f1, f2, alpha=0.5):
        return lambda x, y: alpha*f1(x, y) + (1-alpha)*f2(x, y)
    
    def omer_test(x, y, A=1.0, k=0.4):
        r = x**2 + y**2
        return A * np.sin(k * r)

    
    import os
    import random

    OUTPUT_DIR = "generated_surfaces_high_res/train_surfaces"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Base surface functions and parameters
    param_grid = {
        ripple: [{'A': A, 'k': k} for A in [0.5, 1.0] for k in [1, 2, 2.5]],
        gaussian_hill: [{'A': A, 'sigma': s} for A in [1, 3] for s in [1.0, 1.5, 2.0]],
        mexican_hat: [{'A': A, 'sigma': s} for A in [1, 3] for s in [1.0, 1.5, 2.0]],
        saddle: [{'a': a, 'b': b} for a in [1.5, 2.0] for b in [1.5, 2.0]],
        paraboloid: [{'a': a, 'b': b} for a in [1.5, 2.0] for b in [1.5, 2.0]],
        grid_waves: [{'A': A, 'kx': kx, 'ky': ky} for A in [0.3, 0.6] for kx in [2, 3] for ky in [2, 3]],
        omer_test: [{'A': A, 'k': k} for A in [1.0, 0.5] for k in [0.2, 0.3, 0.4]],
    }

    # -- Mixed surface generators
    def mix1(alpha): return mix_surfaces(ripple, gaussian_hill, alpha)
    def mix2(alpha): return mix_surfaces(gaussian_hill, mexican_hat, alpha)
    def mix3(alpha): return mix_surfaces(saddle, paraboloid, alpha)
    def mix4(alpha): return mix_surfaces(grid_waves, bump_grid, alpha)
    def mix5(alpha): return mix_surfaces(ripple, bump_grid, alpha)
    def mix6(alpha): return mix_surfaces(paraboloid, grid_waves, alpha)


    # -- Main loop
    all_configs = [(f, param) for f, param_list in param_grid.items() for param in param_list]
    i = 0
    for f, param in tqdm(all_configs, desc="Generating surfaces"):
            try:
                # Detect if it's a mix (by checking for alpha)
                if "alpha" in param:
                    surface_func = f(param["alpha"])  # returns (x, y) -> z
                else:
                    surface_func = lambda x, y, f=f, param=param: f(x, y, **param)

                (graph_high_res, faces_high_res, points3d_high_res, mesh_high_res, h_high_res), sub_meshes, coor_to_index = \
                    make_uniform_surface_multimesh(f=surface_func, radius=5, spacing=0.005, sparse_strides=[10,20,30,50,60,70,80])

                print(f'{getattr(f, "__name__", str(f))} with params {param}, high resolution h = {h_high_res}')

                for graph_sparse, faces_sparse, points3d_sparse, mesh_sparse, h_sparse in sub_meshes:
                    print(f'sparse resolution h = {h_sparse}')
                    sparse_source_idx = random.randint(0, len(points3d_sparse) - 1)
                    high_res_source_idx = coor_to_index[tuple(points3d_sparse[sparse_source_idx])]

                    distances_high_res = fmm(V=points3d_high_res, F=faces_high_res, source_index=high_res_source_idx)
                    
                    sparse_points_tuples = [tuple(p) for p in points3d_sparse]
                    mask = [coor_to_index[p] for p in sparse_points_tuples]
                    sparse_distances = distances_high_res[mask]
                    
                    mesh_sparse["Distance"] = sparse_distances
                    
                    # save file:
                    filename = f"function_{f.__name__}_{i}_h_{h_sparse:.3f}_source_{sparse_source_idx}.txt"
                    i += 1
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, "w") as file:
                        file.write(f"source_idx: {sparse_source_idx}\n")
                        file.write(f"h_value: {h_sparse:.6f}\n\n")

                        file.write("points:\n")
                        for p, d in zip(points3d_sparse, sparse_distances):
                            file.write(f"{p[0]:.17f}, {p[1]:.17f}, {p[2]:.17f}, {d:.17f}\n")

                        file.write("\ngraph:\n")
                        coo = graph_sparse.tocoo()
                        for a, b, w in zip(coo.row, coo.col, coo.data):
                            if a < b:  # write each edge once
                                file.write(f"{a}, {b}, {w:.17f}\n")

                    print(f"✓ Saved {filename} (h = {h_sparse:.3f}, points = {len(points3d_sparse)})")

            except Exception as e:
                print(f'❌ Failed for {getattr(f, "__name__", str(f))} with params {param}: {e}')
