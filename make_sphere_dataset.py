import torch
import math
import pyvista as pv
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import os
import random

# ---------- Config ----------
OUTPUT_DIR = "high_resolution/generated_spheres/train_spheres"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_sphere(radius=1, nsub=4):
    # sphere = pv.Sphere(radius=radius, theta_resolution=theta_resolution, phi_resolution=phi_resolution)
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

def true_distances(points, radius, source_idx):
    points = points.astype(np.float64)
    norm_points = points / np.linalg.norm(points, axis=1, keepdims=True)
    norm_source = norm_points[source_idx]
    cosines = np.sum(norm_points * norm_source, axis=1)
    cosines = np.clip(cosines, -1.0, 1.0)
    true_distances = radius * np.arccos(cosines)
    true_distances[source_idx] = 0.0

    return true_distances

def make_spheres_radiuses_set(nsub=4, start_h=0.1, end_h=1.6, step=0.1):   #to calculate desired h and get r,nsub
    graph, faces, points, sphere, h = make_sphere(radius=1, nsub=nsub)
    radiuses = (np.arange(start_h, end_h, step) / h)  #here we change
    nsub_radises = [(r, nsub) for r in radiuses] 

    return nsub_radises

def generate_dataset(save_dir):
    for i in range (1, 7): #range for nsub
        nsub_radius = make_spheres_radiuses_set(nsub=i, start_h=0.1, end_h=1.6, step=0.1)
        for radius, nsub in nsub_radius: #for the specific nsub, play with the radius to get h between 0.1 and something
            for j in range(10): #make x random sources for this ball
                graph, faces, points, sphere, h = make_sphere(radius=radius, nsub=nsub)
                source_idx = random.randint(0, len(points) - 1)
                distances = true_distances(points, radius, source_idx)

                filename = f"sphere_radius_{radius}_nsub_{nsub}_source_{source_idx}.txt"
                filepath = os.path.join(save_dir, filename)

                with open(filepath, "w") as f:
                    f.write(f"source_idx: {source_idx}\n")
                    f.write(f"h_value: {h:.17f}\n\n")

                    f.write("points:\n")
                    for p, d in zip(points, distances):
                        f.write(f"{p[0]:.17f}, {p[1]:.17f}, {p[2]:.17f}, {d:.17f}\n")

                    f.write("\ngraph:\n")
                    coo = graph.tocoo()
                    for a, b, w in zip(coo.row, coo.col, coo.data):
                        if a < b:  # write each edge once
                            f.write(f"{a}, {b}, {w:.17f}\n")

                print(f"✓ Saved {filename} (h = {h:.17f}, points = {len(points)})")


        
def test_sphere_dataset(filepath, radius=10.0, tol=1e-2):
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()

    print(f"Testing {os.path.basename(filepath)}...")

    assert lines[0].startswith("source_idx:"), "Missing source_idx"
    source_idx = int(lines[0].split(":")[1].strip())

    assert lines[1].startswith("h_value:"), "Missing h_value"
    h_val = float(lines[1].split(":")[1].strip())
    assert h_val > 0, "h_value must be positive"

    points_start = lines.index("points:") + 1
    graph_start = lines.index("graph:")
    point_lines = lines[points_start:graph_start]
    graph_lines = lines[graph_start + 1:]

    points = []
    for line in point_lines:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        try:
            values = list(map(float, line.split(",")))
            assert len(values) == 4, f"Point line malformed: {line}"
            x, y, z, dist = values
        except Exception as e:
            raise ValueError(f"Failed to parse line in points: '{line}'\nError: {e}")

        r = math.sqrt(x**2 + y**2 + z**2)
        assert abs(r - radius) < tol, f"Point not on sphere: {x,y,z}"
        assert dist >= 0, f"Negative geodesic distance: {dist}"
        points.append((x, y, z, dist))


    assert 0 <= source_idx < len(points), f"Invalid source_idx: {source_idx}"
    source = np.array(points[source_idx][:3])
    source_unit = source / np.linalg.norm(source)

    for idx, (x, y, z, reported_dist) in tqdm(enumerate(points), total=len(points), desc="Checking geodetic distances"):
        p = np.array([x, y, z])
        p_unit = p / np.linalg.norm(p)
        dot = np.clip(np.dot(source_unit, p_unit), -1.0, 1.0)
        geo_dist = radius * math.acos(dot)
        error = abs(geo_dist - reported_dist)
        assert error < tol, f"Geodetic mismatch at idx={idx}: reported={reported_dist:.4f}, calc={geo_dist:.4f}, error={error:.4f}"

    print(f"{os.path.basename(filepath)} passed all checks including geodetic distance.")

    for line in tqdm(graph_lines, desc="Checking graph edge distances"):
        parts = line.strip().split(",")
        if not line or len(parts) != 3:
            continue  

        i, j, w = int(parts[0]), int(parts[1]), float(parts[2])
        assert 0 <= i < len(points), f"Invalid node index in graph: {i}"
        assert 0 <= j < len(points), f"Invalid node index in graph: {j}"
        assert w >= 0, f"Edge weight negative: {w}"

        p1 = np.array(points[i][:3])
        p2 = np.array(points[j][:3])
        euclidean_dist = np.linalg.norm(p1 - p2)

        error = abs(euclidean_dist - w)
        assert error < tol, (
            f"Edge distance mismatch between {i} and {j}: "
            f"reported={w:.4f}, computed={euclidean_dist:.4f}, rel_error={error:.4f}"
        )

def test_all_generated_spheres(directory, radius=10.0, tol=0.001):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            test_sphere_dataset(filepath, radius, tol)

if __name__ == "__main__":
    generate_dataset(OUTPUT_DIR)
    #test_all_generated_spheres(OUTPUT_DIR, radius=RADIUS, tol=0.001)