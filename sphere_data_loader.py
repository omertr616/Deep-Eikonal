import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import utils.utils as utils
from collections import deque
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed


class SphereDataset(Dataset):
    def __init__(self, data_dir, ring_size=3, precompute_neighbors=False, cache_file="neighbor_cache.pkl"):
        self.data_dir = data_dir
        self.ring_size = ring_size
        self.max_neighbors = utils.ring_size_mapping_sphere[ring_size]
        self.precompute_neighbors = precompute_neighbors
        self.cache_file = os.path.join(data_dir, cache_file)

        # Load all files
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
        self.data = []  # List of (points, gt, graph)
        self.index_map = []  # List of (file_index, point_index)

        for file_index, file_path in tqdm(enumerate(self.files), total=len(self.files), desc="Loading files", leave=False):
            with open(file_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            points_idx = lines.index("points:") + 1
            graph_idx = lines.index("graph:")

            points_lines = lines[points_idx:graph_idx]
            points, gt = [], []
            for line in points_lines:
                x, y, z, d = map(float, line.split(","))
                points.append([x, y, z])
                gt.append(d)
            points = np.array(points, dtype=np.float64)
            gt = np.array(gt, dtype=np.float64)


            graph_lines = lines[graph_idx + 1:]
            rows, cols, dists = [], [], []
            for line in graph_lines:
                a, b, w = map(float, line.split(","))
                a, b = int(a), int(b)
                rows.extend([a, b])
                cols.extend([b, a])
                dists.extend([w, w])
            graph = csr_matrix((dists, (rows, cols)), shape=(len(points), len(points)))

            self.data.append((points, gt, graph))

            for p_idx in range(len(points)):
                self.index_map.append((file_index, p_idx))

        # Optional: precompute neighbors
        self.neighbor_cache = {}
        if self.precompute_neighbors:
            if os.path.exists(self.cache_file):
                print(f"Loading cached neighbors from {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    self.neighbor_cache = pickle.load(f)
            else:
                print("Precomputing neighbors...")
                for idx, (f_idx, p_idx) in tqdm(enumerate(self.index_map), total=len(self.index_map), desc="Precomputing neighbors", leave=False):
                    points, gt, graph = self.data[f_idx]
                    target_gt = gt[p_idx]
                    neighbors = self.get_3ring_neighbors(p_idx, graph, gt, target_gt)
                    self.neighbor_cache[idx] = neighbors

                print(f"Saving cached neighbors to {self.cache_file}")
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self.neighbor_cache, f)        


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        file_index, p = self.index_map[index]
        points, gt, graph = self.data[file_index]

        if self.precompute_neighbors:
            neighbors = self.neighbor_cache.get(index, [])
        else:
            target_gt = gt[p]
            neighbors = self.get_3ring_neighbors(p, graph, gt, target_gt)

        coords = points[neighbors]
        distances = gt[neighbors].reshape(-1, 1)
        data = np.concatenate([coords, distances], axis=1)

        #compute mask and padding if needed

        mask = np.zeros(self.max_neighbors, dtype=np.float64)
        valid_len = len(data)
        mask[:valid_len] = 1
        mask = torch.from_numpy(mask).double()  # (90)

        if len(data) < self.max_neighbors:
            pad = np.zeros((self.max_neighbors - len(data), 4), dtype=np.float64)
            data = np.vstack([data, pad])
        else:
            data = data[:self.max_neighbors]

        x = torch.from_numpy(data).double().transpose(0, 1)             # (4, 90)
        point_xyz = torch.from_numpy(points[p]).double()               # (3)

        #add extra stuff for the network
        neighbors_1ring_cords = []
        # for neighbor in neighbors:
        #     n1_ring = self.get_1ring_neighbors(neighbor, graph)
        #     n1_ring_cords = points[n1_ring]
        #     neighbors_1ring_cords.append(n1_ring_cords)
        
        #---------------------------------
        point_gt = torch.tensor([gt[p]], dtype=torch.float64)         # (1)

        return x, point_xyz, point_gt, mask

    def get_3ring_neighbors(self, p, graph, gt, target_gt):
        visited = set()
        seen = set([p])
        queue = deque([(p, 0)])
        neighbors = []

        while queue:
            current, depth = queue.popleft()
            if depth > self.ring_size:
                continue
            if current != p and gt[current] < target_gt:
                neighbors.append(current)
            for neighbor in graph[current].nonzero()[1]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return neighbors
    
    def get_1ring_neighbors(self, p, graph):
        seen = set([p])
        queue = deque([(p, 0)])
        neighbors = []

        while queue:
            current, depth = queue.popleft()
            if depth > 1:
                continue
            if current != p:
                neighbors.append(current)
            for neighbor in graph[current].nonzero()[1]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return neighbors

    def __repr__(self):
        return f"SphereDataset(num_files={len(self.files)}, total_points={len(self.index_map)}, precomputed={self.precompute_neighbors})"


