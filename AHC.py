import random
import hashlib
import bisect
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class ACHSystem:
    def __init__(self, servers, total_capacity, k_clusters, replicas=100, model_name='all-MiniLM-L6-v2'):
        self.servers = servers
        self.k_clusters = k_clusters
        self.replicas = replicas
        self.total_capacity = total_capacity
        self.embedder = SentenceTransformer(model_name)
        self.weights = self.assign_random_capacities()
        self.vnode_rings = {}
        self.vnode_to_server = {}
        self.centroids = None
        self.server_store = {server: [] for server in self.servers}

    def assign_random_capacities(self):
        cuts = sorted(random.sample(range(1, self.total_capacity), len(self.servers) - 1))
        capacities = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [self.total_capacity - cuts[-1]]
        random.shuffle(self.servers)
        return {s: c for s, c in zip(self.servers, capacities)}

    def _hash(self, key):
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**64)

    def embed_keys(self, keys):
        return self.embedder.encode(keys, convert_to_numpy=True)

    def run_kmeans(self, embeddings):
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        self.centroids = kmeans.cluster_centers_
        return kmeans.labels_

    def build_vnode_rings(self):
        for cid in range(self.k_clusters):
            self.vnode_rings[cid] = [f'C{cid}_V{j}' for j in range(self.replicas)]

    def assign_vnodes_hrw(self):
        for cid, vnode_ids in self.vnode_rings.items():
            for vnode_id in vnode_ids:
                best_server, best_score = None, -1
                for sid in self.servers:
                    score = self._hash(vnode_id + sid) * self.weights[sid]
                    if score > best_score:
                        best_score, best_server = score, sid
                self.vnode_to_server[vnode_id] = best_server

    def save_config(self, filepath):
        config = {
            'servers': self.servers,
            'weights': self.weights,
            'centroids': self.centroids.tolist(),
            'vnode_rings': self.vnode_rings,
            'vnode_to_server': self.vnode_to_server,
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self, filepath):
        with open(filepath, 'r') as f:
            config = json.load(f)
        self.servers = config['servers']
        self.weights = config['weights']
        self.centroids = np.array(config['centroids'])
        self.vnode_rings = config['vnode_rings']
        self.vnode_to_server = config['vnode_to_server']

    def lookup_key(self, key):
        """
        Input: key (str)
        Output: server (str) assigned to the key
        """
        e = self.embedder.encode(key, convert_to_numpy=True)
        dists = np.linalg.norm(self.centroids - e, axis=1)
        cluster_id = np.argmin(dists)
        vnode_ids = self.vnode_rings[str(cluster_id)]
        vnode_hashes = sorted([(self._hash(v), v) for v in vnode_ids])
        key_hash = self._hash(key)
        pos_list = [pos for pos, v in vnode_hashes]
        idx = bisect.bisect_right(pos_list, key_hash)
        if idx == len(vnode_hashes):
            idx = 0
        vnode_id = vnode_hashes[idx][1]
        return self.vnode_to_server[vnode_id]

    def store_key(self, key):
        """
        Input: key (str)
        Action: stores key → server mapping in key_store
        """
        server = self.lookup_key(key)
        self.server_store[server].append(key)
