import random
import hashlib
import bisect
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class ACHSystem:
    def __init__(self, num_servers, total_capacity, k_clusters, replicas=100, model_name='all-MiniLM-L6-v2'):
        """
        Initialize ACHSystem.

        Input:
            num_servers (int): number of servers to create (server IDs will be '0', '1', ...)
            total_capacity (int): total capacity to split among servers
            k_clusters (int): number of clusters for k-means
            replicas (int): number of virtual nodes per cluster
            model_name (str): embedding model name

        Output:
            Initializes system attributes (servers, weights, rings, storage)
        """
        raw_servers = [i for i in range(num_servers)]
        self.servers = [self._hash(str(s)) for s in raw_servers]
        self.original_server_ids = raw_servers
        self.k_clusters = k_clusters
        self.replicas = replicas
        self.total_capacity = total_capacity
        self.embedder = SentenceTransformer(model_name)
        self.weights = self.assign_random_capacities()
        self.vnode_rings = {i: [] for i in range(k_clusters)}
        self.vnode_to_server = {}
        self.centroids = None
        self.server_store = {self._hash(s): [] for s in raw_servers}

    def assign_random_capacities(self):
        """
        Assign random integer capacities across servers.

        Output:
            dict: {hashed_server_id: capacity}
        """
        cuts = sorted(random.sample(range(1, self.total_capacity), len(self.servers) - 1))
        capacities = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [self.total_capacity - cuts[-1]]
        shuffled_servers = self.servers.copy()
        random.shuffle(shuffled_servers)
        return {s: c for s, c in zip(shuffled_servers, capacities)}

    def _hash(self, key):
        """
        Compute a 64-bit hash for a string.

        Output:
            int: 64-bit hash value
        """
        return int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % (2 ** 64)

    def embed_keys(self, keys):
        """
        Generate embeddings for a list of keys.

        Output:
            np.ndarray: array of embeddings (n_samples, embedding_dim)
        """
        return self.embedder.encode(keys, convert_to_numpy=True)

    def run_kmeans(self, embeddings):
        """
        Run k-means clustering on embeddings.

        Input:
            embeddings (np.ndarray): array of shape (n_samples, embedding_dim)
        """
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        self.centroids = kmeans.cluster_centers_
        

    def build_vnode_rings(self):
        """
        Build virtual node IDs for each cluster.

        Output:
            Updates self.vnode_rings (cluster_id → list of vnode IDs)
        """
        for cid in range(self.k_clusters):
            for sid in self.servers:
                for i in range(self.replicas):
                    self.vnode_rings[cid].append(self._hash(f"{cid}:{sid}:{i}"))
            self.vnode_rings[cid].sort()

    def assign_vnodes_hrw(self):
        """
        Assign each virtual node to a server using Highest Random Weight (HRW) hashing.

        Output:
            Updates self.vnode_to_server (vnode_id → hashed_server_id)
        """
        for cid, vnode_ids in self.vnode_rings.items():
            for vnode_id in vnode_ids:
                best_server, best_score = None, -1
                for sid in self.servers:
                    # Use a separator between vnode_id and sid to avoid ambiguity
                    score = self._hash(f"{vnode_id}:{sid}") * self.weights[sid]
                    if score > best_score:
                        best_score, best_server = score, sid
                self.vnode_to_server[vnode_id] = best_server


    def store(self, key, value):
        """
        Store a key-value pair in the system.
        
        Input:
            key: Identifier for the data
            value: The data to store
            
        Output:
            server_id: The ID of the server where the data is stored
        """
        # Encode the value to find the right cluster
        e = self.embedder.encode(value, convert_to_numpy=True)
        dists = np.linalg.norm(self.centroids - e, axis=1)
        cluster_id = np.argmin(dists)
        
        # Get the virtual nodes for this cluster
        vnode_ids = self.vnode_rings[cluster_id]
        
        # Hash the key and find the right vnode
        key_hash = self._hash(key)
        
        # Find the right position on the consistent hash ring
        idx = bisect.bisect_right(vnode_ids, key_hash)
        if idx == len(vnode_ids):
            idx = 0
            
        # Get the vnode ID and find the server for this vnode
        vnode_id = vnode_ids[idx]
        server_id = self.vnode_to_server[vnode_id]
        
        # Store the record
        record = {'key': key, 'value': value, 'cluster': cluster_id, 'vnode': vnode_id, 'hash': key_hash}
        self.server_store[server_id].append(record)
        return server_id

    def get(self, key):
        """
        Retrieve a value from the system based on a key.

        Input:
            key: Identifier for the data
            
        Output:
            value: The data stored in the system
        """
        # Encode the key to find the right cluster
        e = self.embedder.encode(key, convert_to_numpy=True)    
        dists = np.linalg.norm(self.centroids - e, axis=1)
        cluster_id = np.argmin(dists)
        
        # Get the virtual nodes for this cluster
        vnode_ids = self.vnode_rings[cluster_id]    
        
        # Hash the key and find the right vnode
        key_hash = self._hash(key)
        
        # Find the right position on the consistent hash ring
        idx = bisect.bisect_right(vnode_ids, key_hash)
        if idx == len(vnode_ids):
            idx = 0 

        # Get the vnode ID and find the server for this vnode
        vnode_id = vnode_ids[idx]
        server_id = self.vnode_to_server[vnode_id]
        
        # Get the records for this server
        records = self.server_store[server_id]
        
        # Find the record with the matching key
        for record in records:
            if record['key'] == key:
                return record['value']  
        
        # If no record is found, return None
        return None

    def save_config(self, filepath):
        def convert_to_python(obj):
            if isinstance(obj, dict):
                return {convert_to_python(k): convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        config = {
            'servers': self.servers,
            'original_server_ids': self.original_server_ids,
            'weights': self.weights,
            'k_clusters': self.k_clusters,
            'replicas': self.replicas,
            'total_capacity': self.total_capacity,
            'centroids': self.centroids.tolist() if self.centroids is not None else None,
            'vnode_rings': self.vnode_rings,
            'vnode_to_server': self.vnode_to_server,
            'server_store': self.server_store
        }
        with open(filepath, 'w') as f:
            json.dump(convert_to_python(config), f, indent=2)


    def load_config(self, filepath):
        """
        Load system configuration from a JSON file.
        
        Input:
            filepath (str): Path to the configuration file
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.servers = config['servers']
        self.original_server_ids = config['original_server_ids']
        self.weights = {int(k): v for k, v in config['weights'].items()}
        self.k_clusters = config['k_clusters']
        self.replicas = config['replicas']
        self.total_capacity = config['total_capacity']
        self.centroids = np.array(config['centroids']) if config['centroids'] is not None else None
        self.vnode_rings = {int(k): v for k, v in config['vnode_rings'].items()}
        self.vnode_to_server = {int(k): int(v) for k, v in config['vnode_to_server'].items()}
        self.server_store = {int(k): v for k, v in config['server_store'].items()}

    def add_server(self, capacity):
        """
        Add a new server to the system.

        Input:
            capacity (int): The capacity of the new server

        Output:
            Dictionary mapping affected old vnodes to new vnodes for data migration
        """
        # Add the new server
        new_server_id = self._hash(str(len(self.servers)))
        self.servers.append(new_server_id)
        self.original_server_ids.append(len(self.original_server_ids))

        # Update weights and initialize storage
        self.weights[new_server_id] = capacity
        self.server_store[new_server_id] = []

        # Prepare new vnodes per cluster
        new_vnodes_by_cluster = {cid: [] for cid in range(self.k_clusters)}
        for cid in range(self.k_clusters):
            for i in range(self.replicas):
                new_vnode = self._hash(f"{cid}:{new_server_id}:{i}")
                new_vnodes_by_cluster[cid].append(new_vnode)

        affected_vnodes = {}
        # Assign each new vnode via HRW and insert into ring
        for cid, new_vnodes in new_vnodes_by_cluster.items():
            # Copy current ring
            current_ring = self.vnode_rings[cid].copy()

            for new_vnode in new_vnodes:
                # HRW assignment: pick server with highest score
                best_server, best_score = None, float('-inf')
                for sid in self.servers:
                    score = self._hash(f"{new_vnode}:{sid}") * self.weights[sid]
                    if score > best_score:
                        best_score, best_server = score, sid
                self.vnode_to_server[new_vnode] = best_server

                # Find affected old vnode (consistent-ring split logic)
                idx = bisect.bisect_right(current_ring, new_vnode)
                if idx < len(current_ring):
                    old_vnode = current_ring[idx]
                elif current_ring:
                    old_vnode = current_ring[0]
                else:
                    continue

                old_server = self.vnode_to_server.get(old_vnode)
                if old_server is not None:
                    affected_vnodes[old_vnode] = {
                        'new_vnode': new_vnode,
                        'old_server': old_server,
                        'new_server': best_server,
                        'cluster': cid
                    }

                # Insert and sort
                self.vnode_rings[cid].append(new_vnode)
            self.vnode_rings[cid].sort()

        # Migrate records for affected vnodes
        for old_vnode, info in affected_vnodes.items():
            old_server = info['old_server']
            new_vnode = info['new_vnode']
            new_server = info['new_server']
            cluster_id = info['cluster']

            records_to_move, records_to_keep = [], []
            for record in self.server_store[old_server]:
                if record['vnode'] == old_vnode:
                    key_hash = record['hash']
                    vnode_ids = self.vnode_rings[cluster_id]
                    i = bisect.bisect_right(vnode_ids, key_hash)
                    if i == len(vnode_ids): i = 0
                    if vnode_ids[i] == new_vnode:
                        record['vnode'] = new_vnode
                        records_to_move.append(record)
                    else:
                        records_to_keep.append(record)
                else:
                    records_to_keep.append(record)

            # Perform migration
            self.server_store[new_server].extend(records_to_move)
            self.server_store[old_server] = records_to_keep

        return affected_vnodes


    def remove_server(self, server_id_to_remove):
        """
        Remove a server from the system, reassign its v-nodes via HRW, and migrate data.

        Input:
            server_id_to_remove: The original or hashed ID of the server to remove
        Output:
            Mapping from each removed vnode to its new assigned server
        """
        # Prevent removing the last server
        if len(self.servers) <= 1:
            raise ValueError("Cannot remove the last server from the system")

        # Resolve hashed ID if original ID given
        if isinstance(server_id_to_remove, int) and server_id_to_remove in self.original_server_ids:
            idx = self.original_server_ids.index(server_id_to_remove)
            hashed_server_id = self.servers[idx]
        else:
            hashed_server_id = server_id_to_remove
        if hashed_server_id not in self.servers:
            raise ValueError(f"Server {server_id_to_remove} not found in the system")

        # Identify v-nodes owned by this server
        removed_vnodes = []
        for cid in range(self.k_clusters):
            for vnode in self.vnode_rings[cid]:
                if self.vnode_to_server.get(vnode) == hashed_server_id:
                    removed_vnodes.append(vnode)

        # Reassign each removed vnode via HRW among remaining servers
        vnode_reassign = {}
        for vnode in removed_vnodes:
            best_server, best_score = None, float('-inf')
            for sid in self.servers:
                if sid == hashed_server_id:
                    continue
                score = self._hash(f"{vnode}:{sid}") * self.weights.get(sid, 1)
                if score > best_score:
                    best_score, best_server = score, sid
            vnode_reassign[vnode] = best_server
            self.vnode_to_server[vnode] = best_server

        # Migrate records from removed server to new servers
        records_to_migrate = {}  # new_server -> list of records
        for record in self.server_store.get(hashed_server_id, []):
            vnode = record.get('vnode')
            new_server = vnode_reassign.get(vnode)
            if new_server is None:
                continue
            record['vnode'] = vnode
            records_to_migrate.setdefault(new_server, []).append(record)

        for new_srv, recs in records_to_migrate.items():
            self.server_store.setdefault(new_srv, []).extend(recs)

        # Remove the old server from system lists
        # use original index to remove both in sync
        orig_idx = self.servers.index(hashed_server_id)
        self.servers.pop(orig_idx)
        self.original_server_ids.pop(orig_idx)
        self.server_store.pop(hashed_server_id, None)
        # weights can also be removed
        self.weights.pop(hashed_server_id, None)

        return vnode_reassign
    
    def split_vnode(self, cluster_id, vnode_id):
        """
        Split an overloaded vnode into two sub-vnodes via k=2 k-means.

        Steps (per ACH.pdf):
        1. Gather embeddings for all keys in vnode_id.
        2. Run k=2 k-means to get two subclusters.
        3. Create sub-vnode X1 -> original server, X2 -> next-best server via HRW.
        4. Update ring: replace vnode_id with X1 and X2.
        5. Migrate only keys in cluster2 to X2.
        """
        # 1. Identify records in this vnode and their embeddings
        orig_server = self.vnode_to_server[vnode_id]
        records = [rec for rec in self.server_store[orig_server] if rec['vnode'] == vnode_id]
        if not records:
            return {}
        keys = [rec['key'] for rec in records]
        values = [rec['value'] for rec in records]
        embs = self.embedder.encode(values, convert_to_numpy=True)

        # 2. k=2 k-means
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(embs)

        # 3. Create sub-vnodes
        x1 = self._hash(f"split:{vnode_id}:1")
        x2 = self._hash(f"split:{vnode_id}:2")
        # Assign X1 to original server
        self.vnode_to_server[x1] = orig_server
        # Temporarily zero out orig weight for X2
        orig_w = self.weights[orig_server]
        self.weights[orig_server] = 0
        # HRW for X2
        best = None; best_score = float('-inf')
        for sid, w in self.weights.items():
            score = self._hash(f"{x2}:{sid}") * w
            if score > best_score:
                best_score, best = score, sid
        self.weights[orig_server] = orig_w
        self.vnode_to_server[x2] = best

        # 4. Update ring
        ring = self.vnode_rings[cluster_id]
        # remove original vnode
        ring.remove(vnode_id)
        # add sub-vnodes
        ring.extend([x1, x2])
        ring.sort()

        # 5. Migrate C2 keys
        migrated = {}
        new_store = []
        for rec, lbl in zip(records, labels):
            if lbl == 0:
                # cluster1 -> X1, stays
                rec['vnode'] = x1
                continue
            # cluster2 -> X2, move
            rec['vnode'] = x2
            migrated.setdefault(best, []).append(rec)
        # filter original server store
        self.server_store[orig_server] = [rec for rec in self.server_store[orig_server] if rec['vnode'] != x2]
        # add to new server
        self.server_store.setdefault(best, []).extend(migrated.get(best, []))

        return {'vnode': vnode_id, 'sub1': x1, 'sub2': x2, 'new_server': best}

'''
import pandas as pd
from AHC import ACHSystem
import argparse

def main():
    parser = argparse.ArgumentParser(description='Embed and store data using AHC system')
    parser.add_argument('--input', type=str, default='data.csv', help='Input CSV file path')
    parser.add_argument('--config', type=str, default='config.json', help='AHC system config file')
    parser.add_argument('--num_servers', type=int, default=3, help='Number of servers (if no config)')
    parser.add_argument('--total_capacity', type=int, default=1000, help='Total capacity (if no config)')
    parser.add_argument('--k_clusters', type=int, default=2, help='Number of clusters (if no config)')
    parser.add_argument('--replicas', type=int, default=3, help='Number of replicas (if no config)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to store')
    args = parser.parse_args()

    # Initialize AHC system
    try:
        # Try to load existing configuration
        print(f"Attempting to load existing AHC configuration from {args.config}")
        ach = ACHSystem(num_servers=1, total_capacity=1, k_clusters=1)  # Placeholder values
        ach.load_config(args.config)
        print("Successfully loaded existing configuration")
    except FileNotFoundError:
        # Create new system if config doesn't exist
        print(f"No existing configuration found. Creating new AHC system with {args.num_servers} servers")
        ach = ACHSystem(
            num_servers=args.num_servers, 
            total_capacity=args.total_capacity, 
            k_clusters=args.k_clusters,
            replicas=args.replicas
        )
        
        # We need to initialize the system by running k-means on some data
        # and building the virtual node rings
        print("Reading initial batch of data for clustering...")
        df = pd.read_csv(args.input)
        sample_data = df.head(min(1000, len(df)))
        
        # Extract text for embedding (country, device, title)
        # Format as simple string for embedding
        text_data = sample_data.apply(
            lambda row: f"{row['country']} {row['device']} {row['title']}", 
            axis=1
        ).tolist()
        
        # Run clustering to initialize the system
        print("Generating embeddings and clustering...")
        embeddings = ach.embed_keys(text_data)
        ach.run_kmeans(embeddings)
        
        # Build and assign virtual nodes
        print("Building virtual node rings...")
        ach.build_vnode_rings()
        ach.assign_vnodes_hrw()
        
        # Save the initial configuration
        ach.save_config(args.config)
        print(f"Saved initial configuration to {args.config}")

    # Process the CSV file in batches
    print(f"Processing data from {args.input}...")
    stored_count = 0
    
    # If max_records is set, limit the amount of data to process
    for chunk in pd.read_csv(args.input, chunksize=args.batch_size):
        for idx, row in chunk.iterrows():
            # Skip if we've reached the maximum number of records
            if args.max_records is not None and stored_count >= args.max_records:
                break
                
            # Get ID from the first column
            record_id = row['id']
            
            # Combine fields to create the text for embedding
            text_value = f"{row['country']} {row['device']} {row['title']}"
            
            # Store the data using the AHC system
            ach.store(record_id, text_value)
            stored_count += 1
            
            # Show progress
            if stored_count % 1000 == 0:
                print(f"Stored {stored_count} records")
                
        # Break the outer loop too if we've reached the limit
        if args.max_records is not None and stored_count >= args.max_records:
            print(f"Reached maximum number of records ({args.max_records})")
            break
    
    # Save the final state of the system
    ach.save_config(args.config)
    print(f"Completed storing {stored_count} records")
    print(f"Final configuration saved to {args.config}")

if __name__ == "__main__":
    main() 
'''

    
