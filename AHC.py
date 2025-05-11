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
        # Add the new server to the list of servers
        new_server_id = self._hash(str(len(self.servers)))
        self.servers.append(new_server_id)
        self.original_server_ids.append(len(self.original_server_ids))
        
        # Update the weights
        self.weights[new_server_id] = capacity
        
        # Initialize server storage for the new server
        self.server_store[new_server_id] = []

        # Store all new vnodes that will be added to each cluster
        new_vnodes_by_cluster = {cid: [] for cid in range(self.k_clusters)}
        
        # Create new virtual nodes for each cluster
        for cid in range(self.k_clusters):
            for i in range(self.replicas):
                new_vnode = self._hash(f"{cid}:{new_server_id}:{i}")
                new_vnodes_by_cluster[cid].append(new_vnode)
        
        # Track affected old vnodes and their information
        affected_vnodes = {}
        
        for cid, new_vnodes in new_vnodes_by_cluster.items():
            # Get a copy of the current ring before adding new vnodes
            current_ring = self.vnode_rings[cid].copy()
            
            for new_vnode in new_vnodes:
                # Find where the new vnode would be inserted
                idx = bisect.bisect_right(current_ring, new_vnode)
                
                # The affected vnode is the one that follows the position where new_vnode would be inserted
                if idx < len(current_ring):
                    affected_old_vnode = current_ring[idx]
                elif len(current_ring) > 0:  # If insertion point is at the end, the first vnode is affected
                    affected_old_vnode = current_ring[0]
                else:
                    continue  # No existing vnodes in this cluster
                
                # Get the server for the affected old vnode
                old_server = self.vnode_to_server.get(affected_old_vnode)
                
                if old_server is not None:
                    # Store the affected vnode information
                    affected_vnodes[affected_old_vnode] = {
                        'new_vnode': new_vnode,
                        'old_server': old_server,
                        'new_server': new_server_id,
                        'cluster': cid
                    }
                
                # Directly assign this new vnode to the new server (no HRW needed)
                self.vnode_to_server[new_vnode] = new_server_id
            
            # Now add the new vnodes to the ring
            for new_vnode in new_vnodes:
                self.vnode_rings[cid].append(new_vnode)
            
            # Sort the ring after adding all new vnodes
            self.vnode_rings[cid].sort()
        
        # Migrate data from affected old vnodes to new vnodes
        for old_vnode, info in affected_vnodes.items():
            old_server = info['old_server']
            new_vnode = info['new_vnode']
            new_server = info['new_server']
            cluster_id = info['cluster']
            
            # Find records on the old server that need to be moved
            records_to_move = []
            records_to_keep = []
            
            # Loop through all records in the old server
            for record in self.server_store[old_server]:
                # Check if this record is associated with the affected vnode
                if record['vnode'] == old_vnode:
                    # Get the key hash
                    key_hash = record['hash']
                    
                    # Check if this key should now hash to the new vnode
                    # by finding its position in the updated ring
                    vnode_ids = self.vnode_rings[cluster_id]
                    idx = bisect.bisect_right(vnode_ids, key_hash)
                    if idx == len(vnode_ids):
                        idx = 0
                    target_vnode = vnode_ids[idx]
                    
                    # If this key should now be stored on the new vnode
                    if target_vnode == new_vnode:
                        # Update the record's vnode reference
                        record['vnode'] = new_vnode
                        # Add to the list of records to move
                        records_to_move.append(record)
                    else:
                        # Keep it on the current server
                        records_to_keep.append(record)
                else:
                    # This record is not associated with the affected vnode
                    records_to_keep.append(record)
            
            # Move records to the new server
            self.server_store[new_server].extend(records_to_move)
            
            # Update the old server to only keep relevant records
            self.server_store[old_server] = records_to_keep
        
        return affected_vnodes

    def remove_server(self, server_id_to_remove):
        """
        Remove a server from the system and migrate its data to appropriate servers.
        
        Input:
            server_id_to_remove: The ID of the server to remove (original/unhashed ID)
            
        Output:
            Dictionary mapping removed vnodes to their new target vnodes
        """
        if len(self.servers) <= 1:
            raise ValueError("Cannot remove the last server from the system")
            
        # Convert original ID to hashed server ID if needed
        if isinstance(server_id_to_remove, int) and server_id_to_remove in self.original_server_ids:
            idx = self.original_server_ids.index(server_id_to_remove)
            hashed_server_id = self.servers[idx]
        else:
            hashed_server_id = server_id_to_remove
            
        if hashed_server_id not in self.servers:
            raise ValueError(f"Server {server_id_to_remove} not found in the system")
            
        # Find all vnodes that belong to this server
        removed_vnodes = {}
        for cid in range(self.k_clusters):
            removed_vnodes[cid] = []
            for vnode_id in self.vnode_rings[cid]:
                if self.vnode_to_server.get(vnode_id) == hashed_server_id:
                    removed_vnodes[cid].append(vnode_id)
        
        # Create a copy of the current state before we modify anything
        current_vnode_rings = {cid: self.vnode_rings[cid].copy() for cid in range(self.k_clusters)}
        
        # Track which records need to be migrated to which servers
        records_to_migrate = {}  # target_server -> [records]
        
        # Process all records on the server to be removed
        for record in self.server_store[hashed_server_id]:
            cluster_id = record['cluster']
            key_hash = record['hash']
            
            # Create a copy of the vnode ring without the server's vnodes
            new_ring = [v for v in current_vnode_rings[cluster_id] 
                        if self.vnode_to_server.get(v) != hashed_server_id]
            
            if not new_ring:
                # If there are no other vnodes in this cluster, we can't migrate
                # This is an error state that should be prevented
                raise ValueError(f"No target servers available in cluster {cluster_id}")
            
            # Find where this key would hash in the new ring
            idx = bisect.bisect_right(new_ring, key_hash)
            if idx == len(new_ring):
                idx = 0
                
            # Get the target vnode and server
            target_vnode = new_ring[idx]
            target_server = self.vnode_to_server[target_vnode]
            
            # Update the record's vnode
            record['vnode'] = target_vnode
            
            # Add to migration list
            if target_server not in records_to_migrate:
                records_to_migrate[target_server] = []
            records_to_migrate[target_server].append(record)
        
        # Perform the actual migration
        for target_server, records in records_to_migrate.items():
            self.server_store[target_server].extend(records)
            
        # Remove all vnodes belonging to this server from the rings
        for cid in range(self.k_clusters):
            self.vnode_rings[cid] = [v for v in self.vnode_rings[cid] 
                                    if self.vnode_to_server.get(v) != hashed_server_id]
            
        # Remove the server's vnodes from vnode_to_server mapping
        self.vnode_to_server = {k: v for k, v in self.vnode_to_server.items() 
                               if v != hashed_server_id}
            
        # Update servers list and original_server_ids
        idx = self.servers.index(hashed_server_id)
        self.servers.pop(idx)
        self.original_server_ids.pop(idx)
        
 
        # Clear the server store for the removed server
        self.server_store.pop(hashed_server_id, None)
        
        return records_to_migrate
    
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


    
