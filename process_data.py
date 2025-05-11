import pandas as pd
import numpy as np
import argparse
import sys
from AHC import ACHSystem

def main():
    parser = argparse.ArgumentParser(description='Process data with ACH system: 30% for training, 70% for storage')
    parser.add_argument('--input', type=str, default='data.csv', help='Input CSV file path')
    parser.add_argument('--config', type=str, default='config.json', help='ACH system config file')
    parser.add_argument('--num_servers', type=int, default=3, help='Number of servers')
    parser.add_argument('--total_capacity', type=int, default=100, help='Total capacity')
    parser.add_argument('--k_clusters', type=int, default=10, help='Number of clusters')
    parser.add_argument('--replicas', type=int, default=5, help='Number of replicas')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to store')
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Load the dataset
    print(f"Loading dataset from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Split the dataset: 30% for training, 70% for storage
    train_indices = np.random.choice(
        df.index, 
        size=int(len(df) * 0.3), 
        replace=False
    )
    train_df = df.loc[train_indices]
    store_df = df.drop(train_indices)
    
    # Apply max_records limit to storage data if specified
    if args.max_records is not None:
        # Ensure we have enough data for training
        total_desired = int(args.max_records / 0.7)  # Calculate total needed based on 70% storage ratio
        train_count = int(total_desired * 0.3)
        store_count = args.max_records
        
        # Adjust training data if needed
        if len(train_df) > train_count:
            train_df = train_df.sample(train_count)
        
        # Limit storage data
        if len(store_df) > store_count:
            store_df = store_df.sample(store_count)
            
        print(f"Limited dataset: {len(train_df)} records for training, {len(store_df)} records for storage (max_records={args.max_records})")
    else:
        print(f"Split dataset: {len(train_df)} records for training, {len(store_df)} records for storage")
    
    # Initialize the ACH system
    print("Initializing ACH system...")
    ach = ACHSystem(
        num_servers=args.num_servers,
        total_capacity=args.total_capacity,
        k_clusters=args.k_clusters,
        replicas=args.replicas
    )
    
    # Prepare training data for embedding
    print("Preparing training data for embedding...")
    train_texts = train_df.apply(
        lambda row: f"{row['country']} {row['device']} {row['title']}", 
        axis=1
    ).tolist()
    
    # Generate embeddings for training data
    print("Generating embeddings for training data...")
    train_embeddings = ach.embed_keys(train_texts)
    
    # Run k-means clustering
    print(f"Running k-means clustering with k={args.k_clusters}...")
    ach.run_kmeans(train_embeddings)
    
    # Build and assign virtual nodes
    print("Building virtual node rings...")
    ach.build_vnode_rings()
    print("Assigning virtual nodes to servers...")
    ach.assign_vnodes_hrw()
    
    # Save the initial configuration
    print(f"Saving initial configuration to {args.config}...")
    ach.save_config(args.config)
    
    # Store the remaining 70% of data
    print("Storing the remaining data...")
    stored_count = 0
    
    # Process in batches to avoid memory issues
    for start_idx in range(0, len(store_df), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(store_df))
        batch = store_df.iloc[start_idx:end_idx]
        
        for idx, row in batch.iterrows():
            # Extract ID
            record_id = row['id']
            
            # Combine fields for value
            text_value = f"{row['country']} {row['device']} {row['title']}"
            
            # Store in ACH system
            ach.store(record_id, text_value)
            stored_count += 1
            
            # Show progress
            if stored_count % 1000 == 0:
                print(f"Stored {stored_count}/{len(store_df)} records")
    
    # Save the final configuration
    print(f"Saving final configuration to {args.config}...")
    ach.save_config(args.config)
    
    print(f"Completed! Used {len(train_df)} records for training and stored {stored_count} records")
    print(f"Configuration saved to {args.config}")
    
    # Basic stats
    print("\nBasic statistics:")
    server_counts = {sid: len(records) for sid, records in ach.server_store.items()}
    for sid, count in server_counts.items():
        original_id = ach.original_server_ids[ach.servers.index(sid)]
        print(f"Server {original_id}: {count} records ({count/stored_count*100:.1f}%)")

if __name__ == "__main__":
    main() 