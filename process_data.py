#!/usr/bin/env python3
"""
Enhanced benchmark script comparing AHC vs traditional consistent hashing (CH)
with server capacity, extended metrics, charts, locality measurements,
controlled sampling, and inter-/intra-cluster similarity.
"""
import sys, os, random, argparse, pandas as pd, numpy as np, time, hashlib, bisect, gc
import matplotlib.pyplot as plt
from AHC import ACHSystem


def run_ach(cluster_keys, test_keys, test_values,
            num_servers, total_capacity, k_clusters, replicas,
            model_name, config_path=None):
    ach = ACHSystem(
        num_servers=num_servers,
        total_capacity=total_capacity,
        k_clusters=k_clusters,
        replicas=replicas,
        model_name=model_name
    )
    # Optional config load
    if config_path and os.path.exists(config_path):
        ach.load_config(config_path)
        offline_times = {'embed_cluster':0.0,'kmeans':0.0,'ring_build':0.0}
    else:
        t0=time.perf_counter(); key_embs=ach.embed_keys(cluster_keys); t1=time.perf_counter()
        t2=time.perf_counter(); ach.run_kmeans(key_embs); t3=time.perf_counter()
        t4=time.perf_counter(); ach.build_vnode_rings(); ach.assign_vnodes_hrw(); t5=time.perf_counter()
        offline_times={'embed_cluster':t1-t0,'kmeans':t3-t2,'ring_build':t5-t4}
        if config_path: ach.save_config(config_path)
    # Precompute embeddings
    key_test_embs=ach.embed_keys(test_keys)
    t0=time.perf_counter(); val_embs=ach.embed_keys(test_values); t1=time.perf_counter()
    offline_times['embed_test']=t1-t0
    # AHC assign & latencies
    ach_assign=[]; ach_lat=[]; server_counts_ach={sid:0 for sid in ach.servers}
    gc.disable()
    for key,val_emb in zip(test_keys,val_embs):
        t0=time.perf_counter()
        cid=np.argmin(np.linalg.norm(ach.centroids-val_emb,axis=1))
        vnode_ids=ach.vnode_rings[cid]
        kh=ach._hash(key); idx=bisect.bisect_right(vnode_ids,kh)
        if idx==len(vnode_ids): idx=0
        srv=ach.vnode_to_server[vnode_ids[idx]]
        server_counts_ach[srv]+=1
        ach_assign.append(srv)
        ach_lat.append(time.perf_counter()-t0)
    gc.enable()
    # CH assign & latencies
    ch_assign=[]; ch_lat=[]; server_counts_ch={sid:0 for sid in range(num_servers)}
    ring=[(int(hashlib.sha256(f"{sid}:{i}".encode()).hexdigest(),16)%(2**64),sid)
          for sid in range(num_servers) for i in range(replicas)]
    ring.sort(key=lambda x:x[0])
    gc.disable()
    for key in test_keys:
        t0=time.perf_counter()
        kh=int(hashlib.sha256(str(key).encode()).hexdigest(),16)%(2**64)
        idx=bisect.bisect_right(ring,(kh,None))
        if idx==len(ring): idx=0
        srv=ring[idx][1]; server_counts_ch[srv]+=1
        ch_assign.append(srv); ch_lat.append(time.perf_counter()-t0)
    gc.enable()
    # metrics arrays
    ach_times=np.array(ach_lat); ch_times=np.array(ch_lat)
    counts_ach=np.array(list(server_counts_ach.values())); counts_ch=np.array(list(server_counts_ch.values()))
    # capacities
    caps=np.array([ach.weights[sid] for sid in ach.servers])
    # locality (top-5%)
    ne=key_test_embs/np.linalg.norm(key_test_embs,axis=1,keepdims=True)
    n=len(ne); num_pairs=min(50000,n*(n-1)//2)
    pairs=random.sample([(i,j) for i in range(n) for j in range(i)],num_pairs)
    sims=np.array([ne[i].dot(ne[j]) for i,j in pairs])
    thr=np.percentile(sims,95)
    top=[pairs[k] for k,v in enumerate(sims) if v>=thr]
    ach_loc=np.mean([ach_assign[i]==ach_assign[j] for i,j in top])
    ch_loc=np.mean([ch_assign[i]==ch_assign[j] for i,j in top])
    # inter/intra similarity
    cluster_ids=[np.argmin(np.linalg.norm(ach.centroids-e,axis=1)) for e in key_test_embs]
    intra=[]; inter=[]
    M=min(10000,n*(n-1)//2)
    while len(intra)<M or len(inter)<M:
        i,j=random.randrange(n),random.randrange(n)
        if i==j: continue
        sim=ne[i].dot(ne[j])
        if cluster_ids[i]==cluster_ids[j] and len(intra)<M: intra.append(sim)
        if cluster_ids[i]!=cluster_ids[j] and len(inter)<M: inter.append(sim)
        if len(intra)>=M and len(inter)>=M: break
    sim_stats={'intra_mean':np.mean(intra),'intra_std':np.std(intra),
               'inter_mean':np.mean(inter),'inter_std':np.std(inter)}
        # Compute cluster assignments for test keys
    cluster_ids = np.argmin(
        np.linalg.norm(key_test_embs[:, None, :] - ach.centroids[None, :, :], axis=2),
        axis=1
    )
    # Compare intra- vs inter-cluster similarity
    same, diff = [], []
    max_pairs = min(len(pairs), 5000)
    for (i,j) in pairs:
        if len(same) < max_pairs and cluster_ids[i] == cluster_ids[j]:
            same.append((i,j))
        if len(diff) < max_pairs and cluster_ids[i] != cluster_ids[j]:
            diff.append((i,j))
        if len(same) >= max_pairs and len(diff) >= max_pairs:
            break
    sims_same = [ne[i].dot(ne[j]) for i,j in same]
    sims_diff = [ne[i].dot(ne[j]) for i,j in diff]
    intra_mean, intra_std = float(np.mean(sims_same)), float(np.std(sims_same))
    inter_mean, inter_std = float(np.mean(sims_diff)), float(np.std(sims_diff))
    # Server capacities (AHC weights)
    cap_ach = np.array([ach.weights[sid] for sid in ach.servers])

    return {
        'offline': offline_times,
        'capacity': {'AHC': cap_ach.tolist()},
        'offline': offline_times,
        'servers': ach.servers,
        'capacities': caps,
        'offline':offline_times,
        'throughput':{'AHC':len(test_keys)/ach_times.sum(),'CH':len(test_keys)/ch_times.sum()},
        'latency':{'AHC':np.percentile(ach_times,[50,90,99])*1000,'CH':np.percentile(ch_times,[50,90,99])*1000},
        'counts':{'AHC':counts_ach,'CH':counts_ch},
        'imbalance':{'AHC':counts_ach.max()/counts_ach.min(),'CH':counts_ch.max()/counts_ch.min()},
        'locality':{'AHC':ach_loc,'CH':ch_loc},
        'similarity':sim_stats
    }

def plot_and_save(metrics, num_servers):
    # Latency
    plt.figure(figsize=(6,4))
    plt.hist(metrics['latency']['AHC'], bins=50, alpha=0.6, label='AHC')
    plt.hist(metrics['latency']['CH'], bins=50, alpha=0.6, label='CH')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('latency_distribution.png')

    # Load distribution with capacity annotation
    plt.figure(figsize=(8,4))
    x = np.arange(num_servers)
    w = 0.35
    ach_counts = metrics['counts']['AHC']
    ch_counts = metrics['counts']['CH']
    caps = metrics['capacity']['AHC']
    plt.bar(x-w/2, ach_counts, w, label='AHC')
    plt.bar(x+w/2, ch_counts, w, label='CH')
    labels = [f'S{i}(cap={caps[i]})' for i in x]
    plt.xticks(x, labels)
    plt.xlabel('Server (capacity)')
    plt.ylabel('Assigned Records')
    plt.legend()
    plt.tight_layout()
    plt.savefig('load_distribution.png')

    # Locality
    plt.figure(figsize=(4,4))
    vals = [metrics['locality']['AHC'], metrics['locality']['CH']]
    plt.bar(['AHC','CH'], vals)
    plt.ylabel('Co-location Ratio')
    plt.tight_layout()
    plt.savefig('locality_distribution.png')

    # Similarity intra vs inter-cluster
    sim = metrics['similarity']
    plt.figure(figsize=(4,4))
    plt.bar(['Intra','Inter'], [sim['intra_mean'], sim['inter_mean']], yerr=[sim['intra_std'], sim['inter_std']])
    plt.ylabel('Cosine Similarity')
    plt.title('Intra vs Inter-cluster Similarity')
    plt.tight_layout()
    plt.savefig('similarity_distribution.png')


def print_metrics(m):
    srv=m['servers']; cap=m['capacities']
    print("Server capacities vs. assigned counts:")
    for i,sid in enumerate(srv):
        print(f"  Server {sid}: capacity={cap[i]}, AHC={m['counts']['AHC'][i]}, CH={m['counts']['CH'][i]}")
    off=m['offline']; print(f"Offline times: embed_cluster={off['embed_cluster']:.2f}s, kmeans={off['kmeans']:.2f}s, ring_build={off['ring_build']:.2f}s, embed_test={off['embed_test']:.2f}s")
    th=m['throughput']; print(f"Throughput recs/s: AHC={th['AHC']:.2f}, CH={th['CH']:.2f}")
    la=m['latency']; print(f"Latency p50/p90/p99: AHC={la['AHC'][0]:.2f}/{la['AHC'][1]:.2f}/{la['AHC'][2]:.2f} ms, CH={la['CH'][0]:.2f}/{la['CH'][1]:.2f}/{la['CH'][2]:.2f} ms")
    im=m['imbalance']; print(f"Imbalance max/min: AHC={im['AHC']:.2f}, CH={im['CH']:.2f}")
    loc=m['locality']; print(f"Locality top-5% pairs co-located: AHC={loc['AHC']:.3f}, CH={loc['CH']:.3f}")
    sim=m['similarity']; print(f"Similarity intra-cluster mean/std: {sim['intra_mean']:.3f}/{sim['intra_std']:.3f}")
    print(f"Similarity inter-cluster mean/std:   {sim['inter_mean']:.3f}/{sim['inter_std']:.3f}")


def main():
    parser=argparse.ArgumentParser(description="AHC vs CH benchmarking with capacity & similarity")
    parser.add_argument('--input',required=True)
    parser.add_argument('--num_servers',type=int,default=3)
    parser.add_argument('--total_capacity',type=int,default=100)
    parser.add_argument('--k_clusters',type=int,default=10)
    parser.add_argument('--replicas',type=int,default=100)
    parser.add_argument('--model_name',type=str,default='all-MiniLM-L6-v2')
    parser.add_argument('--cluster_frac',type=float,default=0.3)
    parser.add_argument('--test_n',type=int,default=20000)
    parser.add_argument('--config',type=str,default=None)
    args=parser.parse_args()

    df=pd.read_csv(args.input)
    cluster_df=df.sample(frac=args.cluster_frac,random_state=42)
    cluster_df = df.sample(n=20000, random_state=42)
    remain=df.drop(cluster_df.index)
    test_df=remain.sample(n=min(args.test_n,len(remain)),random_state=42)
    cluster_keys=cluster_df['key'].tolist()
    test_keys=test_df['key'].tolist()
    test_values=test_df['values'].tolist()

    metrics=run_ach(cluster_keys,test_keys,test_values,
                    args.num_servers,args.total_capacity,
                    args.k_clusters,args.replicas,
                    args.model_name,args.config)
    print_metrics(metrics)
    plot_and_save(metrics,args.num_servers)
    print("Charts: latency_distribution.png, load_distribution.png, locality_distribution.png, similarity_distribution.png")

if __name__=='__main__': main()
