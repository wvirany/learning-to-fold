import numpy as np
from collections import Counter

def prepare_cluster_features(msa, clusters):

    cluster_features = {}
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    for cluster_id in set(clusters):
        cluster_seqs = [seq for seq, cid in zip(msa, clusters) if cid == cluster_id]

        # Cluster center sequence
        center_seq = cluster_seqs[0] # For simplicity, assume first sequence is center

        # Deletion mask and values
        deletion_mask = np.array([[aa == '-' for aa in seq] for seq in cluster_seqs])
        deletion_mean = np.mean(deletion_mask, axis=0)