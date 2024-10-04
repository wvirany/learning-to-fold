import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, fcluster

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment


def cluster_msa(msa, threshold=0.8):

    # Convert to a distance matrix
    n_sequences = len(msa)
    distances = np.zeros((n_sequences, n_sequences))

    """ Compute pairwise Hamming distance between strings: Count number of positions in which two sequences differ, then divide by total length of sequences """
    for i in range(n_sequences):
        for j in range(i+1, n_sequences):
            distances[i,j] = distances[j,i] = sum(a != b for a,b in zip(msa[i], msa[j])) / n_sequences

    # Compute linkage matrix, which performs hierarchical clustering (see scipy documentation)
    # np.triu_indices returns the indices of the elements of distances in the upper triangle as 
    # a vector, then these elements are passed to the linkage function which performs hierarchical
    # clustering
    linkage_matrix = linkage(distances[np.triu_indices(n_sequences, k=1)], method='average')
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # Get cluster centers
    cluster_centers = []
    for cluster_id in set(clusters):
        cluster_members = [seq for seq, cid in zip(msa, clusters) if cid == cluster_id]
        center = min(cluster_members, key=lambda x: sum(y != x for y in cluster_members))
        cluster_centers.append(center)
    
    return clusters, cluster_centers


def visualize_msa_clusters(msa, clusters):

    # Create a color map
    unique_clusters = np.unique(clusters)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    cluster_colors = {c: colors[i] for i, c in enumerate(unique_clusters)}

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    for i, seq in enumerate(msa):
        cluster = clusters[i]
        color = cluster_colors[cluster]
        
        for j, aa in enumerate(seq):
            if aa != '-':
                rect = plt.Rectangle((j, -i-1), 1, 1, facecolor=color, edgecolor='none', alpha=0.5)
                ax.add_patch(rect)
        
    ax.set_ylim(-len(msa), 0)
    ax.set_xlim(0, len(msa[0]))
    ax.set_yticks([-(i+0.5) for i in range(0, len(msa), 2)])
    ax.set_yticklabels([f'Sequence {i+1}' for i in range(0, len(msa), 2)])
    ax.set_xlabel('Sequence Position')
    ax.set_title('Multiple Sequence Alignment with Cluster Coloring')

    # Create legend
    legend_elements = [Patch(facecolor=cluster_colors[c], edgecolor='none', alpha=0.5, label=f'Cluster {c}') 
                       for c in unique_clusters]
    ax.legend(handles=legend_elements, title='Clusters', loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()

# alignment = AlignIO.read('synthetic_msa.fasta', 'fasta')
# msa = np.array([str(record.seq) for record in alignment])

# clusters, cluster_centers = cluster_msa(msa, threshold=.44)

# visualize_msa_clusters(msa, clusters)