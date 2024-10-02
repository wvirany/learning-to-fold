import torch
import torch.nn as nn
import numpy as np

# Define mapping between amino acids and integers
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
idx_to_aa = {i: aa for i, aa in enumerate(amino_acids)}


# Exercise 1: Basic MSA Representation

def create_msa(sequences):
    # Convert sequences to numpy array
    msa = np.array([list(seq) for seq in sequences])
    return msa


sequences = [
    "MKILVALVF",
    "MRILVALIF",
    "MKILVALIS",
    "MKVLVALVF"
]


msa = create_msa(sequences)
print(msa)
print(f"Number of sequences: {msa.shape[0]}, Length of sequences: {msa.shape[1]}\n")


# Exercise 2: Calculate Sequence Conservation

def calculate_conservation(msa):
    """ Calculate conservation scores:
    
    Conservation scores is the number of most frequently occuring amino acids in a certain position
    divided by the total number of sequences. This gives the proportion of how many times the amino
    acid in this position is preserved across all sequences.
    """

    num_sequences, seq_length = msa.shape
    conservation = np.zeros(seq_length)

    for i in range(seq_length):
        _, counts = np.unique(msa[:, i], return_counts=True)
        conservation[i] = np.max(counts) / num_sequences
    
    return conservation

conservation = calculate_conservation(msa)
print(f"Conservation scores: {conservation}\n")


# Exercise 3: Identify Coevolving Pairs

def identify_coevolving_pairs(msa, threshold=0.5):
    """ Identify Coevolving Pairs
    
    We iterate through the columns of the MSA. To check for coevolution between two amino acids, we
    check how many times the amino acids in that position are equal, and then compute the proportion
    of amino acids which are equal in that position across all sequences. If more than threshold-%
    are equal, we say they may be coevolving.
    """

    num_sequences, seq_length = msa.shape
    coevolving_pairs = []

    for i in range(seq_length):
        for j in range(i+1, seq_length):
            matching = np.sum(msa[:, i] == msa[:, j]) / num_sequences
            if matching >= threshold:
                coevolving_pairs.append((i, j))
    
    return coevolving_pairs

coevolving_pairs = identify_coevolving_pairs(msa)
print(f"Potentially coevolving pairs: {coevolving_pairs}\n")


# Exercise 4: MSA Embedding

def one_hot_encode(msa):
    """ Returns one-hot encoding of MSA """

    num_sequences, seq_length = msa.shape
    encoding = torch.zeros((num_sequences, seq_length, len(amino_acids)))

    for i in range(num_sequences):
        for j in range(seq_length):
            aa = msa[i, j]
            idx = aa_to_idx[aa]

            encoding[i, j, idx] = 1

    return encoding

msa_embedding = one_hot_encode(msa)

# 4, 9, 20 - 4 sequences, 9 amino acids each. For each amino acid, we have a 20-dimensional one-hot vector
print(f"MSA embedding shape: {msa_embedding.shape}\n")


# Exercise 5: Simulate AlphaFold's MSA Processing

class SimpleMSAProcessor(nn.Module):
    """ A simple neural network module to process MSA's """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # Feedforward
        ff_output = self.feedforward(x)
        x = self.layer_norm2(x + ff_output)

        return x

# Initialize and apply the MSA processor
model = SimpleMSAProcessor(input_dim= 20, hidden_dim=64)

processed_msa = model(msa_embedding)
print(f"Model output shape: {processed_msa.shape}")