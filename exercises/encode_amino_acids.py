import matplotlib.pyplot as plt

"""
Exercise 1: Amino Acid Encoding

Demonstrates how to encode amino acid sequences into a format suitable for
machine learning models via one-hot encoding vectors.
"""

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def one_hot_encode_amino_acid(amino_acid):
    """
    Performs one-hot encoding for a single amino acid
    """

    encoding = [0] * 20 # Create 20-dimensional vector of 0's

    if amino_acid in amino_acids:
        encoding[amino_acids.index(amino_acid)] = 1 # Set bit at corresponding amino acid index to 1

    return encoding


def encode_sequence(sequence):
    """
    Encode a protein sequence using one-hot encoding
    """
    return [one_hot_encode_amino_acid(a) for a in sequence]


def make_plot(encoded_sequence):
    fig, ax = plt.subplots()
    im = ax.imshow(encoded_sequence, cmap='Blues')

    ax.set_title("One-hot Encoding of Amino Acids")

    ax.set_xticks(range(len(amino_acids)))
    ax.set_xticklabels(amino_acids)
    ax.set_xlabel("Amino Acids")
    ax.set_ylabel("Position in Sequence")

    plt.tight_layout()
    plt.show()


test_sequence = "ACDK"
encoded_sequence = encode_sequence(test_sequence)
make_plot(encoded_sequence)