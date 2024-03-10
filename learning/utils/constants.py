from enum import Enum

class Action(Enum):
    MATCH = 0
    FRAMESHIFT_1 = 1
    FRAMESHIFT_3 = 2
    DELETE = 3
    INSERT = 4
    MISMATCH = 5
    NONE = -999

CODON_TABLE = {
    "TTT" : "F", "TTC" : "F", "TTA" : "L", "TTG" : "L",
    "CTT" : "L", "CTC" : "L", "CTA" : "L", "CTG" : "L",
    "ATT" : "I", "ATC" : "I", "ATA" : "I", "ATG" : "M",
    "GTT" : "V", "GTC" : "V", "GTA" : "V", "GTG" : "V",
    "TCT" : "S", "TCC" : "S", "TCA" : "S", "TCG" : "S",
    "CCT" : "P", "CCC" : "P", "CCA" : "P", "CCG" : "P",
    "ACT" : "T", "ACC" : "T", "ACA" : "T", "ACG" : "T",
    "GCT" : "A", "GCC" : "A", "GCA" : "A", "GCG" : "A",
    "TAT" : "Y", "TAC" : "Y", "TAA" : "*", "TAG" : "*",
    "CAT" : "H", "CAC" : "H", "CAA" : "Q", "CAG" : "Q",
    "AAT" : "N", "AAC" : "N", "AAA" : "K", "AAG" : "K",
    "GAT" : "D", "GAC" : "D", "GAA" : "E", "GAG" : "E",
    "TGT" : "C", "TGC" : "C", "TGA" : "*", "TGG" : "W",
    "CGT" : "R", "CGC" : "R", "CGA" : "R", "CGG" : "R",
    "AGT" : "S", "AGC" : "S", "AGA" : "R", "AGG" : "R",
    "GGT" : "G", "GGC" : "G", "GGA" : "G", "GGG" : "G",
}

PARAMS = {
    'epsilon' : 0.99999,    # Starting Epsilon
    'epsilon_min': 0.01,    # Minimum Epsilon
    'decay' : 0.99,         # Epsilon Decay
    'gamma' : 0.99,         # Discount Factor for target q vals
    'buffer_size' : 50000,  # Size of Buffer
    'max_ep' : 1000,        # Max training episodes
    'batch_size' : 64,      # Batch size for training
    'train_freq' : 100,     # How Many Steps before Updating Main Q-Network
    'tau': 0.01             # Discount Factor for Updating Target Q-Network
}

GAP_EXTENSION_PENALTY = 2

GAP_OPEN_PENALTY = 3

FRAMESHIFT_PENALTY = 4

