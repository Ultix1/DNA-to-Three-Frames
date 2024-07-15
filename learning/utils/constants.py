from enum import Enum

class Action(Enum):
    MATCH = 1
    FRAMESHIFT_1 = 2
    FRAMESHIFT_3 = 3
    MISMATCH = 4
    DELETE = -1
    INSERT = -2
    INDEL = 4
    NONE = 0

PROTEINS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', "*"]

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

NEG_INF = -999

GAP_EXTENSION_PENALTY = 2

GAP_OPEN_PENALTY = 3

FRAMESHIFT_PENALTY = 4

