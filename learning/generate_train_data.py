from models_v2.environment import renderSeq
from utils.fasta_reader import read_fasta, convert_to_dna
from utils.sequence_gen import SeqGen
from utils.constants import CODON_TABLE, PROTEINS
import random
import os
import re
import numpy as np

DATA_DIR = "data"
FASTA_DIR = "fasta_tests/fasta_files/fruit_fly.fasta"

def generate_rand_dna(length):
    dna = ""
    while len(dna) < length:
        dna += random.choice(list(CODON_TABLE.keys()))
    return dna[:length]

def generate_rand_protein(length):
    protein = ""
    while len(protein) < length:
        protein += random.choice(PROTEINS)
    return protein[:length]

def generate_data(size, range):
    
    # GET a Max of N Proteins from FASTA File, whose lengths range from 100-200 Characters
    proteins = read_fasta(FASTA_DIR, max_size=size, protein_len_range=range)

    # Get last number in list
    start_key = int(re.findall(r'\d+', os.listdir(f"{DATA_DIR}/dna")[-1])[0]) + 1
    
    for _, value in proteins.items():

        rand_dna = generate_rand_dna(len(value) * 3)
        rand_protein = generate_rand_protein(len(value))

        protein = value
        matching_dna = convert_to_dna(value)

        rand_dna_filename = f"{DATA_DIR}/dna/DNA{start_key}_random.txt"
        rand_protein_filename = f"{DATA_DIR}/proteins/AA{start_key}_random.txt"

        trans_dna_filename = f"{DATA_DIR}/dna/DNA{start_key}_match.txt"
        protein_filename = f"{DATA_DIR}/proteins/AA{start_key}_match.txt"
        
        # Save Protein to File
        file = open(f"{protein_filename}", 'a')
        file.write(f"{protein.strip()}")
        file.close()

        # Save Randomly Generated Protein
        file = open(f"{rand_protein_filename}", 'a')
        file.write(f"{protein.strip()}")
        file.close()

        # Save Directly Translated DNA
        file = open(f"{trans_dna_filename}", 'a')
        file.write(f"{matching_dna.strip()}")
        file.close()

        # Save Randomly Generated DNA to File
        file = open(f"{rand_dna_filename}", 'a')
        file.write(f"{rand_dna.strip()}")
        file.close()

        start_key += 1

# if __name__ == '__main__':
#     generate_data(size=100, range=(100,200))

# if __name__ == "__main__":
#     x = input("Window Size: ")
#     get_state_test(
#         dna="0000AGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGT",
#         protein="*SAINTSSAINTSSAINTS",
#         window_size=int(x),
#         dna_pointer=4,
#         protein_pointer=1
#     )