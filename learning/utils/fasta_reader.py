
from Bio import SeqIO
from utils.constants import CODON_TABLE

def read_fasta(filename: str, max_size:int = 0, protein_len:int = 100):
    test_sequences = {}
    with open(filename, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if len(record) == protein_len:
                test_sequences[record.id] = record.seq
                if len(test_sequences)  == max_size:
                    break
            
            # print('ID:', record.id)
            # print('Description:', record.description)
            # print('Sequence Length:', len(record))

    return test_sequences

def convert_to_dna(protein: str):
    dna = ""
    for char in protein:
        codon = "*"
        for key, value in CODON_TABLE.items():
            if(value.lower == char.lower):
                codon = key
                break
        
        dna = dna + codon

    return dna