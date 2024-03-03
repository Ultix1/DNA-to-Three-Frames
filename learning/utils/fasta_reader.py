
from Bio import SeqIO

def read_fasta(filename: str):
    counter = 0
    with open(filename, "r") as file:
        for line in file:
            if line.startswith('>'):
                # This line is a header, indicating the start of a new sequence
                header = line.strip()
                sequence = ""
            else:
                # This line contains part of the sequence
                sequence += line.strip()
            # Process the sequence here, if necessary
    
        print(len(sequence))

with open("GRCh38_dna.fna", "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        print('ID:', record.id)
        print('Description:', record.description)
        print('Sequence Length:', len(record))
        print()



# read_fasta("Genome_1.fna")