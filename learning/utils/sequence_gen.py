import numpy as np
import random
from Bio.Seq import Seq

class SeqGen:
    
    def __init__(self, lseqs, num_sets):
        self.lseqs = lseqs
        self.num_sets = num_sets
        self.sequences = [] 
        self.mutated_sequences = []
        self.proteins = []
        self.BP = ['A', 'G', 'C', 'T']
        self.maxIndel = 2
        self.p_snp = 0.1
        self.p_indel = 0.05
        self.zif_s = 1.6

    def zipfian(self, s, N):
        temp0 = np.array(range(1, N + 1))
        temp0 = np.sum(1 / temp0**s)
        temp = random.random() * temp0
        for i in range(N):
            temp2 = 1 / ((i + 1) ** s)
            if temp < temp2:
                return i + 1
            else:
                temp -= temp2
        return 0

    def generate_sequence(self):
        seq = np.random.randint(4, size=self.lseqs)
        return seq

    def mutate_sequence(self, seq):
        mutated_seq = np.mod(seq + (np.random.rand(self.lseqs) < self.p_snp) * np.random.randint(1, 4, size=self.lseqs), 4)
        return mutated_seq

    def introduce_indels(self, seq):
        for i in range(len(seq)):
            if np.random.rand() < self.p_indel:
                indel_length = self.zipfian(self.zif_s, self.maxIndel)
                if np.random.rand() < 0.5:
                    seq = np.insert(seq, i, np.random.randint(4, size=indel_length))
                else:
                    del_start = max(i - indel_length, 0)
                    seq = np.delete(seq, slice(del_start, i))
        return seq

    def dna_to_protein(self, dna_seq):
        dna_seq_str = ''.join([self.BP[nuc] for nuc in dna_seq])
        dna_seq_obj = Seq(dna_seq_str)
        protein_seq = str(dna_seq_obj.translate())
        return protein_seq

    def generate_sequences_and_proteins(self, mutated=True):
        for _ in range(self.num_sets):
            original_seq = self.generate_sequence()
            mutated_seq = self.mutate_sequence(original_seq.copy())
            
            original_seq_with_indels = self.introduce_indels(original_seq)
            mutated_seq_with_indels = self.introduce_indels(mutated_seq)
            
            protein_seq = self.dna_to_protein(mutated_seq_with_indels if mutated else original_seq)
            
            self.sequences.append(original_seq_with_indels if mutated else original_seq)
            self.mutated_sequences.append(mutated_seq_with_indels)
            self.proteins.append(protein_seq)

    def save_sequences_to_files(self):
        for i, seq in enumerate(self.sequences, start=1):
            filename = f"DNA{i}.txt"
            with open(filename, 'w') as f:
                f.write(''.join([self.BP[nuc] for nuc in seq]))
                
        for i, protein in enumerate(self.proteins, start=1):
            filename = f"AA{i}.txt"
            with open(filename, 'w') as f:
                f.write(protein)
    
    def save_sequences_to_fasta(self):
        for i, seq in enumerate(self.sequences, start=1):
            filename = f"DNA{i}.fasta"
            with open(filename, 'w') as f:
                f.write(f">Test DNA Sequence length={self.lseqs}\n{''.join([self.BP[nuc] for nuc in seq])}")
                
        for i, protein in enumerate(self.proteins, start=1):
            filename = f"AA{i}.fasta"
            with open(filename, 'w') as f:
                f.write(f">Test Protein Sequence length={self.lseqs}\n{protein}")

    def save_sequences_to_fastq(self):
        seq_id = "@FRAMERL-TEST-DATA;mate1:1081-1180;mate2:1268-1367"
        for i, seq in enumerate(self.sequences, start=1):
            filename = f"DNA{i}.fastq"
            with open(filename, 'w') as f:
                f.write(f"{seq_id}\n{''.join([self.BP[nuc] for nuc in seq])}\n+\n{''.join('?' for _ in seq)}")
                
        for i, protein in enumerate(self.proteins, start=1):
            filename = f"AA{i}.fastq"
            with open(filename, 'w') as f:
                f.write(f"{seq_id}\n{protein}\n+\n{''.join('?' for _ in protein)}")


if __name__ == '__main__':
    lseqs = int(input("Enter the length of DNA string: "))
    num_sets = int(input("Enter the number of sets to produce: "))
    
    gen = SeqGen(lseqs=lseqs, num_sets=num_sets)
    gen.generate_sequences_and_proteins()
    gen.save_sequences_to_files()
    gen.save_sequences_to_fasta()
    gen.save_sequences_to_fastq()
    print(f"Generated {num_sets} sets of DNA and protein sequences.")