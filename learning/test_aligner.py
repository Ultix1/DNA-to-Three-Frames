from utils.aligner import ThreeFrameAligner
import os

if __name__ == "__main__":
    aligner = ThreeFrameAligner(backtrace=ThreeFrameAligner.Backtrace.SEMI_GLOBAL)
    outuput_dir = "results/aligner_tests"
    data_dir = "fasta_tests"

    dna = []
    proteins = []
    ids = []

    for fn in os.listdir(data_dir):
        if(fn != "fasta_files"):
            file = open(f"{data_dir}/{fn}/id.txt", 'r')
            ids.append(file.read().strip())
            file.close()

            dna.append(f"{data_dir}/{fn}/translated_dna.txt")
            proteins.append(f"{data_dir}/{fn}/protein.txt")

    for i in range(len(ids)):
        file = open(f"{outuput_dir}/test_{i}", 'a')
        file.write(f"ID: {ids[i]}\n\n")

        for p in proteins:
            prot_seq = open(p, 'r').read().strip()
            dna_seq = open(dna[i], 'r').read().strip()
            score= aligner.align(dna_input=dna_seq, protein_input=prot_seq)

            file.write(f"Compared with: {p}\n")
            file.write(f"Alignment Score :{score}\n\n")

        file.close
