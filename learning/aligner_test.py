from utils.aligner import ThreeFrameAligner
from utils.sequence_gen import SeqGen
from os import remove as remove_file

if __name__ == '__main__':
    # base_pairs = [10, 30, 60, 100, 300, 500, 800, 1000, 1500, 3000, 4500, 6000, 7500, 9000, 13500, 15000]
    base_pairs = [15, 15, 15, 15, 15]

    print("Three Frame Aligner Memory Test\n")
    for base_pair_len in base_pairs:
        retries = 10
        while True:
            seq_gen = SeqGen(lseqs=base_pair_len, num_sets=1)
            try:
                seq_gen.generate_sequences_and_proteins()
                seq_gen.save_sequences_to_files()
                break
            except Exception as e:
                if retries == 0:
                    raise e
                retries -= 1

        with open("AA1.txt", "r") as a, open("DNA1.txt", "r") as b:
            dna = b.read().strip()
            protein = a.read().strip()
            aligner = ThreeFrameAligner()
            _, actions, seq = aligner.align(dna, protein, debug=True)
            print(dna)
            print(protein)
            print(actions)
            print(seq)

        print(f'seq_len={base_pair_len}\tave_mem_usage={float(aligner.ave_mem_usage)}')
        remove_file("AA1.txt")
        remove_file("DNA1.txt")