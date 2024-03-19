from enum import Enum
import blosum as bl
import psutil
from utils.sequence_gen import SeqGen
from utils.constants import Action, CODON_TABLE, FRAMESHIFT_PENALTY, GAP_OPEN_PENALTY, GAP_EXTENSION_PENALTY, NEG_INF
from os import remove as remove_file

class ThreeFrameAligner():

    class Backtrace(Enum):
        GLOBAL = "global"
        SEMI_GLOBAL = "semi-global"

    def __init__(self, 
                 gep=GAP_EXTENSION_PENALTY, 
                 gop=GAP_OPEN_PENALTY,
                 frameshift=FRAMESHIFT_PENALTY, 
                 table=CODON_TABLE,
                 substition=None,
                 backtrace: Backtrace = Backtrace.GLOBAL):
        self.gep = gep
        self.gop = gop
        self.frameshift = frameshift
        self.table = table
        self.substitution = substition or bl.BLOSUM(62)
        self.backtrace = backtrace
        self.process = psutil.Process()
        self.ave_mem_usage = 0

    def _translate_codon(self, codon):
        return self.table.get(codon, 'FAIL')

    def _get_score(self, dna, protein, i, j):
        return self.substitution[self._translate_codon(dna[i-1:i+2])][protein[j-1]] if self._translate_codon(dna[i-1:i+2]) != 'FAIL' else NEG_INF

    def _matrix_printer(self, matrices: list[list]):
        for matrix in matrices:
            for row in matrix:
                print(row)
            print()

    def _traceback(self, C, T, N, M):
        actions = []
        i = 0
        j = M

        if self.backtrace is self.Backtrace.GLOBAL:
            i = N
        if self.backtrace is self.Backtrace.SEMI_GLOBAL:
            i = max(reversed(list(enumerate([e[-1] for e in C], start=1))), key=lambda x: x[1])[0]

        while i > 0 and j > 0:
            actions.append(max(list(zip(
                [
                    C[i-1][j],
                    C[i-2][j],
                    C[i-3][j]
                ],
                [
                    Action(T[i-1][j]),
                    Action(T[i-2][j]),
                    Action(T[i-3][j])
                ])), key=lambda x: x[0])[1])

            i -= 3 if i % 3 == 0 else i % 3
            j -= 1

        return actions[::-1]

    def align(self, dna_input, protein_input, debug=False):
        # Define I, D, C, and Traceback matrices of size m x n
        N, M = len(dna_input), len(protein_input)
        I = [[int() for _ in range(M+1)] for _ in range(N)]
        D = [[int() for _ in range(M+1)] for _ in range(N)]
        C = [[int() for _ in range(M+1)] for _ in range(N)]
        T = [[int() for _ in range(M+1)] for _ in range(N)]

        # Initialization
        for j in range(M+1):
            I[j][0] = NEG_INF
            D[0][j] = D[2][j] = D[3][j] = NEG_INF
            D[1][j] = C[0][j] - self.gop - self.gep

        # Note: Placed j-1 for accessing protein_input since index out of bounds error
        C[0][0] = 0
        for j in range(1, M+1):
            C[0][j], T[0][j] = I[0][j], Action.INSERT.value
            C[j][0], T[j][0] = D[j][0], Action.DELETE.value

            C[1][j], T[1][j] = max(list(zip(
                [ 
                    I[1][j], 
                    D[1][j],
                    C[0][j-1] + self._get_score(dna_input, protein_input, 1, j)
                ], [
                    Action.INSERT.value, Action.DELETE.value, Action.MATCH.value
                ])), key=lambda x: x[0])

            C[2][j], T[2][j] = max(list(zip(
                [ 
                    I[2][j],
                    C[0][j-1] + self._get_score(dna_input, protein_input, 2, j) - self.frameshift                
                ], [
                    Action.INSERT.value, Action.FRAMESHIFT_3.value
                ])), key=lambda x: x[0])

            C[3][j], T[3][j] = max(list(zip(
                [ 
                    I[3][j],
                    C[1][j-1] + self._get_score(dna_input, protein_input, 3, j) - self.frameshift
                ], [
                    Action.INSERT.value, Action.FRAMESHIFT_1.value
                ])), key=lambda x: x[0])

            C[4][j], T[4][j] = max(list(zip(
                [
                    I[4][j],
                    D[4][j],
                    C[1][j-1] + self._get_score(dna_input, protein_input, 4, j),
                    C[2][j-1] + self._get_score(dna_input, protein_input, 4, j) - self.frameshift 
                ],[
                    Action.INSERT.value, Action.DELETE.value, Action.MATCH.value, Action.FRAMESHIFT_3.value
                ])), key=lambda x: x[0])

        # Matrix filling
        num_samples = 0
        for i in range(N):
            for j in range(1, M+1):

                self.ave_mem_usage = (self.ave_mem_usage * num_samples + self.process.memory_info().rss) / (num_samples + 1)
                num_samples += 1

                I[i][j] = max(I[i][j-1] - self.gep, C[i][j-1] - self.gop - self.gep)

                if i < 4:
                    continue

                D[i][j] = max(D[i-3][j] - self.gep, C[i-3][j] - self.gop - self.gep)

                C[i][j], T[i][j] = max(list(zip(
                    [ 
                        C[i-4][j-1] + self._get_score(dna_input, protein_input, i, j) - self.frameshift,
                        C[i-3][j-1] + self._get_score(dna_input, protein_input, i, j),
                        C[i-2][j-1] + self._get_score(dna_input, protein_input, i, j) - self.frameshift,
                        D[i][j],
                        I[i][j]
                    ], [
                        Action.FRAMESHIFT_1.value, Action.MATCH.value, Action.FRAMESHIFT_3.value, Action.DELETE.value, Action.INSERT.value
                    ])), key=lambda x: x[0])

                if i == N-1 and j == M:
                    # Note: -1 in index is to account for 0-indexing
                    C[N-1][M], T[N-1][M] = max(list(zip(
                        [ 
                                D[N-3-1][M] - self.frameshift - self.gep,
                                C[N-3-1][M] - self.gop - self.gep - self.frameshift,
                                C[N-2-1][M] - self.frameshift,
                                C[N-1-1][M],
                        ], [
                            Action.DELETE.value, Action.DELETE.value, Action.INSERT.value, Action.MATCH.value
                        ])), key=lambda x: x[0])

        score = C[N-1][M] if self.backtrace is self.Backtrace.GLOBAL else max([e[-1] for e in C])
        actions = self._traceback(C, T, N, M)

        if debug:
            self._matrix_printer([I, D, C, T])

        return score, actions


if __name__ == '__main__':
    base_pairs = [10, 30, 60, 100, 300, 500, 800, 1000, 1500, 3000, 4500, 6000, 7500, 9000, 13500, 15000]

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
            _, _ = aligner.align(dna, protein, debug=False)

        print(f'seq_len={base_pair_len}\tave_mem_usage={float(aligner.ave_mem_usage)}')
        remove_file("AA1.txt")
        remove_file("DNA1.txt")
