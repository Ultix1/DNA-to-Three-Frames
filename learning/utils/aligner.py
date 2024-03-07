from enum import Enum
import blosum as bl
from constants import CODON_TABLE, FRAMESHIFT_PENALTY, GAP_OPEN_PENALTY, GAP_EXTENSION_PENALTY
from constants import Action


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
        self.substitution = substition or bl.BLOSUM(62, default=0)
        self.backtrace = backtrace

    def _translate_codon(self, codon):
        return self.table.get(codon, 'FAIL')

    def _get_score(self, dna, protein, i, j):
        return self.substitution[self._translate_codon(dna[i-1:i+2])][protein[j-1]] if self._translate_codon(dna[i-1:i+2]) != 'FAIL' else float('-inf')

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
                    T[i-1][j],
                    T[i-2][j],
                    T[i-3][j]
                ])), key=lambda x: x[0])[1])

            i -= 3 if i % 3 == 0 else i % 3
            j -= 1

        return actions[::-1]

    def align(self, dna_input, protein_input, debug=False):
        # Define I, D, C, and Traceback matrices of size m x n
        N, M = len(dna_input), len(protein_input)
        I = [[float(0) for _ in range(M+1)] for _ in range(N)]
        D = [[float(0) for _ in range(M+1)] for _ in range(N)]
        C = [[float(0) for _ in range(M+1)] for _ in range(N)]
        T = [[Action.NONE for _ in range(M+1)] for _ in range(N)]

        # Initialization
        for j in range(M+1):
            I[j][0] = float('-inf')
            D[0][j] = D[2][j] = D[3][j] = float('-inf')
            D[1][j] = C[0][j] - self.gop - self.gep

        # Note: Placed j-1 for accessing protein_input since index out of bounds error
        C[0][0] = 0
        for j in range(1, M+1):
            C[0][j], T[0][j] = I[0][j], Action.INSERT
            C[j][0], T[j][0] = D[j][0], Action.DELETE

            C[1][j], T[1][j] = max(list(zip(
                [ 
                    I[1][j], 
                    D[1][j],
                    C[0][j-1] + self._get_score(dna_input, protein_input, 1, j)
                ], [
                    Action.INSERT, Action.DELETE, Action.MATCH
                ])), key=lambda x: x[0])

            C[2][j], T[2][j] = max(list(zip(
                [ 
                    I[2][j],
                    C[0][j-1] + self._get_score(dna_input, protein_input, 2, j) - self.frameshift                
                ], [
                    Action.INSERT, Action.FRAMESHIFT_3
                ])), key=lambda x: x[0])

            C[3][j], T[3][j] = max(list(zip(
                [ 
                    I[3][j],
                    C[1][j-1] + self._get_score(dna_input, protein_input, 3, j) - self.frameshift
                ], [
                    Action.INSERT, Action.FRAMESHIFT_1
                ])), key=lambda x: x[0])

            C[4][j], T[4][j] = max(list(zip(
                [
                    I[4][j],
                    D[4][j],
                    C[1][j-1] + self._get_score(dna_input, protein_input, 4, j),
                    C[2][j-1] + self._get_score(dna_input, protein_input, 4, j) - self.frameshift 
                ],[
                    Action.INSERT, Action.DELETE, Action.MATCH, Action.FRAMESHIFT_3
                ])), key=lambda x: x[0])

        # Matrix filling
        for i in range(N):
            for j in range(1, M+1):
                I[i][j] = max(I[i][j-1] - self.gep, C[i][j-1] - self.gop - self.gep)

                if i < 4:
                    continue

                D[i][j] = max(D[i-3][j] - self.gep, C[i-3][j] - self.gop - self.gep)

                C[i][j], T[i][j] = max(list(zip(
                    [ 
                        I[i][j],
                        D[i][j],
                        C[i-2][j-1] + self._get_score(dna_input, protein_input, i, j) - self.frameshift,
                        C[i-3][j-1] + self._get_score(dna_input, protein_input, i, j),
                        C[i-4][j-1] + self._get_score(dna_input, protein_input, i, j) - self.frameshift
                    ], [
                        Action.INSERT, Action.DELETE, Action.FRAMESHIFT_1, Action.MATCH, Action.FRAMESHIFT_3
                    ])), key=lambda x: x[0])

                if i == N-1 and j == M:
                    # Note: -1 in index is to account for 0-indexing
                    C[N-1][M], T[N-1][M] = max(list(zip(
                        [ 
                                C[N-1-1][M],
                                C[N-2-1][M] - self.frameshift,
                                C[N-3-1][M] - self.gop - self.gep - self.frameshift,
                                D[N-3-1][M] - self.frameshift - self.gep
                        ], [
                            Action.MATCH, Action.INSERT, Action.DELETE, Action.DELETE
                        ])), key=lambda x: x[0])

        score = C[N-1][M] if self.backtrace is self.Backtrace.GLOBAL else max([e[-1] for e in C])
        actions = self._traceback(C, T, N, M)

        if debug:
            self._matrix_printer([I, D, C, T])

        return score, actions


if __name__ == '__main__':
    dna_inputs = ['CTGGTGATG', 'ATGCGA', 'ATGCGATACGCTTGA', 'CTTGGTCCGAAT', 'CCCCACACA']
    protein_inputs = ['LVM', 'MR', 'MRIR', 'LGPL', 'PPT']
    aligner = ThreeFrameAligner()

    for dna_input, protein_input in zip(dna_inputs, protein_inputs):
        print(f'DNA: {dna_input}')
        print(f'Protein: {protein_input}\n')
        score, actions = aligner.align(dna_input, protein_input, debug=False)
        print(f'Score: {score}\n')
        print(f'Actions: {[e.name for e in actions]}\n')

    # with open("AA1.txt", "r") as a, open("DNA1.txt", "r") as b:
    #     dna = b.read()
    #     protein = a.read()
    #     print(f'DNA: {dna}')
    #     print(f'Protein: {protein}\n')
    #     score, actions = aligner.align(dna, protein, debug=False)
    #     print(f'Score: {score}\n')
    #     print(f'Actions: {[e.name for e in actions]}\n')

