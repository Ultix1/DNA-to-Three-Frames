import blosum as bl
from constants import CODON_TABLE, FRAMESHIFT_PENALTY, GAP_OPEN_PENALTY, GAP_EXTENSION_PENALTY
from constants import Action


class ThreeFrameAligner():
    def __init__(self, 
                 gep=GAP_EXTENSION_PENALTY, 
                 gop=GAP_OPEN_PENALTY,
                 frameshift=FRAMESHIFT_PENALTY, 
                 table=CODON_TABLE,
                 substition=None):
        self.gep = gep
        self.gop = gop
        self.frameshift = frameshift
        self.table = table
        self.substitution = substition or bl.BLOSUM(62, default=0)

    def _translate_codon(self, codon):
        return self.table.get(codon, 'FAIL')

    def _get_score(self, scoring, dna, protein, i, j):
        return scoring[self._translate_codon(dna[i-1:i+2])][protein[j-1]] if self._translate_codon(dna[i-1:i+2]) != 'FAIL' else -999

    def _matrix_printer(self, matrices: list[list]):
        for matrix in matrices:
            for row in matrix:
                print(row)
            print()

    def _backtrace(self, C_matrix, action_matrix, N, M):
        actions = []
        i = N - 1
        j = M

        actions.append(action_matrix[i][j])
        while i > 1 and j > 1:
            best = max(enumerate([C_matrix[i][j-1], 
                                  C_matrix[i-1][j-1]]), 
                       key=lambda x: x[1])
            if best[0] == 0:
                actions.append(action_matrix[i][j-1])
                j -= 1
            else:
                actions.append(action_matrix[i-1][j-1])
                i -= 1
                j -= 1

        return actions[::-1]

    def align(self, dna_input, protein_input, print_matrix=False):
        # Define I, D, C matrices of size m x n
        N, M = len(dna_input), len(protein_input)
        I = [[float(0) for _ in range(M+1)] for _ in range(N)]
        D = [[float(0) for _ in range(M+1)] for _ in range(N)]
        C = [[float(0) for _ in range(M+1)] for _ in range(N)]
        action_matrix = [[Action.NONE for _ in range(M+1)] for _ in range(N)]

        # Initialization
        for j in range(M+1):
            I[j][0] = float('-inf')
            D[0][j] = D[2][j] = D[3][j] = float('-inf')
            D[1][j] = C[0][j] - self.gop - self.gep

        # Note: Placed j-1 for accessing protein_input since index out of bounds error
        C[0][0] = 0
        for j in range(1, M+1):
            C[0][j] = I[0][j]
            C[j][0] = D[j][0]

            # Row 1
            best = max(enumerate([
                I[1][j],
                D[1][j],
                C[0][j-1] + self._get_score(self.substitution, dna_input, protein_input, 1, j)
            ]), key=lambda x: x[1])

            C[1][j] = best[1]

            if best[0] == 0:
                action_matrix[1][j] = Action.INSERT
            elif best[0] == 1:
                action_matrix[1][j] = Action.DELETE
            else:
                action_matrix[1][j] = Action.MATCH

            # Row 2
            best = max(enumerate([
                I[2][j],
                C[0][j-1] + self._get_score(self.substitution, dna_input, protein_input, 2, j) - self.frameshift
            ]), key=lambda x: x[1])
                          
            C[2][j] = best[1]

            if best[0] == 0:
                action_matrix[2][j] = Action.INSERT
            else:
                action_matrix[2][j] = Action.MISMATCH

            # Row 3
            best = max(enumerate([
                I[3][j],
                C[1][j-1] + self._get_score(self.substitution, dna_input, protein_input, 3, j) - self.frameshift
            ]), key=lambda x: x[1])

            C[3][j] = best[1]

            if best[0] == 0:
                action_matrix[3][j] = Action.INSERT
            else:
                action_matrix[3][j] = Action.MISMATCH

            # Row 4
            best = max(enumerate([
                I[4][j],
                D[4][j],
                C[1][j-1] + self._get_score(self.substitution, dna_input, protein_input, 4, j),
                C[2][j-1] + self._get_score(self.substitution, dna_input, protein_input, 4, j) - self.frameshift
            ]), key=lambda x: x[1])

            C[4][j] = best[1]

            if best[0] == 0:
                action_matrix[4][j] = Action.INSERT
            elif best[0] == 1:
                action_matrix[4][j] = Action.DELETE
            elif best[0] == 2:
                action_matrix[4][j] = Action.MATCH
            else:
                action_matrix[4][j] = Action.MISMATCH

        # Matrix filling
        for i in range(N):
            for j in range(1, M+1):
                I[i][j] = max(I[i][j-1] - self.gep, C[i][j-1] - self.gop - self.gep)
                if i < 4: continue
                D[i][j] = max(D[i-3][j] - self.gep, C[i-3][j] - self.gop - self.gep)

                best = max(enumerate([
                    I[i][j],
                    D[i][j],
                    C[i-2][j-1] + self._get_score(self.substitution, dna_input, protein_input, i, j) - self.frameshift,
                    C[i-3][j-1] + self._get_score(self.substitution, dna_input, protein_input, i, j),
                    C[i-4][j-1] + self._get_score(self.substitution, dna_input, protein_input, i, j) - self.frameshift
                ]), key=lambda x: x[1])

                C[i][j] = best[1]

                if best[0] == 0:
                    action_matrix[i][j] = Action.INSERT
                elif best[0] == 1:
                    action_matrix[i][j] = Action.DELETE
                elif best[0] == 3:
                    action_matrix[i][j] = Action.MATCH
                else:
                    action_matrix[i][j] = Action.MISMATCH

                if i == N-1 and j == M:
                    # Note: -1 in index is to account for 0-indexing
                    best = max(enumerate([
                        C[N-1-1][M],
                        C[N-2-1][M] - self.frameshift,
                        C[N-3-1][M] - self.gop - self.gep - self.frameshift,
                        C[N-4-1][M] - self.frameshift - self.gep
                    ]), key=lambda x: x[1])

                    C[N-1][M] = best[1]

                    if best[0] == 0:
                        action_matrix[N-1][M] = Action.MATCH
                    else:
                        action_matrix[N-1][M] = Action.MISMATCH
                                    
        score = C[N-1][M]
        actions = self._backtrace(C, action_matrix, N, M)

        if print_matrix:
            self._matrix_printer([I, D, C])

        return score, actions, action_matrix


if __name__ == '__main__':
    dna_inputs = ['CTGGTGATG', 'ATGCGA', 'ATGCGATACGCTTGA', 'CTTGGTCCGAAT']
    protein_inputs = ['LVM', 'MR', 'MRIR', 'LGPL']
    aligner = ThreeFrameAligner()

    for dna_input, protein_input in zip(dna_inputs, protein_inputs):
        print(f'DNA: {dna_input}')
        print(f'Protein: {protein_input}\n')
        score, actions, matrix = aligner.align(dna_input, protein_input, print_matrix=True)
        print(f'Score: {score}\n')
        print(f'Actions: {[e.name for e in actions]}\n')
        print('Action Matrix:')
        for row in matrix:
            print([e.name for e in row])
        print()

