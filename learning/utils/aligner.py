from enum import Enum
import blosum as bl
import psutil
from utils.constants import Action, CODON_TABLE, FRAMESHIFT_PENALTY, GAP_OPEN_PENALTY, GAP_EXTENSION_PENALTY, NEG_INF

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

    def _traceback(self, T, N, M, dna, protein):
        actions = []
        sequence = []
        i = N - 1
        j = M

        # Set pointer to correct location
        if Action(T[i][j]) is Action.MATCH:
            i -= 1

        while i > 0:
            action = Action(T[i][j])
            if action is Action.MATCH:
                if self._translate_codon(dna[i-1:i+2]) != protein[j-1]:
                    action = Action.MISMATCH
                sequence.append((dna[i-1:i+2], protein[j-1]))
                i -= 3
                j -= 1
            if action is Action.FRAMESHIFT_1:
                sequence.append((dna[i-2:i+2], protein[j-1]))
                i -= 4
                j -= 1
            if action is Action.FRAMESHIFT_3:
                sequence.append((dna[i:i+2], protein[j-1]))
                i -= 2
                j -= 1
            if action is Action.INSERT:
                sub_prot = ""
                while Action(T[i][j]) is Action.INSERT:
                    sub_prot = protein[j-1] + sub_prot
                    j -= 1
                sequence.append(("---", sub_prot))
            if action is Action.DELETE:
                sub_dna = ""
                while Action(T[i][j]) is Action.DELETE:
                    sub_dna = dna[i-1:i+2] + sub_dna
                    i -= 3
                sequence.append((sub_dna, "-"))
            actions.append(action)

        return actions[::-1], sequence[::-1]

    def align(self, dna_input, protein_input, debug=False):
        """
        Pairwise alignment of a DNA and Protein sequence using Zhang's Three Frame Algorithm

        Parameters:
            dna_input: str - DNA string
            protein_input: str - protein string
            debug=False: bool - print debug matrices

        Returns:
            score: int - maximal alignment score
            actions: list[Action] - list of actions that produces the alignment
            alignment: list[tuple[str, str]] - list of tuples containing DNA-Protein pairings
        """
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
                            Action.DELETE.value, Action.FRAMESHIFT_3.value, Action.FRAMESHIFT_1.value, Action.MATCH.value
                        ])), key=lambda x: x[0])

        if debug:
            self._matrix_printer([I, D, C, T])

        score = C[N-1][M]
        actions, alignment = self._traceback(T, N, M, dna_input, protein_input)

        return score, actions, alignment
