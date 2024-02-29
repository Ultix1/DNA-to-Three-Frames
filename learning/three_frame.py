import blosum as bl

CODON_TABLE = {
    "TTT" : "F", "TTC" : "F", "TTA" : "L", "TTG" : "L",
    "CTT" : "L", "CTC" : "L", "CTA" : "L", "CTG" : "L",
    "ATT" : "I", "ATC" : "I", "ATA" : "I", "ATG" : "M",
    "GTT" : "V", "GTC" : "V", "GTA" : "V", "GTG" : "V",
    "TCT" : "S", "TCC" : "S", "TCA" : "S", "TCG" : "S",
    "CCT" : "P", "CCC" : "P", "CCA" : "P", "CCG" : "P",
    "ACT" : "T", "ACC" : "T", "ACA" : "T", "ACG" : "T",
    "GCT" : "A", "GCC" : "A", "GCA" : "A", "GCG" : "A",
    "TAT" : "Y", "TAC" : "Y", "TAA" : "*", "TAG" : "*",
    "CAT" : "H", "CAC" : "H", "CAA" : "Q", "CAG" : "Q",
    "AAT" : "N", "AAC" : "N", "AAA" : "K", "AAG" : "K",
    "GAT" : "D", "GAC" : "D", "GAA" : "E", "GAG" : "E",
    "TGT" : "C", "TGC" : "C", "TGA" : "*", "TGG" : "W",
    "CGT" : "R", "CGC" : "R", "CGA" : "R", "CGG" : "R",
    "AGT" : "S", "AGC" : "S", "AGA" : "R", "AGG" : "R",
    "GGT" : "G", "GGC" : "G", "GGA" : "G", "GGG" : "G",
}

def translate_codon(codon):
    return CODON_TABLE.get(codon, 'FAIL')

def get_score(scoring, dna, protein, i, j):
    return scoring[translate_codon(dna[i-1:i+2])][protein[j-1]] if translate_codon(dna[i-1:i+2]) != 'FAIL' else -999

def matrix_printer(matrices: list[list]):
    for matrix in matrices:
        for row in matrix:
            print(row)
        print()

def three_frame(dna_input, protein_input):
    N, M = len(dna_input), len(protein_input)
    gep, gop = 2, 3
    frameshift_penalty = 4
    I = [[0 for col in range(M+1)] for row in range(N)]
    D = [[0 for col in range(M+1)] for row in range(N)]
    C = [[0 for col in range(M+1)] for row in range(N)]
    scoring = bl.BLOSUM(62, default=0)

    # Initialization
    for j in range(M+1):
        I[j][0] = float('-inf')
        D[0][j] = D[2][j] = D[3][j] = float('-inf')
        D[1][j] = C[0][j] - gop - gep

    # Note: Placed j-1 for accessing protein_input since index out of bounds error
    C[0][0] = 0
    for j in range(1, M+1):
        C[0][j] = I[0][j]
        C[j][0] = D[j][0]
        C[1][j] = max(I[1][j],
                      D[1][j],
                      C[0][j-1] + get_score(scoring, dna_input, protein_input, 1, j))
        C[2][j] = max(I[2][j],
                      C[0][j-1] + get_score(scoring, dna_input, protein_input, 2, j) - frameshift_penalty)
        C[3][j] = max(I[3][j],
                      C[1][j-1] + get_score(scoring, dna_input, protein_input, 3, j) - frameshift_penalty)
        C[4][j] = max(I[4][j],
                      D[4][j],
                      C[1][j-1] + get_score(scoring, dna_input, protein_input, 4, j),
                      C[2][j-1] + get_score(scoring, dna_input, protein_input, 4, j) - frameshift_penalty)

    # Matrix filling
    for i in range(N):
        for j in range(1, M+1):
            I[i][j] = max(I[i][j-1] - gep, C[i][j-1] - gop - gep)
            if i < 4: continue
            D[i][j] = max(D[i-3][j] - gep, C[i-3][j] - gop - gep)
            C[i][j] = max(I[i][j],
                          D[i][j],
                          C[i-2][j-1] + get_score(scoring, dna_input, protein_input, i, j) - frameshift_penalty,
                          C[i-3][j-1] + get_score(scoring, dna_input, protein_input, i, j),
                          C[i-4][j-1] + get_score(scoring, dna_input, protein_input, i, j) - frameshift_penalty)
            if i == N-1 and j == M:
                # Note: -1 in index is to account for 0-indexing
                C[N-1][M] = max(C[N-1-1][M],
                                C[N-2-1][M] - frameshift_penalty,
                                C[N-3-1][M] - gop - gep - frameshift_penalty,
                                C[N-4-1][M] - frameshift_penalty - gep)

    matrix_printer([I, D, C])
    return C[N-1][M]

if __name__ == '__main__':
    dna_inputs = ['CTGGTGATG', 'ATGCGA', 'ATGCGATACGCTTGA', 'CTTGGTCCGAAT']
    protein_inputs = ['LVM', 'MR', 'MRIR', 'LGPL']
    # thing = bl.BLOSUM(62, default=0)[translate_codon(dna_input[0:3])][protein_input[1]]
    # print(thing)

    for dna_input, protein_input in zip(dna_inputs, protein_inputs):
        ans = three_frame(dna_input, protein_input)
        print(f'Score: {ans}\n')
        break