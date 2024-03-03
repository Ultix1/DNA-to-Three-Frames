
import colourmap as cm

genetic_code = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

proteins = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', "_"]

protein_mapping = {
    # Unique Colors are generated using colourmap package
    "A" : "#7F00FF", "C" : "#12C7E5", "D" : "#A4F89E", "E" : "#FF8E4A",
    "F" : "#6725FE", "G" : "#2ADCDC", "H" : "#BCED8F", "I" : "#FF6D38",
    "K" : "#4F4AFC", "L" : "#42EDD2", "M" : "#D4DC7F", "N" : "#FF4A25",
    "P" : "#376DF8", "Q" : "#5AF8C7", "R" : "#ECC76E", "S" : "#FF2512",
    "T" : "#1F8EF3", "V" : "#72FEBB", "W" : "#FFAB5C", "Y" : "#8CFEAC",
    "_" : "#FF0000"
}

# Assign New Hex Colors to each protein character
def assign_colors(num : int, proteins : list, protein_map: dict, verbose=False):
    colors = cm.generate(num, cmap='rainbow')
    for i in range(num):
        x, y, z = colors[i]
        protein_map[proteins[i]] = '#{:02X}{:02X}{:02X}'.format(int(x*255), int(y*255), int(z*255))

        if verbose:
            print(f"{proteins[i]} : " + '#{:02X}{:02X}{:02X}'.format(int(x*255), int(y*255), int(z*255)))

    return protein_map

# Converts Hex Color to Binary
def convert_hex(hex : str):
    hex_color = hex.lstrip('#')
    matrix = []
    for i in range(0, len(hex_color),  2):
        binary = bin(int(hex_color[i:i+2],  16))[2:].zfill(8)
        binary_list = [int(digit) for digit in binary]
        matrix.append(binary_list)

    return matrix

# Assign Converted binary colors to each codon
def get_codon_table():
    g = genetic_code.copy()
    for key, value in g.items():
        binary = convert_hex(protein_mapping[value])
        g[key] = binary

    return g

def get_protein_table():
    return assign_colors(len(proteins), proteins, protein_mapping)

def get_table():
    return genetic_code