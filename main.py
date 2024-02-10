import time
import color_mapping

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

def main():
  genetic_code_v2=color_mapping.get_codon_table()
  # print(genetic_code_v2)
  start_time = time.time()
  DNA_length = 100000000
  with open("DNA.txt","r") as file:
    DNA = file.read().strip()
  output = []
  # output_v2 = []
  for _ in range(0,DNA_length-2,1):
    output.insert(_,genetic_code[DNA[_:_+3]])
    # output_*v2.insert(_,genetic_code_v2[DNA[_:_+3]]) 
  # print(output_v2)
  # "".join(output)
  # print(output)
  # print(len(DNA))
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Elapsed time : {elapsed_time} seconds")
  

if __name__ == "__main__":
  main()