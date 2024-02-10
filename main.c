#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define DNA_length 100000000
#define output_length 100000000-2

#define CODON_TABLE_SIZE 64


typedef struct {
    char codon[4];
    char amino_acid;
} CodonEntry;

CodonEntry CODON_TABLE[CODON_TABLE_SIZE]= {
    {"TTT", 'F'}, {"TTC", 'F'}, {"TTA", 'L'}, {"TTG", 'L'},
    {"CTT", 'L'}, {"CTC", 'L'}, {"CTA", 'L'}, {"CTG", 'L'},
    {"ATT", 'I'}, {"ATC", 'I'}, {"ATA", 'I'}, {"ATG", 'M'},
    {"GTT", 'V'}, {"GTC", 'V'}, {"GTA", 'V'}, {"GTG", 'V'},
    {"TCT", 'S'}, {"TCC", 'S'}, {"TCA", 'S'}, {"TCG", 'S'},
    {"CCT", 'P'}, {"CCC", 'P'}, {"CCA", 'P'}, {"CCG", 'P'},
    {"ACT", 'T'}, {"ACC", 'T'}, {"ACA", 'T'}, {"ACG", 'T'},
    {"GCT", 'A'}, {"GCC", 'A'}, {"GCA", 'A'}, {"GCG", 'A'},
    {"TAT", 'Y'}, {"TAC", 'Y'}, {"TAA", '_'}, {"TAG", '_'},
    {"CAT", 'H'}, {"CAC", 'H'}, {"CAA", 'Q'}, {"CAG", 'Q'},
    {"AAT", 'N'}, {"AAC", 'N'}, {"AAA", 'K'}, {"AAG", 'K'},
    {"GAT", 'D'}, {"GAC", 'D'}, {"GAA", 'E'}, {"GAG", 'E'},
    {"TGT", 'C'}, {"TGC", 'C'}, {"TGA", '_'}, {"TGG", 'W'},
    {"CGT", 'R'}, {"CGC", 'R'}, {"CGA", 'R'}, {"CGG", 'R'},
    {"AGT", 'S'}, {"AGC", 'S'}, {"AGA", 'R'}, {"AGG", 'R'},
    {"GGT", 'G'}, {"GGC", 'G'}, {"GGA", 'G'}, {"GGG", 'G'}
};

char translate_codon(char *codon) {
    int i;
    for(i = 0; i < CODON_TABLE_SIZE; i++) {
        if(strcmp(codon, CODON_TABLE[i].codon) == 0) {
            return CODON_TABLE[i].amino_acid;
        }
    }
    return '0';
}


int main(){
  clock_t begin = clock();
  char *DNA = (char *) malloc((DNA_length)*sizeof(char));
  char *output = (char *) malloc((output_length)*sizeof(char));
  FILE *fp = fopen("DNA2.txt","r");
  fgets(DNA, DNA_length, fp);
  fclose(fp);
  for(int i=0;i<output_length;i++){
    char *CODON = (char *)malloc(4*sizeof(char));
    strncpy(CODON, DNA+i, 3);
    output[i] = translate_codon(CODON);
  }
  // printf("%s\n", output);
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%lf", time_spent);
  free(DNA);
  free(output);
  return 0;
}