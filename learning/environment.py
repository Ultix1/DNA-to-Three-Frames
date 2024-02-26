import blosum as bl
from color_mapping import get_codon_table, get_protein_table, convert_hex, get_table
import random
import numpy as np

class Environment:
    def __init__(self, dna, protein, penalties):
        self.rewards = bl.BLOSUM(62, default=0)

        # Whole dna and protein sequence
        self.dna_sequence = dna
        self.protein_sequence = protein
        self.isStart = True

        # Get Color Mapping and Protein Table
        self.color_table = get_codon_table()
        self.protein_table = get_protein_table()
        self.table = get_table()

        # Initial Pointers
        self.dna_pointer = 0
        self.protein_pointer = 0
        
        # Tallies of Frameshift and Matches
        self.frameshifts = 0
        self.matches = 0

        # User Defined penalties and the Codon Table
        self.penalties = penalties

        # Alignment History
        self.alignment_history = []

    # Just resets
    def reset(self):
        # Reset Pointers
        self.dna_pointer = 0
        self.protein_pointer = 0

        # Reset Tallies
        self.frameshifts = 0
        self.matches = 0

        # Reset Alignment History
        self.alignment_history = []

        # Set Back to Start
        self.isStart = True

        return self.get_state()
    
    # Returns colors of the 3 Frames, and color of the Protein Character
    def get_state(self):
        colors = []
        for i in range(self.dna_pointer, self.dna_pointer + 3):
            codon = self.dna_sequence[i:i+3]
            colors.append(self.color_table[codon])

        colors.append(convert_hex(self.protein_table[self.protein_sequence[self.protein_pointer]]))
        
        # Stack the colors of the protein and 3 Frames into a 12x8 matrix
        state = np.vstack(colors).astype(np.float32)

        # Reshape state into (12, 8, 1)
        reshaped_state = np.expand_dims(state, axis = -1)
        reshaped_state = np.expand_dims(state, axis = 0)

        return reshaped_state
    
    # Returns Score, Done
    def step(self, action=0):
        score = 0
        reward = 0

        # Match
        if action == 0:
            p = self.dna_pointer + 1
            codon = self.dna_sequence[p : p + 3]
            protein = self.protein_sequence[self.protein_pointer]

            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = 5
                self.dna_pointer += 2 if self.isStart else 3

            # If they match, set reward
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (self.table[codon] == protein) else -2
                self.dna_pointer += 2 if self.isStart else 3
        

        # Deletion
        elif action == 1:
            p = self.dna_pointer
            codon = self.dna_sequence[p : p + 3]
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = 5
                self.dna_pointer += 2

            # Apply Deletion
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (self.table[codon] == protein) else -2
                self.dna_pointer += 2


        # Insertion
        elif action == 2:
            p = self.dna_pointer + 3
            codon = self.dna_sequence[p : p + 3]
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = 5
                self.dna_pointer += 4

            # Apply Insertion
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (self.table[codon] == protein) else -2
                self.dna_pointer += 2


        # None Matches
        elif action == 3:
            p = self.dna_pointer
            codon_1 = self.dna_sequence[p : p + 3]
            codon_2 = self.dna_sequence[p + 1: p + 4]
            codon_3 = self.dna_sequence[p + 2: p + 5]
            protein = self.protein_sequence[self.protein_pointer]
            condition = (self.table[codon_1] != protein) and (self.table[codon_2] != protein) and (self.table[codon_3] != protein)
            
            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = 5
                self.dna_pointer += 4

            else:
                score += np.argmax([
                    self.blosum_lookup(codon_1, protein),
                    self.blosum_lookup(codon_2, protein),
                    self.blosum_lookup(codon_3, protein),
                ])

                reward += 2 if (condition) else -2 
                self.dna_pointer += 2
        
        self.protein_pointer += 1

        done = self.isDone()
        next_state = np.zeros(shape=(1, 12, 8)) if self.isDone() else self.get_state()

        return score, reward, done, next_state


    # Get Reward from Blosum 62 Matrix
    def blosum_lookup(self, codon, protein):
        if codon == "_":
            return 0
        
        translated_codon = self.table[codon]
        return self.rewards[translated_codon][protein]
    

    def isDone(self):
        return self.dna_pointer >= len(self.dna_sequence) - 5 or self.protein_pointer >= len(self.protein_sequence)
    
