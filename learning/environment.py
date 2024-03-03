import blosum as bl
# from color_mapping import get_codon_table, get_protein_table, convert_hex, get_table
from utils.encoder import get_codon_encoding, get_protein_encoding, get_table
import random
import numpy as np

class Environment:
    def __init__(self, dna: str, protein: str):
        """
        Initialize Dna to Protein Alignment Environment

        Args:
            dna (string): Reference DNA or Nucleuotide Sequence
            protein (string): Target Protein Sequence
        """
        self.rewards = bl.BLOSUM(62, default=0)

        # Whole dna and protein sequence
        self.dna_sequence = dna
        self.protein_sequence = protein

        # Get Color Mapping and Protein Table
        # self.color_table = get_codon_table()
        # self.protein_table = get_protein_table()
        self.table = get_table()
        self.protein_table = get_protein_encoding()
        self.encoded_table = get_codon_encoding(self.protein_table, self.table)

        # Initial Pointers
        self.dna_pointer = 0
        self.protein_pointer = 0
        
        # Tallies of Frameshift and Matches
        self.frameshifts = 0
        self.matches = 0

        # User Defined penalties and the Codon Table
        # self.penalties = penalties

        # Alignment History
        self.alignment_history = []


    def reset(self):
        """
        Resets the Environment

        Returns:
            NDArray (12,8): 2d Matrix representing the colors of current three frames and protein character
        """
        # Reset Pointers
        self.dna_pointer = 0
        self.protein_pointer = 0

        # Reset Tallies
        self.frameshifts = 0
        self.matches = 0

        # Reset Alignment History
        self.alignment_history = []

        return self.get_state()
    

    def get_state(self):
        """
        Returns Current state of environment

        Returns:
            NDArray (12,8): 2d Matrix representing the colors of current three frames and protein character
        """
        # colors = []
        # for i in range(self.dna_pointer, self.dna_pointer + 3):
        #     codon = self.dna_sequence[i:i+3]
        #     colors.append(self.color_table[codon])

        # colors.append(convert_hex(self.protein_table[self.protein_sequence[self.protein_pointer]]))
        
        # Stack the colors of the protein and 3 Frames into a 12x8 matrix
        # state = np.vstack(colors).astype(np.float32)

        state = []
        for i in range(self.dna_pointer, self.dna_pointer + 3):
            codon = self.dna_sequence[i:i+3]
            state.append(self.encoded_table[codon])

        state.append(self.protein_table[self.protein_sequence[self.protein_pointer]])
        state = np.vstack(state).astype(np.float32)

        # Reshape state
        reshaped_state = np.expand_dims(state, axis = -1)

        self.input_shape = state.shape

        return reshaped_state
    

    def step(self, action=0):
        """
        Performs the chosen action on the environment and moves the pointers accordingly

        Args:
            action (int, optional): Chosen Action (0 to 3). Defaults to 0.

        Returns:
            score: Alignment score of the action
            reward: Reward for the action
            Done: If it was the last action in the environment
            Next_State: Next current state following the action
        """
        score = 0
        reward = 0

        # Match
        if action == 0:
            self.print_frames(action)
            # Set Pointer to 0 if at start
            p = self.dna_pointer + 1
            codon = self.dna_sequence[p : p + 3]
            protein = self.protein_sequence[self.protein_pointer]

            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = 3
                self.dna_pointer += 3

            # If they match, set reward
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (self.table[codon] == protein) else -2
                self.dna_pointer += 3


        # Deletion
        elif action == 1:
            self.print_frames(action)
            codon = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = -2
                self.dna_pointer += 4

            # Apply Deletion
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (self.table[codon] == protein) else -2
                self.dna_pointer += 4


        # Insertion
        elif action == 2:
            self.print_frames(action)
            codon = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = -2
                self.dna_pointer += 2

            # Apply Insertion
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (self.table[codon] == protein) else -2
                self.dna_pointer += 2


        # None of the Codons Match
        elif action == 3:
            self.print_frames(action)
            p = self.dna_pointer
            codon_1 = self.dna_sequence[p : p + 3]
            codon_2 = self.dna_sequence[p + 1: p + 4]
            codon_3 = self.dna_sequence[p + 2: p + 5]
            protein = self.protein_sequence[self.protein_pointer]
            condition = (self.table[codon_1] != protein) and (self.table[codon_2] != protein) and (self.table[codon_3] != protein)
            
            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = -2
                self.dna_pointer += 2

            else:
                scores = [self.blosum_lookup(codon_1, protein), self.blosum_lookup(codon_2, protein), self.blosum_lookup(codon_3, protein),]
                score += np.max(scores)
                reward += (np.argmax(scores) + 1) if (condition) else -2 
                self.dna_pointer += 2
    
        self.protein_pointer += 1

        done = self.isDone()

        next_state = np.zeros(shape=(4, 21, 1)) if done else self.get_state()

        return score, reward, done, next_state


    def first_step(self, action=0):
        """
        Performs first step in the environment

        Args:
            action (int, optional): Chosen action taken (0 to 3). Defaults to 0.

        Returns:
            Returns:
            score: Alignment score of the action
            reward: Reward for the action
            Done: If it was the last action in the environment
            Next_State: Next current state following the action
        """
        score = 0
        reward = 0

        # Match
        if action == 0:
            self.print_frames(action)
            # Set Pointer to 0 if at start
            p = 0
            codon = self.dna_sequence[p : p + 3]
            protein = self.protein_sequence[self.protein_pointer]

            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = 3
                self.dna_pointer += 2

            # If they match, set reward
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (self.table[codon] == protein) else -2
                self.dna_pointer += 2

        # Deletion
        elif action == 1:
            self.print_frames(action)
            score += 0
            reward += -10
            self.dna_pointer += 2
            self.protein_pointer += 1

        # Insertion
        elif action == 2:
            self.print_frames(action)
            protein = self.protein_sequence[self.protein_pointer]
            codon_1 = self.dna_sequence[self.dna_pointer + 1 : self.dna_pointer + 4]
            codon_2 = self.dna_sequence[self.dna_pointer + 2 : self.dna_pointer + 5]

            if self.table[codon_1] == protein:
                score += self.blosum_lookup(codon_1, protein)
                reward += 2
                self.dna_pointer += 3
                
            elif self.table[codon_2] == protein:
                score += self.blosum_lookup(codon_2, protein)
                reward += 2
                self.dna_pointer += 4

            else:
                reward += -2
                score += np.max([self.blosum_lookup(codon_1, protein), self.blosum_lookup(codon_2, protein)])
                self.dna_pointer +=2

        # None of the Codons Match
        elif action == 3:
            self.print_frames(action)
            p = self.dna_pointer
            codon_1 = self.dna_sequence[p : p + 3]
            codon_2 = self.dna_sequence[p + 1: p + 4]
            codon_3 = self.dna_sequence[p + 2: p + 5]
            protein = self.protein_sequence[self.protein_pointer]
            condition = (self.table[codon_1] != protein) and (self.table[codon_2] != protein) and (self.table[codon_3] != protein)
            
            # Gap Protein
            if(protein == '_'):
                score += 0
                reward = -2
                self.dna_pointer += 2

            else:
                scores = [self.blosum_lookup(codon_1, protein), self.blosum_lookup(codon_2, protein), self.blosum_lookup(codon_3, protein),]
                score += np.max(scores)
                # reward += (np.argmax(scores) + 1) if (condition) else -2 
                reward += 2 if (condition) else -2
                self.dna_pointer += 2
    
        self.protein_pointer += 1

        return score, reward, self.isDone(), self.get_state()


    def blosum_lookup(self, codon, protein):
        """
        Gets score from Blosum 62 Matrix

        Args:
            codon: Codon consisting of 3 nuclotide characters
            protein : Target protein character

        Returns:
            score: Returns score from matrix
        """
        if codon == "_":
            return 0

        return self.rewards[self.table[codon]][protein]

    def print_frames(self, action):
        frame_1 = self.table[self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]]
        frame_2 = self.table[self.dna_sequence[self.dna_pointer + 1 : self.dna_pointer + 4]]
        frame_3 = self.table[self.dna_sequence[self.dna_pointer + 2 : self.dna_pointer + 5]]
        protein = self.protein_sequence[self.protein_pointer]

        print(f"Action: {action}, {frame_1}, {frame_2}, {frame_3}, {protein}")

    def isDone(self):
        return (self.dna_pointer + 5) >= len(self.dna_sequence) or self.protein_pointer >= len(self.protein_sequence)
    
