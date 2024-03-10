import blosum as bl
from datetime import datetime
from utils.encoder import get_codon_encoding, get_protein_encoding, get_table
from utils.constants import GAP_EXTENSION_PENALTY, GAP_OPEN_PENALTY, FRAMESHIFT_PENALTY, Action
from utils.aligner import ThreeFrameAligner
from utils.step_validation import validate, validate_first
import numpy as np

class Environment:
    def __init__(self):
        """
        Initialize Dna to Protein Alignment Environment

        Args:
            dna (string): Reference DNA or Nucleuotide Sequence
            protein (string): Target Protein Sequence
        """
        self.rewards = bl.BLOSUM(62, default=0)
        self.aligner = ThreeFrameAligner()

        # Get Encoded Codons and Protein Table
        self.table = get_table()
        self.encoded_proteins = get_protein_encoding()
        self.encoded_codons = get_codon_encoding(self.encoded_proteins, self.table)

        # Initial Pointers
        self.dna_pointer = 0
        self.protein_pointer = 0
        
        # Tallies of Frameshift and Matches
        self.frameshifts = 0
        self.matches = 0

        # Alignment History
        self.alignment_history = []


    def reset(self):
        """
        Resets the Environment, dna and protein pointers
        """
        # Reset Pointers
        self.dna_pointer = 0
        self.protein_pointer = 0

        # Reset Tallies
        self.frameshifts = 0
        self.matches = 0

        # Reset Alignment History
        self.alignment_history = []


    def set_seq(self, dna : str, protein : str):
        """
        Sets the dna and protein sequence of the environment

        Args:
            dna (str): Dna Sequence
            protein (str): Protein Sequence
        """
        self.dna_sequence = dna
        self.protein_sequence = protein
        self.reset()


    def get_state(self):
        """
        Returns Current state of environment

        Returns:
            NDArray: 2d Matrix representing the current and past three frames and protein character
        """
        state = []

        # If second step, set gap codon as the previous first frame
        if self.dna_pointer < 3:
            state.append(self.encoded_proteins["*"])
            for i in range(self.dna_pointer - 2, self.dna_pointer):
                codon = self.dna_sequence[i:i+3]
                state.append(self.encoded_codons[codon])

        # Else, normally get previous 3 frames
        else:
            for i in range(self.dna_pointer - 3, self.dna_pointer):
                codon = self.dna_sequence[i:i+3]
                state.append(self.encoded_codons[codon])

        # Add Current 3 reading frames
        for i in range(self.dna_pointer, self.dna_pointer + 3):
            codon = self.dna_sequence[i:i+3]
            state.append(self.encoded_codons[codon])

        # Previous and Current Protein
        state.append(self.encoded_proteins[self.protein_sequence[self.protein_pointer - 1]])
        state.append(self.encoded_proteins[self.protein_sequence[self.protein_pointer]])

        state = np.vstack(state).astype(np.float32)

        # Reshape state (8, 21, 1)
        state = np.expand_dims(state, axis = -1)

        return state
    

    def get_first_state(self):
        """
        Get First State of the environment

        Returns:
            NDArray: 2d Matrix representing the 4 Zero rows, 2nd and 3rd reading frame
        """
        state = []

        # Previous 3 frames are set to gap proteins / codons
        for _ in range(3):
            state.append(self.encoded_proteins['*'])

        frame_2 = self.dna_sequence[0: 3]
        frame_3 = self.dna_sequence[1: 4]

        # Set current frame 1 to gap protein / codon
        state.append(self.encoded_proteins['*'])
        state.append(self.encoded_codons[frame_2])
        state.append(self.encoded_codons[frame_3])

        # Current protein and Previous protein set to a gap protein
        state.append(self.encoded_proteins['*'])
        state.append(self.encoded_proteins[self.protein_sequence[self.protein_pointer]])

        state = np.vstack(state).astype(np.float32)
        state = np.expand_dims(state, axis=-1)

        return state
    

    def step(self, action=0, record=False):
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

        if record:
            self.add_to_history(action)


        if self.dna_pointer < 3:
            dna = "_" + self.dna_sequence[self.dna_pointer - 2 : self.dna_pointer + 5]
            proteins = self.protein_sequence[self.protein_pointer - 1 : self.protein_pointer + 1]
            _, true_action = self.aligner.align(dna, proteins, debug=False)

        else:
            dna = self.dna_sequence[self.dna_pointer - 3 : self.dna_pointer + 5]
            proteins = self.protein_sequence[self.protein_pointer - 1 : self.protein_pointer + 1]
            _, true_action = self.aligner.align(dna, proteins, debug=False)

        # MATCH
        if action == 0:
            p = self.dna_pointer + 1
            codon = self.dna_sequence[p : p + 3]
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '*'):
                score += 0
                reward = -2
                self.dna_pointer += 3

            # If they match, set reward
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (validate(action=action,curr_frame=self.table[codon], protein_=protein)) else -2
                self.dna_pointer += 3

        # FRAMESHIFT 1
        elif action == 1:
            codon = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '*'):
                score += 0
                reward = -2
                self.dna_pointer += 2

            # Apply Deletion
            else:
                # Current Frames
                frame_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
                frame_2 = self.dna_sequence[self.dna_pointer + 1 :self.dna_pointer + 4]
                frame_3 = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]

                score += self.blosum_lookup(codon, protein)
                reward += 2 if (validate(action=action,curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]], protein_=protein)) else -2
                self.dna_pointer += 2

        # FRAMESHIFT 3
        elif action == 2:
            codon = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]
            protein = self.protein_sequence[self.protein_pointer]
            
            
            # Gap Protein
            if(protein == '*'):
                score += 0
                reward = -2
                self.dna_pointer += 4

            # Apply Insertion
            else:
                # Current Frames
                frame_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
                frame_2 = self.dna_sequence[self.dna_pointer + 1 :self.dna_pointer + 4]
                frame_3 = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (validate(action=action,curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]], protein_=protein)) else -2
                self.dna_pointer += 4

        # INSERTION
        elif action == 3:
            protein = self.protein_sequence[self.protein_pointer]
            if(protein == '*'):
                score += 0
                reward = 2
                self.dna_pointer += 3
            else:
                # Past Frames
                frame_a = self.dna_sequence[self.dna_pointer - 3 :self.dna_pointer] if self.dna_pointer > 2 else "TAG"
                frame_b = self.dna_sequence[self.dna_pointer - 2 :self.dna_pointer + 1]
                frame_c = self.dna_sequence[self.dna_pointer - 1 :self.dna_pointer + 2]
                # Current Frames
                frame_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
                frame_2 = self.dna_sequence[self.dna_pointer + 1 :self.dna_pointer + 4]
                frame_3 = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]
                # Current Protein
                curr_protein = self.protein_sequence[self.protein_pointer]
                # Past Protein
                prev_protein = self.protein_sequence[self.protein_pointer - 1]
                scores = [self.blosum_lookup(frame_1, prev_protein) - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY + 1), self.blosum_lookup(frame_2, prev_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY), self.blosum_lookup(frame_3, prev_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY + 1)]
                
                # Compare with aligner
                score += max([self.blosum_lookup(x, prev_protein) for x in [frame_1, frame_2, frame_3]]) - GAP_EXTENSION_PENALTY
                reward += 2 if (validate(action=action, proteins=[prev_protein, curr_protein], prev_frames=[self.table[frame_a],self.table[frame_b],self.table[frame_c]], curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]])) else -2
                self.dna_pointer += (2 + np.argmax(scores))

        # DELETION
        elif action == 4:
            protein = self.protein_sequence[self.protein_pointer]
            if(protein == '*'):
                score += 0
                reward = 2
                self.dna_pointer += 3
                
            else:
                # Past Frames
                frame_a = self.dna_sequence[self.dna_pointer - 3 :self.dna_pointer] if self.dna_pointer > 2 else "TAG"
                frame_b = self.dna_sequence[self.dna_pointer - 2 :self.dna_pointer + 1]
                frame_c = self.dna_sequence[self.dna_pointer - 1 :self.dna_pointer + 2]
                # Current Frames
                frame_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
                frame_2 = self.dna_sequence[self.dna_pointer + 1 :self.dna_pointer + 4]
                frame_3 = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]
                # Current Protein
                curr_protein = self.protein_sequence[self.protein_pointer]
                # Past Protein
                prev_protein = self.protein_sequence[self.protein_pointer - 1]

                scores = [self.blosum_lookup(frame_a, curr_protein) - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY + 1), self.blosum_lookup(frame_b, curr_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY), self.blosum_lookup(frame_c, curr_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY + 1)]
                
                # Compare with aligner
                score += max([self.blosum_lookup(x, curr_protein) for x in [frame_a, frame_b, frame_c]]) - GAP_EXTENSION_PENALTY
                reward += 2 if (validate(action=action, proteins=[prev_protein, curr_protein], prev_frames=[self.table[frame_a],self.table[frame_b],self.table[frame_c]], curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]])) else -2
                self.dna_pointer += (2 + np.argmax(scores))
        
        self.protein_pointer += 1

        done = self.isDone()

        next_state = np.zeros(shape=(8, 21, 1)) if done else self.get_state()

        return score, reward, done, next_state


    def first_step(self, action=0, record=False):
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

        if record:
            self.add_to_history(action)

        dna = "____" + self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
        proteins = "*" + self.protein_sequence[self.protein_pointer]
        _, true_action = self.aligner.align(dna, proteins, debug=False)

        # MATCH
        if action == Action.MATCH.value:
            # Set Pointer to 0 if at start
            p = 0
            codon = self.dna_sequence[p : p + 3]
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '*'):
                score += 0
                reward = 2
                self.dna_pointer += 2

            # If they match, set reward
            else:
                score += self.blosum_lookup(codon, protein)
                reward += 2 if (validate(action=action, curr_frame=self.table[codon], protein_=protein)) else -2
                self.dna_pointer += 2

        # FRAMESHIFT 1
        elif action == Action.FRAMESHIFT_1.value:
            score += 0
            reward += -2
            self.dna_pointer += 2

        # FRAMESHIFT 3
        elif action == Action.FRAMESHIFT_3.value:
            protein = self.protein_sequence[self.protein_pointer]
            codon_1 = self.dna_sequence[self.dna_pointer + 1 : self.dna_pointer + 4]
            codon_2 = self.dna_sequence[self.dna_pointer + 2 : self.dna_pointer + 5]
            
            score += max([self.blosum_lookup(codon_1, protein), self.blosum_lookup(codon_2, protein)]) - GAP_EXTENSION_PENALTY
            reward += 2 if (validate_first(codon_1 = self.table[codon_1],codon_2=self.table[codon_2], protein=protein)) else -2
            self.dna_pointer += 3

        # INSERTION
        elif action == Action.INSERT.value:
            score += 0
            reward += -2
            self.dna_pointer += 2

        # DELETION
        elif action == Action.DELETE.value:
            protein = self.protein_sequence[self.protein_pointer]
            if(protein == '*'):
                score += 0
                reward = 2
                self.dna_pointer += 2
            else:
                score += 0
                reward += -2
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
        if codon == "*":
            return 0

        return self.rewards[self.table[codon]][protein]

    def print_frames(self, action):
        """
        Prints translated codons of three frames, action chosen, and protein character

        Args: 
        """
        frame_1 = self.table[self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]]
        frame_2 = self.table[self.dna_sequence[self.dna_pointer + 1 : self.dna_pointer + 4]]
        frame_3 = self.table[self.dna_sequence[self.dna_pointer + 2 : self.dna_pointer + 5]]
        protein = self.protein_sequence[self.protein_pointer]

        print(f"Action: {action}, {frame_1}, {frame_2}, {frame_3}, {protein}")

    def isDone(self):
        return (self.dna_pointer + 5) >= len(self.dna_sequence) or self.protein_pointer >= len(self.protein_sequence)
    
    def add_to_history(self, action : int):
        p = self.dna_pointer
        self.alignment_history.append([
            self.table[self.dna_sequence[p : p + 3]],
            self.table[self.dna_sequence[p + 1: p + 4]],
            self.table[self.dna_sequence[p + 2: p + 5]],
            self.protein_sequence[self.protein_pointer],
            action
        ])


    def save_aligment(self, filename_1 : str = None, filename_2 : str = None):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        with open(f"results/result_{dt_string}.txt", 'a') as file:
            file.write(f"ALIGNMENT RESULTS - {now}\n")
            file.write(f"DNA File: {filename_1}\n")
            file.write(f"Protein File: {filename_2}\n")

            predictions = []
            for entry in self.alignment_history:
                a = entry[4]

                if(a == 0):
                    a = "MATCH"
                    predictions.append(Action.MATCH)

                elif (a == 1):
                    a = "FRAMESHIFT_1"
                    predictions.append(Action.FRAMESHIFT_1)

                elif (a == 2):
                    a = "FRAMESHIFT_3"
                    predictions.append(Action.FRAMESHIFT_3)

                elif (a == 3):
                    a = "INSERTION"
                    predictions.append(Action.INSERT)

                else:
                    a = "DELETION"
                    predictions.append(Action.DELETE)

                file.write(f"\nThree Frames: {entry[0]}, {entry[1]}, {entry[2]} <====> Protein: {entry[3]} - Action: {entry[4]} ({a})\n")
            
            aligner = ThreeFrameAligner()
            _, actions = aligner.align(self.dna_sequence, self.protein_sequence, debug=False)
            accuracy = sum(1 for x, y in zip(predictions, actions) if x == y) / len(predictions)
            file.write(f"\nAccuracy: {accuracy}\n\n")

