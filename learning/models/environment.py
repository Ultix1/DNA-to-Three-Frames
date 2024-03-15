import os
import blosum as bl
from datetime import datetime
from utils.encoder import get_codon_encoding, get_protein_encoding, get_table
from utils.constants import GAP_EXTENSION_PENALTY, GAP_OPEN_PENALTY, FRAMESHIFT_PENALTY, Action, PARAMS
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

        # Alignment History
        self.alignment_history = []


    def reset(self):
        """
        Resets the Environment, as well as the dna and protein pointers
        """
        # Reset Pointers
        self.dna_pointer = 0
        self.protein_pointer = 0

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

    def set_protein(self, protein: str):
        """
        Sets the protein sequence of the environment

        Args:
            protein (str): protein Sequence
        """
        self.protein_sequence = protein
        self.reset()

    def set_dna(self, dna: str):
        """
        Sets the dna sequence of the environment

        Args:
            dna (str): Dna Sequence
        """
        self.dna_sequence = dna
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

        state.append(self.encoded_proteins[self.protein_sequence[self.protein_pointer - 1]])

        # Add Current 3 reading frames
        for i in range(self.dna_pointer, self.dna_pointer + 3):
            codon = self.dna_sequence[i:i+3]
            state.append(self.encoded_codons[codon])

        # Previous and Current Protein
        state.append(self.encoded_proteins[self.protein_sequence[self.protein_pointer]])

        state = np.vstack(state).astype(np.float32)

        # Expand state
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
            state.append(self.encoded_proteins['_'])

        state.append(self.encoded_proteins['_'])

        frame_2 = self.dna_sequence[0: 3]
        frame_3 = self.dna_sequence[1: 4]

        # Set current frame 1 to gap protein / codon
        state.append(self.encoded_proteins['*'])
        state.append(self.encoded_codons[frame_2])
        state.append(self.encoded_codons[frame_3])

        # Current protein and Previous protein set to a gap protein
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
                reward += 0 if (validate(
                    action=action, 
                    curr_frame=self.table[codon], 
                    protein=protein
                )) else -2
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

                score += self.blosum_lookup(codon, protein) - FRAMESHIFT_PENALTY
                reward += 0 if (validate(
                    action=action,
                    curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]], 
                    protein=protein
                )) else -2
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
                score += self.blosum_lookup(codon, protein) - FRAMESHIFT_PENALTY
                reward += 0 if (validate(
                    action=action,
                    curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]], 
                    protein=protein
                )) else -2
                self.dna_pointer += 4

        # INSERTION and DELETION
        elif action == 3:
            prev_protein = self.protein_sequence[self.protein_pointer - 1]
            protein = self.protein_sequence[self.protein_pointer]
            if(protein == '*'):
                score += 0
                reward = -2
                self.dna_pointer += 3
            if(prev_protein == '*'):
                score += 0
                reward = -2
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
                # Insertion scores
                scores_1 = [
                    self.blosum_lookup(frame_1, prev_protein) - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY ), 
                    self.blosum_lookup(frame_2, prev_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY), 
                    self.blosum_lookup(frame_3, prev_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY )
                ]
                # Deletion scores
                scores_2 = [
                    self.blosum_lookup(frame_a, curr_protein) - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY ), 
                    self.blosum_lookup(frame_b, curr_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY), 
                    self.blosum_lookup(frame_c, curr_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY )
                ]
                if max(scores_1) >= max(scores_2):
                    score += max(scores_1)
                    self.dna_pointer += (2 + np.argmax(scores_1))
                else:
                    score += max(scores_2)
                    self.dna_pointer += (2 + np.argmax(scores_2))
                # Compare with aligner
                # score += max([self.blosum_lookup(x, prev_protein) for x in [frame_1, frame_2, frame_3]])
                reward += 0 if (validate(
                    action=action, 
                    proteins=[prev_protein, curr_protein], 
                    prev_frames=[self.table[frame_a],self.table[frame_b],self.table[frame_c]], 
                    curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]]
                )) else -2

                # self.dna_pointer += (2 + np.argmax(scores))

        # # INSERTION
        # elif action == 3:
        #     prev_protein = self.protein_sequence[self.protein_pointer - 1]
        #     protein = self.protein_sequence[self.protein_pointer]
        #     if(prev_protein == '*'):
        #         score += 0
        #         reward = -2
        #         self.dna_pointer += 3

        #     else:
        #         # Past Frames
        #         frame_a = self.dna_sequence[self.dna_pointer - 3 :self.dna_pointer] if self.dna_pointer > 2 else "TAG"
        #         frame_b = self.dna_sequence[self.dna_pointer - 2 :self.dna_pointer + 1]
        #         frame_c = self.dna_sequence[self.dna_pointer - 1 :self.dna_pointer + 2]
                
        #         # Current Frames
        #         frame_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
        #         frame_2 = self.dna_sequence[self.dna_pointer + 1 :self.dna_pointer + 4]
        #         frame_3 = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]
                
        #         # Current Protein
        #         curr_protein = self.protein_sequence[self.protein_pointer]
                
        #         # Past Protein
        #         prev_protein = self.protein_sequence[self.protein_pointer - 1]
        #         scores = [
        #             self.blosum_lookup(frame_1, prev_protein) - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY + 1), 
        #             self.blosum_lookup(frame_2, prev_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY), 
        #             self.blosum_lookup(frame_3, prev_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY + 1)
        #         ]
                
        #         # Compare with aligner
        #         score += max([self.blosum_lookup(x, prev_protein) for x in [frame_1, frame_2, frame_3]]) - GAP_EXTENSION_PENALTY
        #         reward += 0 if (validate(
        #             action=action, 
        #             proteins=[prev_protein, curr_protein], 
        #             prev_frames=[self.table[frame_a],self.table[frame_b],self.table[frame_c]], 
        #             curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]]
        #         )) else -2
        #         self.dna_pointer += (2 + np.argmax(scores))

        # # DELETION
        # elif action == 4:
        #     protein = self.protein_sequence[self.protein_pointer]
        #     if(protein == '*'):
        #         score += 0
        #         reward = -2
        #         self.dna_pointer += 3
                
        #     else:
        #         # Past Frames
        #         frame_a = self.dna_sequence[self.dna_pointer - 3 :self.dna_pointer] if self.dna_pointer > 2 else "TAG"
        #         frame_b = self.dna_sequence[self.dna_pointer - 2 :self.dna_pointer + 1]
        #         frame_c = self.dna_sequence[self.dna_pointer - 1 :self.dna_pointer + 2]
                
        #         # Current Frames
        #         frame_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
        #         frame_2 = self.dna_sequence[self.dna_pointer + 1 :self.dna_pointer + 4]
        #         frame_3 = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]

        #         # Current Protein
        #         curr_protein = self.protein_sequence[self.protein_pointer]

        #         # Past Protein
        #         prev_protein = self.protein_sequence[self.protein_pointer - 1]

        #         scores = [self.blosum_lookup(frame_a, curr_protein) - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY + 1), self.blosum_lookup(frame_b, curr_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY), self.blosum_lookup(frame_c, curr_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY + 1)]
                
        #         # Compare with aligner
        #         score += max([self.blosum_lookup(x, curr_protein) for x in [frame_a, frame_b, frame_c]]) - GAP_EXTENSION_PENALTY
        #         reward += 0 if (validate(
        #             action=action, 
        #             proteins=[prev_protein, curr_protein], 
        #             prev_frames=[self.table[frame_a],self.table[frame_b],self.table[frame_c]], 
        #             curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]]
        #         )) else -2
        #         self.dna_pointer += (2 + np.argmax(scores))
        
        elif action == 4:
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

            scores = [[
                self.blosum_lookup(frame_1, curr_protein) - FRAMESHIFT_PENALTY,
                self.blosum_lookup(frame_2, curr_protein),
                self.blosum_lookup(frame_3, curr_protein) - FRAMESHIFT_PENALTY,
                0
            ]]
            scores_1 = [
                self.blosum_lookup(frame_1, prev_protein) - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY ), 
                self.blosum_lookup(frame_2, prev_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY), 
                self.blosum_lookup(frame_3, prev_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY )
            ]
                # Deletion scores
            scores_2 = [
                self.blosum_lookup(frame_a, curr_protein) - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY ), 
                self.blosum_lookup(frame_b, curr_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY), 
                self.blosum_lookup(frame_c, curr_protein)  - (GAP_OPEN_PENALTY + GAP_EXTENSION_PENALTY )
            ]
            
            score += max(max(scores,scores_1,scores_2))
            reward += 0 if (validate(
                action=action,
                proteins=[prev_protein, curr_protein],
                prev_frames=[self.table[frame_a], self.table[frame_b], self.table[frame_c]],
                curr_frames=[self.table[frame_1], self.table[frame_2], self.table[frame_3]]
            )) else -2

            self.dna_pointer += np.argmax([
                self.blosum_lookup(frame_1, curr_protein) - FRAMESHIFT_PENALTY,
                self.blosum_lookup(frame_2, curr_protein),
                self.blosum_lookup(frame_3, curr_protein) - FRAMESHIFT_PENALTY
            ]) + 2

        self.protein_pointer += 1

        done = self.isDone()

        next_state = np.zeros(shape=PARAMS['input_shape']) if done else self.get_state()

        if record:
            self.alignment_history[-1].append(reward)

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

        # dna = "____" + self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
        # proteins = "*" + self.protein_sequence[self.protein_pointer]
        # _, true_action = self.aligner.align(dna, proteins, debug=False)

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
                reward += 0 if (validate(
                    action=action, 
                    curr_frame=self.table[codon], 
                    protein=protein
                )) else -2
                self.dna_pointer += 2

        # FRAMESHIFT 1
        elif action == Action.FRAMESHIFT_1.value:
            score += 0
            reward += -2
            self.dna_pointer += 2

        # FRAMESHIFT 3
        elif action == Action.FRAMESHIFT_3.value:
            protein = self.protein_sequence[self.protein_pointer]
            codon_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
            codon_2 = self.dna_sequence[self.dna_pointer + 1 : self.dna_pointer + 4]
            
            score += max([self.blosum_lookup(codon_1, protein), self.blosum_lookup(codon_2, protein)]) - GAP_EXTENSION_PENALTY
            reward += 0 if (validate_first(codon_1=self.table[codon_1],codon_2=self.table[codon_2], protein=protein)) else -2
            self.dna_pointer += 3

        # INSERTION and DELETION
        elif action == Action.INDEL.value:
            protein = self.protein_sequence[self.protein_pointer]
            if(protein == '*'):
                score += 0
                reward = 2
                self.dna_pointer += 2

            else:
                score += 0
                reward += -2
                self.dna_pointer += 2


        # # DELETION
        # elif action == Action.DELETE.value:
        #     protein = self.protein_sequence[self.protein_pointer]
        #     if(protein == '*'):
        #         score += 0
        #         reward = 2
        #         self.dna_pointer += 2

        #     else:
        #         score += 0
        #         reward += -2
        #         self.dna_pointer += 2
        
        elif action == Action.MISMATCH.value:
            protein = self.protein_sequence[self.protein_pointer]
            codon_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
            codon_2 = self.dna_sequence[self.dna_pointer + 1: self.dna_pointer + 4]

            scores = [
                    self.blosum_lookup(codon_1, protein),
                    self.blosum_lookup(codon_2, protein) - FRAMESHIFT_PENALTY,
                    0
                ]
            score += max(scores)
            reward += 0 if (validate_first(self.table[codon_1], self.table[codon_2], protein, action)) else -2
            self.dna_pointer += (
                np.argmax([
                    self.blosum_lookup(codon_1, protein),
                    self.blosum_lookup(codon_2, protein) - FRAMESHIFT_PENALTY
                ]) + 2
            )

        self.protein_pointer += 1

        if record:
            self.alignment_history[-1].append(reward)

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
        # Pad previous 3 Frames
        if self.dna_pointer == 0:
            frame_a = "000"
            frame_b = "000"
            frame_c = "000"
            prev_protein = "*"

        else:
            frame_a = self.dna_sequence[self.dna_pointer - 3 :self.dna_pointer] if self.dna_pointer > 2 else "TAG"
            frame_b = self.dna_sequence[self.dna_pointer - 2 :self.dna_pointer + 1]
            frame_c = self.dna_sequence[self.dna_pointer - 1 :self.dna_pointer + 2]
            prev_protein = self.protein_sequence[self.protein_pointer - 1]
        
        if self.dna_pointer < 2:
            frame_1 = "TAG"
            frame_2 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
            frame_3 = self.dna_sequence[self.dna_pointer + 1 :self.dna_pointer + 4]

        else:
            frame_1 = self.dna_sequence[self.dna_pointer : self.dna_pointer + 3]
            frame_2 = self.dna_sequence[self.dna_pointer + 1:self.dna_pointer + 4]
            frame_3 = self.dna_sequence[self.dna_pointer + 2 :self.dna_pointer + 5]

        curr_protein = self.protein_sequence[self.protein_pointer]

        self.alignment_history.append([
            self.table[frame_a],
            self.table[frame_b],
            self.table[frame_c],
            prev_protein,
            self.table[frame_1],
            self.table[frame_2],
            self.table[frame_3],
            curr_protein,
            action
        ])


    def save_aligment(self, filename_1 : str = None, filename_2 : str = None):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

        result_path = "results/debug"

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        with open(f"{result_path}/result_{dt_string}.txt", 'a') as file:
            file.write(f"ALIGNMENT RESULTS - {now}\n")
            file.write(f"DNA File: {filename_1}\n")
            file.write(f"Protein File: {filename_2}\n")

            predictions = []
            tallies = [0, 0, 0, 0, 0]
            for i in range(len(self.alignment_history)):
                entry = self.alignment_history[i]
                a = entry[-2]

                if(a == 0):
                    a = "MATCH"
                    predictions.append(Action.MATCH)
                    tallies[0] += 1 if entry[-1] < 0 else 0

                elif (a == 1):
                    a = "FRAMESHIFT_1"
                    predictions.append(Action.FRAMESHIFT_1)
                    tallies[1] += 1 if entry[-1] < 0 else 0

                elif (a == 2):
                    a = "FRAMESHIFT_3"
                    predictions.append(Action.FRAMESHIFT_3)
                    tallies[2] += 1 if entry[-1] < 0 else 0

                # elif (a == 3):
                #     a = "INSERTION"
                #     predictions.append(Action.INSERT)
                #     tallies[3] += 1 if entry[-1] < 0 else 0

                elif (a == 3):
                    a = "INDEL"
                    predictions.append(Action.INDEL)
                    tallies[3] += 1 if entry[-1] < 0 else 0

                else:
                    a = "MISMATCH"
                    predictions.append(Action.MISMATCH)
                    tallies[4] += 1 if entry[-1] < 0 else 0

                file.write(f"\nPrev. Frames: {entry[0]}, {entry[1]}, {entry[2]} => PP: {entry[3]}")
                file.write(f"\nCurr Frames: {entry[4]}, {entry[5]}, {entry[6]} => CP: {entry[7]}")
                file.write(f"\nAction: {entry[-2]} ({a})")
                file.write(f"\nReward of Action: {entry[-1]}\n")

            file.write("\nERROR COUNT\n")
            file.write(f"Matches: {tallies[0]}\n")
            file.write(f"F_1: {tallies[1]}\n")
            file.write(f"F_3: {tallies[2]}\n")
            file.write(f"InDel: {tallies[3]}\n")
            # file.write(f"Dels: {tallies[4]}\n")
            file.write(f"Mis: {tallies[4]}\n")

            # accuracy = sum(1 for x, y in zip(predictions, actions) if x == y) / len(predictions)
            # file.write(f"\nAccuracy: {accuracy}\n\n")
            file.close()

