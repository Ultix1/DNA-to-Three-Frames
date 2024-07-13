import os
import blosum as bl
from datetime import datetime
from utils.encoder import get_codon_encoding, get_protein_encoding, get_table
from utils.constants import GAP_EXTENSION_PENALTY, GAP_OPEN_PENALTY, FRAMESHIFT_PENALTY, Action
from params import PARAMS
from utils.aligner import ThreeFrameAligner
from utils.step_validation import validate
import numpy as np

class Environment:
    def __init__(self, window_size=1):
        """
        Initialize Dna to Protein Alignment Environment

        Args:
            dna (string): Reference DNA or Nucleuotide Sequence
            protein (string): Target Protein Sequence
        """
        self.rewards = bl.BLOSUM(62, default=0)
        self.aligner = ThreeFrameAligner(backtrace=ThreeFrameAligner.Backtrace.SEMI_GLOBAL)
        self.window_size = window_size

        # Get Encoded Codons and Protein Table
        self.table = get_table()
        self.encoded_proteins = get_protein_encoding()
        self.encoded_codons = get_codon_encoding(self.encoded_proteins, self.table)

        # Initial Pointers
        self.dna_pointer = 4
        self.protein_pointer = 1
        self.dna_len = 0
        self.protein_len = 0

        # Alignment History
        self.alignment_history = []

    def pad_sequences(self):
        """
        Pads the DNA and Protein Sequences
        """
        if(len(self.dna_sequence) > 0 and len(self.protein_sequence) > 0):
            dna_pad = ["000" for _ in range(self.window_size)]
            protein_pad = ["*" for _ in range(self.window_size)]

            # '0000' + DNA + Pad
            self.dna_sequence = "0000" + self.dna_sequence + ''.join(dna_pad)

            # '*' + Protein + Pad
            self.protein_sequence = "*" + self.protein_sequence + ''.join(protein_pad)

    def reset(self):
        """
        Resets the Environment, as well as the dna and protein pointers
        """
        # Reset Pointers
        self.dna_pointer = 4        # Accounting for the default padding on the left (0000)
        self.protein_pointer = 1    # Accounting for the default padding on the left (*)

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
        self.dna_len = len(self.dna_sequence)
        self.protein_len = len(self.protein_sequence)
        self.pad_sequences()
        self.reset()

    def set_protein(self, protein: str):
        """
        Sets the protein sequence of the environment

        Args:
            protein (str): protein Sequence
        """
        self.protein_sequence = protein
        self.protein_len = len(protein)
        self.pad_sequences()
        self.reset()

    def set_dna(self, dna: str):
        """
        Sets the dna sequence of the environment

        Args:
            dna (str): Dna Sequence
        """
        self.dna_sequence = dna
        self.dna_len = len(dna)
        self.pad_sequences()
        self.reset()

    def get_state(self):
        state = []
        encoded_proteins = get_protein_encoding()
        encoded_codons = get_codon_encoding(encoded_proteins, get_table())

        # NOTE: We considered Padded Codons: 000 == 00* == 0** == *00 == **0, where * is any nucleotide (A, C, T, G)

        # Append Previous 3 Frames (3 Codons)
        for i in range(self.dna_pointer - 4, self.dna_pointer - 1):
           
            # If the current codon contains a padding, append 000 instead
            state.append(encoded_codons[self.get_codon_by_index(i)])

        # Append Previous Protein
        state.append(encoded_proteins[self.protein_sequence[self.protein_pointer - 1]])

        # Append Current Window Frames (N Codons)
        for i in range(self.dna_pointer - 1, self.dna_pointer + (self.window_size * 3) - 1):
            
            # If the current codon contains a padding, append 000 instead
            state.append(encoded_codons[self.get_codon_by_index(i)])

        # Append Current Window Protein
        for i in range(self.protein_pointer, self.protein_pointer + self.window_size):
            state.append(encoded_proteins[self.protein_sequence[i]])

        # Convert to np array with Float Type
        state = np.vstack(state).astype(np.float32)

        # Expand state
        state = np.expand_dims(state, axis = -1)

        return state
    
    def get_codon_by_index(self, index):
        return "000" if "0" in self.dna_sequence[index : index + 3] else self.dna_sequence[index : index + 3]

    def isDone(self):
        return (self.dna_pointer) >= self.dna_len or self.protein_pointer >= self.protein_len

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
            codon = self.get_codon_by_index(self.dna_pointer)
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
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '*'):
                score += 0
                reward = -2
                self.dna_pointer += 2

            # Apply Deletion
            else:
                # Current Frames
                frame_1 = self.get_codon_by_index(self.dna_pointer - 1)
                frame_2 = self.get_codon_by_index(self.dna_pointer)
                frame_3 = self.get_codon_by_index(self.dna_pointer + 1)

                score += self.blosum_lookup(frame_1, protein) - FRAMESHIFT_PENALTY
                reward += 0 if (validate(
                    action=action,
                    curr_frames=[self.table[frame_1],self.table[frame_2],self.table[frame_3]], 
                    protein=protein
                )) else -2

                self.dna_pointer += 2

        # FRAMESHIFT 3
        elif action == 2:
            codon = self.get_codon_by_index(self.dna_pointer + 1)
            protein = self.protein_sequence[self.protein_pointer]
            
            # Gap Protein
            if(protein == '*'):
                score += 0
                reward = -2
                self.dna_pointer += 4

            # Apply Insertion
            else:
                # Current Frames
                frame_1 = self.get_codon_by_index(self.dna_pointer - 1)
                frame_2 = self.get_codon_by_index(self.dna_pointer)
                frame_3 = self.get_codon_by_index(self.dna_pointer + 1)
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
                frame_a = self.get_codon_by_index(self.dna_pointer - 4)
                frame_b = self.get_codon_by_index(self.dna_pointer - 3)
                frame_c = self.get_codon_by_index(self.dna_pointer - 2)
                
                # Current Frames
                frame_1 = self.get_codon_by_index(self.dna_pointer - 1)
                frame_2 = self.get_codon_by_index(self.dna_pointer)
                frame_3 = self.get_codon_by_index(self.dna_pointer + 1)
                
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

                # If Insertion Score is greater than Deletion Score
                if max(scores_1) >= max(scores_2):
                    score += max(scores_1)
                    self.dna_pointer += (2 + np.argmax(scores_1))

                # If Deletion Score is greater than Insertion Score
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

        # MISMATCH
        elif action == 4:
            # Past Frames
            frame_a = self.get_codon_by_index(self.dna_pointer - 4)
            frame_b = self.get_codon_by_index(self.dna_pointer - 3)
            frame_c = self.get_codon_by_index(self.dna_pointer - 2)
            
            # Current Frames
            frame_1 = self.get_codon_by_index(self.dna_pointer - 1)
            frame_2 = self.get_codon_by_index(self.dna_pointer)
            frame_3 = self.get_codon_by_index(self.dna_pointer + 1)

            # Current Protein
            curr_protein = self.protein_sequence[self.protein_pointer]

            # Past Protein
            prev_protein = self.protein_sequence[self.protein_pointer - 1]

            scores = [
                self.blosum_lookup(frame_1, curr_protein) - FRAMESHIFT_PENALTY,
                self.blosum_lookup(frame_2, curr_protein),
                self.blosum_lookup(frame_3, curr_protein) - FRAMESHIFT_PENALTY,
                0
            ]
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
            self.alignment_history[-1]["reward"] = reward

        return score, reward, done, next_state

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
    
    def add_to_history(self, action : int):

        # Pad previous 3 Frames
        frame_a = self.table[self.get_codon_by_index(self.dna_pointer - 4)]
        frame_b = self.table[self.get_codon_by_index(self.dna_pointer - 3)]
        frame_c = self.table[self.get_codon_by_index(self.dna_pointer - 2)]

        prev_protein = self.protein_sequence[self.protein_pointer - 1]

        curr_frames = []
        for i in range(self.dna_pointer - 1, self.dna_pointer + (self.window_size * 3) - 1):
            curr_frames.append(self.table[self.get_codon_by_index(i)])
        
        curr_proteins = []
        for i in range(self.protein_pointer, self.protein_pointer + self.window_size):
            curr_proteins.append(self.protein_sequence[i])

        self.alignment_history.append(
            {
                "prev_frames": [frame_a, frame_b, frame_c],
                "prev_protein": prev_protein,
                "curr_frames": curr_frames,
                "curr_protein": curr_proteins,
                "action": action
            }
        )


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

            for record in self.alignment_history:
                if(record['action'] == 0):
                    a = "MATCH"
                    predictions.append(Action.MATCH)
                    tallies[0] += 1 if record['reward'] < 0 else 0

                elif (record['action'] == 1):
                    a = "FRAMESHIFT_1"
                    predictions.append(Action.FRAMESHIFT_1)
                    tallies[1] += 1 if record['reward'] < 0 else 0

                elif (record['action'] == 2):
                    a = "FRAMESHIFT_3"
                    predictions.append(Action.FRAMESHIFT_3)
                    tallies[2] += 1 if record['reward'] < 0 else 0

                elif (record['action'] == 3):
                    a = "INDEL"
                    predictions.append(Action.INDEL)
                    tallies[3] += 1 if record['reward'] < 0 else 0

                else:
                    a = "MISMATCH"
                    predictions.append(Action.MISMATCH)
                    tallies[4] += 1 if record['reward'] < 0 else 0

                file.write(f"\nPrev. Frames: {record['prev_frames']} => PP: {record['prev_protein']}")
                file.write(f"\nCurr Frames: {record['curr_frames']} => CP: {record['curr_protein']}")
                file.write(f"\nAction: {record['action']} ({a})")
                file.write(f"\nReward of Action: {record['reward']}\n")
                
            file.write("\nERROR COUNT\n")
            file.write(f"Matches: {tallies[0]}\n")
            file.write(f"F_1: {tallies[1]}\n")
            file.write(f"F_3: {tallies[2]}\n")
            file.write(f"InDel: {tallies[3]}\n")
            file.write(f"Mis: {tallies[4]}\n")
            # file.write(f"Dels: {tallies[4]}\n")

            # accuracy = sum(1 for x, y in zip(predictions, actions) if x == y) / len(predictions)
            # file.write(f"\nAccuracy: {accuracy}\n\n")
            file.close()

# def get_state_test(dna, protein, window_size=2, dna_pointer=4, protein_pointer=1):
#     """
#     Returns Current state of environment

#     Returns:
#         NDArray: 2D Matrix representing the current and past three frames and protein character
#     """
#     state = []
#     chars = []
#     encoded_proteins = get_protein_encoding()
#     encoded_codons = get_codon_encoding(encoded_proteins, get_table())

#     # NOTE: We considered Padded Codons: 000 == 00* == 0** == *00 == **0, where * is any nucleotide (A, C, T, G)

#     # Append Previous 3 Frames (3 Codons)
#     for i in range(dna_pointer - 4, dna_pointer - 1):
        
#         # If the current codon contains a padding, append 000 instead
#         state.append(encoded_codons[
#             "000" if ("0" in dna[i:i+3]) else dna[i:i+3]
#         ])
#         chars.append(dna[i:i+3])

#     # Append Previous Protein
#     state.append(encoded_proteins[protein[protein_pointer - 1]])
#     chars.append(protein[protein_pointer - 1])

#     # Append Current Window Frames (N Codons)
#     for i in range(dna_pointer - 1, dna_pointer + 2 + ((window_size - 1) * 3)):
        
#         # If the current codon contains a padding, append 000 instead
#         state.append(encoded_codons[
#             "000" if ("0" in dna[i:i+3]) else dna[i:i+3]
#         ])
#         chars.append(dna[i:i+3])

#     # Append Current Window Protein
#     for i in range(protein_pointer, protein_pointer + window_size):
#         state.append(encoded_proteins[protein[i]])
#         chars.append(protein[i])

#     for i in range(len(state)):
#         print(state[i], "=>", chars[i])

#     state = np.vstack(state).astype(np.float32)

#     # Expand state
#     state = np.expand_dims(state, axis = -1)

#     print(f"DNA LEN: {len(dna)}")
#     print(f"PROTEIN LEN: {len(protein)}")
#     print(f"WINDOW SIZE: {window_size}")
#     print(f"STATE SHAPE: {state.shape}")

#     return state


