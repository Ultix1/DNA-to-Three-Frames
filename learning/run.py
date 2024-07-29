import os
import random
import time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from models_v2.environment import Environment
from models_v2.main_agent import Agent
from models_v2.network import DDDQN
from params import PARAMS

# Directory Paths
checkpoint_paths = ["./saved_weights/main/main_checkpoint.weights.h5", "./saved_weights/target/target_checkpoint.weights.h5"]
test_output_path = "results/alignments"
fasta_dir = "fasta_tests"
NUM_TESTS = 20
GAP = 15

if __name__ == '__main__':
    if(os.path.isfile(checkpoint_paths[0]) and os.path.isfile(checkpoint_paths[1])):
        tests = []
        filenames = []

        # LOOP THROUGH EACH TEST IN FASTA TEST DIRECTORY
        for fn in os.listdir(fasta_dir):
            if(fn != "fasta_files"):
                if(len(tests) < NUM_TESTS):
                    filenames.append(fn)
                    tests.append(f"{fasta_dir}/{fn}")


        # INITIALIZATIONS
        input_shape = PARAMS['input_shape']
        actions = PARAMS['actions']
        learning_rate = PARAMS['lr']

        MainQN = DDDQN(learning_rate, len(actions), input_shape)
        TargetQN = DDDQN(learning_rate, len(actions), input_shape)

        environment = Environment(window_size=PARAMS['window_size'])
        agent = Agent(MainQN, TargetQN, environment, PARAMS, actions)

        agent.epsilon = 0.01
        agent.load_weights(checkpoint_paths[0], checkpoint_paths[1])
 
        # Initial Environment
        file_1 = open(f"{tests[0]}/translated_dna.txt", 'r')
        file_2 = open(f"{tests[0]}/protein.txt", 'r')

        dna = file_1.read().strip()
        protein = file_2.read().strip()
        
        file_1.close()
        file_2.close()
        environment.set_seq(dna, protein)

        # RUN AGENT ON EACH TEST
        for index in range(len(tests)):
            dna_pointer = 0
            protein_pointer = 0

            results = []
            dna_file = open(f"{tests[index]}/translated_dna.txt", 'r')
            dna = dna_file.read().strip()
            dna_file.close()
            
            agent.env.set_seq(dna, protein)
            score, reward = agent.test("", "", "", save=False)

            save_file = open(f"{test_output_path}/{filenames[index]}.txt", 'w')
            save_file.write(f"Origin: {tests[index]}/protein.txt\n")
            save_file.write(f"Protein ID: {id}\n\n")

            dna_alignment =     list(environment.dna_sequence)
            alignment =         list(" " * (len(environment.dna_sequence) + 100))
            protein_alignment = list(" " * (len(environment.dna_sequence) + 100))
            asterisks =         list(" " * (len(environment.dna_sequence) + 100))

            ins_counter = 0
            del_counter = 0


            for entry in environment.alignment_history:
                # Fetch pointers (x, y)
                x, y = entry["pointers"]
                counter = ins_counter + del_counter

                # IF MATCH
                if(entry["action"] == 0):
                    alignment[x + 1 + counter] = "|"
                    protein_alignment[x + 1 + counter] = environment.protein_sequence[y]

                # IF Frameshift 1
                elif (entry["action"] == 1):
                    alignment[x + counter] = "|"
                    asterisks[x + counter] = "*"
                    protein_alignment[x + counter] = environment.protein_sequence[y]

                # IF Frameshift 3
                elif (entry["action"] == 2):
                    alignment[x + 2 + counter] = "|"
                    asterisks[x + 2 + counter] = "*"
                    protein_alignment[x + 2 + counter] = environment.protein_sequence[y]

                # IF Insertion
                elif (entry["action"] == 3):
                    alignment[x + 1 + counter] = "|"
                    
                    for i in range(3):
                        dna_alignment[x + 1 + counter + i] = "_"

                    protein_alignment[x + 1 + counter] = environment.protein_sequence[y]

                # IF Deletion
                elif (entry["action"] == 4):
                    alignment[x + 1 + counter] = "|"
                    protein_alignment[x + 1 + counter] = "-"

                # # IF Substitution / Mismatch
                elif (entry["action"] == 5):
                    alignment[x + 1 + counter] = "|"
                    protein_alignment[x + 1 + counter] = environment.protein_sequence[y]
                
            save_file.write(f"{''.join(dna_alignment)}\n")
            save_file.write(f"{''.join(alignment)}\n")
            save_file.write(f"{''.join(protein_alignment)}\n")
            save_file.write(f"{''.join(asterisks)}\n\n\n")

            save_file.write(f"DNA: {dna}\n")
            save_file.write(f"Protein {protein}\n\n")

            for entry in environment.alignment_history:
                action = entry["action"]
                pointers = entry["pointers"]
                reward = entry["reward"]
                save_file.write(f"Action: {action}, Occured at: {pointers}, Reward: {reward}\n")

            save_file.close()