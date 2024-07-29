import os
import random
import time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from utils.fasta_reader import read_fasta
from models_v2.environment import Environment
from models_v2.main_agent import Agent
from models_v2.network import DDDQN
from params import PARAMS

# Directory Paths
checkpoint_paths = ["./saved_weights/main/main_checkpoint.weights.h5", "./saved_weights/target/target_checkpoint.weights.h5"]
test_output_path = "results/correctness_tests"
fasta_dir = "fasta_tests"

if __name__ == '__main__':
    if(os.path.isfile(checkpoint_paths[0]) and os.path.isfile(checkpoint_paths[1])):
        tests = []
        filenames = []

        # Loop through list of tests in fasta tests
        for fn in os.listdir(fasta_dir):
            if(fn != "fasta_files"):
                filenames.append(fn)
                tests.append(f"{fasta_dir}/{fn}")

        # Load Parameters
        input_shape = PARAMS['input_shape']
        actions = PARAMS['actions']
        learning_rate = PARAMS['lr']

        # Initialize Main and Target QN
        MainQN = DDDQN(learning_rate, len(actions), input_shape)
        TargetQN = DDDQN(learning_rate, len(actions), input_shape)

        # Initialize Environment and Agent
        environment = Environment(window_size=PARAMS['window_size'])
        agent = Agent(MainQN, TargetQN, environment, PARAMS, actions)

        # Load Weights
        agent.epsilon = 0.01
        agent.load_weights(checkpoint_paths[0], checkpoint_paths[1])
 
        # Load 1000 Reference Proteins
        fasta_file = "fasta_tests/fasta_files/fruit_fly.fasta"
        test_size = 1000
        reference_proteins = read_fasta(fasta_file, max_size=test_size, protein_len_range=(100, 300))

        for index in range(len(tests)):
            # Read Target DNA Sequence
            with open(f"{tests[index]}/translated_dna.txt", 'r') as target_dna_file:
                target_dna = target_dna_file.read().strip()

            # Read Protein ID
            with open(f"{tests[index]}/id.txt", 'r') as id_file:
                id = id_file.read().strip()

            # Create Output File
            with open(f"{test_output_path}/{filenames[index]}.txt", 'w') as save_file:
                save_file.write(f"Origin: {tests[index]}/protein.txt\n")
                save_file.write(f"Protein ID: {id}\n\n")

                header = "{:<10} {:<10} {:<30}".format("Score", "Reward", "Protein ID")
                save_file.write(header + "\n")

                # Compare to all other proteins
                for ref_id, ref_prot in reference_proteins.items():
                    # Set Protein and DNA
                    environment.set_seq(target_dna, ref_prot)
                    score, reward = agent.test("", "", "", save=False)
                    
                    output = "{:<10} {:<10} {:<30}".format(score, reward, ref_id)
                    print(output)

                    save_file.write(output + "\n")