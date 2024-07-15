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
test_output_path = "results/query_tests"
fasta_dir = "fasta_tests"

if __name__ == '__main__':
    if(os.path.isfile(checkpoint_paths[0]) and os.path.isfile(checkpoint_paths[1])):
        tests = []
        filenames = []

        # Loop through list of tests in fasta tests
        for fn in os.listdir(fasta_dir):
            if(fn != "fasta_files"):
                if(len(tests) < 20):
                    filenames.append(fn)
                    tests.append(f"{fasta_dir}/{fn}")

        input_shape = PARAMS['input_shape']
        actions = PARAMS['actions']
        learning_rate = PARAMS['lr']

        MainQN = DDDQN(learning_rate, len(actions), input_shape)
        TargetQN = DDDQN(learning_rate, len(actions), input_shape)

        environment = Environment(window_size=PARAMS['window_size'])
        agent = Agent(MainQN, TargetQN, environment, PARAMS, actions)

        agent.epsilon = 0.01
        agent.load_weights(checkpoint_paths[0], checkpoint_paths[1])
 
        for index in range(len(tests)):
            file_1 = open(f"{tests[index]}/translated_dna.txt", 'r')
            file_2 = open(f"{tests[index]}/protein.txt", 'r')
            file_3 = open(f"{tests[index]}/id.txt", 'r')

            dna = file_1.read().strip()
            protein = file_2.read().strip()
            id = file_3.read().strip()
            
            file_1.close()
            file_2.close()
            file_3.close()

            results = []
            
            save_file = open(f"{test_output_path}/{filenames[index]}.txt", 'a')
            save_file.write(f"Origin: {tests[index]}/protein.txt\n")
            save_file.write(f"Protein ID: {id}\n\n")

            # Compare to all other proteins
            for i in range(len(tests)):
                # Read Protein
                protein_file = open(f"{tests[i]}/protein.txt", 'r')
                ref_protein = protein_file.read().strip()
                protein_file.close()

                # Red Protein ID
                proteinID_file = open(f"{tests[i]}/id.txt", 'r')
                refID = proteinID_file.read().strip()
                proteinID_file.close()

                # Set Protein
                environment.set_seq(dna, ref_protein)
                score, reward = agent.test("", "", "", save=False)
                
                save_file.write(f"{score}\t{reward}\t{tests[i]}/protein.txt\t{refID}\n")

                # print([reward, score, f"{tests[i]}/protein.txt"])
                # save_file.write(f"Compared with {tests[i]}/protein.txt\n")
                # save_file.write(f"Alignment Score: {score}\n")
                # save_file.write(f"Agent Reward: {reward}\n\n\n")

            save_file.close()
            print(f"Test {index + 1} Done\n")

        # for i in range(len(tests)):
        #     file_1 = open(tests[i][0], 'r')
        #     file_2 = open(tests[i][1], 'r')

        #     dna = file_1.read().strip()
        #     protein = file_2.read().strip()

        #     file_1.close()
        #     file_2.close()

        #     index, target_protein = getN_chars(protein, N_CHARS)
        #     environment.set_seq(dna, protein)

        #     print(f"\nTarget Protein: {target_protein}, Index at: {index}")
        #     start = time.time()
        #     agent.find(filename_1=tests[i][0], filename_2=tests[i][1], target_protein=target_protein, protein_len=N_CHARS, save=True)
        #     duration = time.time() - start
        #     print(f"Time Taken: {duration} seconds\n")



# DNA: ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG
# Original Protein: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC => ~0
# Target: CCCCCCCCCC
# 
# Other Protein: AAAAAAAAAAAAAAAAAAAAAAA => -40
# Other Protein: AAAAAAAAAAAAAAAAAAAAAAA => -300
# Other Protein: AAAAAAAAAAAAAAAAAAAAAAA => -50