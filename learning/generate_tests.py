from utils.fasta_reader import convert_to_dna, read_fasta
import os

if __name__ == '__main__':
    output_dir = "fasta_tests"
    filename = "fasta_tests/fasta_files/fruit_fly.fasta"

    # Max Number of Protein to find/sample
    max_test_size = 10

    # Chosen Protein Length
    protein_length = 333

    # Select Proteins from FASTA File that have a specific PROTEIN LENGTH
    selected_proteins = read_fasta(filename, max_test_size, protein_length)

    counter = 0
    for key, value in selected_proteins.items():
        translated_dna = convert_to_dna(value)
        folder_path = f"{output_dir}/test_{counter}"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # # Save Protein to File
        file = open(f"{folder_path}/protein.txt", 'a')
        file.write(f"{value.strip()}")
        file.close()

        # # Save Translated DNA to File
        file = open(f"{folder_path}/translated_dna.txt", 'a')
        file.write(f"{translated_dna.strip()}")
        file.close()

        file = open(f"{folder_path}/id.txt", 'a')
        file.write(f"{key}")
        file.close()

        counter += 1


    



    