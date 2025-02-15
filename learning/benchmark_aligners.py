from utils.aligner import ThreeFrameAligner
from models_v2.environment import Environment
from models_v2.main_agent import Agent
from models_v2.network import DDDQN
from utils.sequence_gen import SeqGen
from pathlib import Path
from textwrap import dedent
from params import PARAMS
from multiprocessing import Process
import subprocess as sp
import psutil
import time
import glob

def gen_seqs(bp_len, mutated=True):
    retries = 10
    while True:
        seq_gen = SeqGen(lseqs=bp_len, num_sets=1)
        try:
            seq_gen.generate_sequences_and_proteins(mutated=mutated)
            seq_gen.save_sequences_to_files()
            seq_gen.save_sequences_to_fasta()
            break
        except Exception as e:
            if retries == 0:
                raise e
            retries -= 1


def seq_zhang(dna, protein, result: Path):
    aligner = ThreeFrameAligner()
    start = time.time()
    score, actions, align = aligner.align(dna, protein, debug=False)
    end = time.time()
    with result.open("a") as out:
        out.writelines(dedent("""
        +===================================+
                Sequential Zhang Test
        +===================================+
        """))
        out.writelines("{:11s} {}\n".format("Query:", dna))
        out.writelines("{:11s} {}\n".format("Reference:", protein))
        for i in range(len(actions)):
            out.writelines("{:7s} {:20s} ---> {}\n".format(f"[{i}]", actions[i], align[i]))
        out.writelines(f'score={score}\taction_len={len(actions)}\talign_len={len(align)}\texecution_time={end - start}\tave_mem_usage={int(aligner.ave_mem_usage)}')


def framerl_agent(dna, protein, result: Path):
    weights_path = ["./saved_weights/main/main_checkpoint.weights.h5", "./saved_weights/target/target_checkpoint.weights.h5"]
    main_path, target_path = Path(weights_path[0]), Path(weights_path[1])
    if (not main_path.is_file() or not target_path.is_file()):
        raise FileNotFoundError("Agent weights files are invalid or does not exists.")

    input_shape = PARAMS['input_shape']
    actions = PARAMS['actions']
    learning_rate = PARAMS['lr']

    main_qn = DDDQN(learning_rate, len(actions), input_shape)
    target_qn = DDDQN(learning_rate, len(actions), input_shape)
    environment = Environment(window_size=PARAMS['window_size'])
    agent = Agent(main_qn, target_qn, environment, PARAMS, actions)

    agent.epsilon = 0.01
    agent.load_weights(main_path, target_path)
    environment.set_seq(dna, protein)

    proc = psutil.Process()
    start = time.time()
    score, reward = agent.test("", "", "", save=False)
    mem_sample = proc.memory_info().rss
    end = time.time()

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
        
    with result.open("a") as out:
        out.writelines(dedent("""
        +===================================+
                    FrameRL Test
        +===================================+
        """))
        out.write(f"{''.join(dna_alignment)}\n")
        out.write(f"{''.join(alignment)}\n")
        out.write(f"{''.join(protein_alignment)}\n")
        out.write(f"{''.join(asterisks)}\n\n\n")

        out.write(f"DNA: {dna}\n")
        out.write(f"Protein {protein}\n\n")

        for entry in environment.alignment_history:
            action = entry["action"]
            pointers = entry["pointers"]
            reward = entry["reward"]
            out.write(f"Action: {action}, Occured at: {pointers}, Reward: {reward}\n")

        out.writelines(f'score={score}\talign_history_len={len(environment.alignment_history)}\texecution_time={end - start}\tave_mem_usage={mem_sample}')


def blastx(dna_fasta: Path, protein_fasta: Path, result: Path):
    start = time.time()
    try:
        cmd = sp.Popen(["makeblastdb", "-in", str(protein_fasta.resolve()), "-dbtype", "prot", "-out", "db.blast"], stdout=sp.DEVNULL)
        proc = psutil.Process(cmd.pid)
        mem_usage_db = 0
        num_samples = 0
        while cmd.poll() is None:
            mem_usage_db = (mem_usage_db * num_samples + proc.memory_info().rss) / (num_samples + 1)
            num_samples += 1
        cmd.wait()
        if cmd.returncode != 0:
            raise sp.SubprocessError()
        print("Database created successfully.")
    except Exception as e:
        print(f"Failed to create database. Error: {e}")
        raise e
    try:
        cmd = sp.Popen(["blastx", "-query", str(dna_fasta.resolve()), "-db", "db.blast"], text=True, stdout=sp.PIPE)
        proc = psutil.Process(cmd.pid)
        mem_usage_align = 0
        num_samples = 0
        while cmd.poll() is None:
            mem_usage_align = (mem_usage_align * num_samples + proc.memory_info().rss) / (num_samples + 1)
            num_samples += 1
        cmd.wait()
        if cmd.returncode != 0:
            raise sp.SubprocessError()
        end = time.time()
        for path in glob.glob("./db.blast.*"):
            Path(path).unlink()
        with result.open("a") as out:
            out.writelines(dedent("""
            +===================================+
                        BLASTX Test
            +===================================+
            """))
            stdout, _ = cmd.communicate()
            out.writelines(stdout)
            out.writelines(f'execution_time={end - start}\tave_mem_usage={int(mem_usage_align + mem_usage_db)}')
    except Exception as e:
        print(f"Failed to execute BLASTX. Error: {e}")
        raise e

def clustal(dna_fasta: Path, protein_fasta: Path, result: Path):
    start = time.time()
    clustal_file = Path("clustal.fasta")
    with clustal_file.open("w") as o:
        with dna_fasta.open("r") as i:
            dna = i.readlines()
            for _ in range(2):
                o.writelines(dna)
                o.writelines("\n")
    try:
        cmd = sp.Popen(["clustalo", "-i", str(clustal_file)], text=True, stdout=sp.PIPE)
        proc = psutil.Process(cmd.pid)
        mem_usage = 0
        num_samples = 0
        while cmd.poll() is None:
            mem_usage = (mem_usage * num_samples + proc.memory_info().rss) / (num_samples + 1)
            num_samples += 1
        cmd.wait()
        if cmd.returncode != 0:
            raise sp.SubprocessError()
        end = time.time()
        clustal_file.unlink()
        with result.open("a") as out:
            out.writelines(dedent("""
            +===================================+
                     Clustal Omega Test
            +===================================+
            """))
            stdout, _ = cmd.communicate()
            out.writelines(stdout)
            out.writelines(f'execution_time={end - start}\tave_mem_usage={int(mem_usage)}')
    except Exception as e:
        print(f"Failed to execute Clustal Omega. Error: {e}")
        raise e

def spawn_proc(target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join()

if __name__ == '__main__':
    base_pairs = [10, 30, 60, 100, 300, 500, 800, 1000, 1500, 3000, 4500, 6000, 7500, 9000, 13500]

    for base_pair_len in base_pairs:
        print(f"\nRunning benchmarks for seq_len={base_pair_len}")
        Path('./results/benchmarks').mkdir(parents=True, exist_ok=True)
        result = Path(f"./results/benchmarks/seqlen_{base_pair_len}.txt")
        result.unlink(missing_ok=True) # Remove older result log

        gen_seqs(base_pair_len, mutated=False)
        prot_fasta, dna_fasta = Path("AA1.fasta"), Path("DNA1.fasta")
        with prot_fasta.open("r") as a, dna_fasta.open("r") as b :
            spawn_proc(blastx, (dna_fasta, prot_fasta, result))
            spawn_proc(clustal, (dna_fasta, prot_fasta, result))

        gen_seqs(base_pair_len, mutated=True)
        prot_seq, dna_seq = Path("AA1.txt"), Path("DNA1.txt")
        with prot_seq.open("r") as a, dna_seq.open("r") as b :
            dna = b.read().strip()
            protein = a.read().strip()
            spawn_proc(framerl_agent, (dna, protein, result))
            spawn_proc(seq_zhang, (dna, protein, result))

        dna_fasta.unlink()
        prot_fasta.unlink()
        dna_seq.unlink()
        prot_seq.unlink()