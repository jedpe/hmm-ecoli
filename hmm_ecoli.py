import mchmm
import random
import pickle
from Bio import SeqIO

# Set
res_dir = "/home/jorge/Documents/Maestria/" + \
    "Computational and Applied Mathematics/" + \
    "Discrete Mathematical Modeling/Project/Results/"

# Set the path for the FASTA file
fasta_file = "/home/jorge/Documents/Maestria/" + \
    "Computational and Applied Mathematics/" + \
    "Discrete Mathematical Modeling/Project/Data/" + \
    "GCF_000008865.2/ncbi_dataset/data/GCF_000008865.2/" + \
    "GCF_000008865.2_ASM886v2_genomic.fna"

# Read the records
records = list(SeqIO.parse(fasta_file, "fasta"))
rec = records[1]

# Create 2-state models with different initial states
for i in range(1, 20):
    # Set a HMM instance
    hmm = mchmm.HiddenMarkovModel()

    # Train the model with Baum-Welch algorithm
    obs_seq = str(rec.seq)
    p = random.uniform(0, 1)
    mod = hmm.from_baum_welch(obs_seq, states=['0', '1'], pi=[p, 1-p])

    # Make a graph
    graph = mod.graph_make()
    fname = f"pOSAK1_initial{i}_p{round(p, 2)}"
    graph.render(filename=f"{fname}", format="png",
                 directory=res_dir + "graphs")

    # Save the model
    with open(f"{res_dir + 'models'}/{fname}.pkl", "wb") as outfile:
        pickle.dump(mod, outfile)
