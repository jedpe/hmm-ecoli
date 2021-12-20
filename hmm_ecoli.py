import os
import pickle
import mchmm
import joblib as jlb
from Bio import SeqIO
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------------
def n_state_mod(obs_seq, n_states, outdir, n_iter=1000, n_init=20):
    '''
        Function to compute the best model out of n_init initializations
        of the model. The chosen model is the one with the highest
        log-likelihood computed by the score() function.
        ------------------------------------------------------------------
        obs_seq: A nx1 numpy array of the observed sequence encoded through
                 LabelEncoder.

        n_states: Number of states in the hidden MC.

        outdir: Directory to save the model as a pickle file.

        n_iter: Number of iterations until the algorithm stops if the tolerance
                is not reached.

        n_init: Number of times the model will be initialized.
    '''
    # Create and fit an initial HMM
    best_mod = hmm.MultinomialHMM(n_components=n_states)
    best_mod.fit(obs_seq)

    for i in range(n_init):
        # Initialize a HMM instance (Multinomial for discrete symbols)
        cur_mod = hmm.MultinomialHMM(n_components=n_states,
                                     n_iter=n_iter, tol=0.0001)

        # Fit the model with the observed sequence
        cur_mod.fit(obs_seq)

        # Keep the model with the highest log-likelihood
        if cur_mod.score(obs_seq) > best_mod.score(obs_seq):
            best_mod = cur_mod

    # Save the model as a pickle file
    outfile = os.path.join(outdir, f"best_{n_states}-state_model.pkl")
    with open(outfile, "wb") as handle:
        pickle.dump(best_mod, handle)

    # Return the best model
    return best_mod


# ----------------------------------------------------------------------------
def plot_convergence(n_states, modir, outdir):
    '''
        Function to plot the convergence of the selected models.
        ------------------------------------------------------------------
        n_states: Number of states. This will be used to select the
                  corresponding model.

        modir: The directory where the models are stored.

        outdir: Directory for the output plot.
    '''
    # Load the corresponding model
    infile = os.path.join(modir, f"best_{n_states}-state_model.pkl")
    with open(infile, "rb") as handle:
        mod = pickle.load(handle)

    # Extract the convergence history
    hist = list(mod.monitor_.history)

    # Plot the convergence
    outfile = os.path.join(outdir, f"convergence_{n_states}-state_model.png")
    plt.plot(range(1, len(hist) + 1), hist)
    plt.title(f"Convergence for the {n_states}-state model", fontsize=18)
    plt.xlabel("Iteration number", fontsize=16)
    plt.ylabel("log-likelihood", fontsize=14)
    plt.savefig(outfile)
    plt.clf()
    plt.close()


# ----------------------------------------------------------------------------
def make_digraphs(n_states, modir, outdir):
    '''
        Function to fit a mchmm model with the best model parameters
        and make the graph representation of the hidden states.
        ------------------------------------------------------------------
        n_states: Number of states. This will be used to select the
                  corresponding model.

        modir: The directory where the models are stored.

        outdir: Directory for the output graph.
    '''
    # Load the corresponding model
    infile = os.path.join(modir, f"best_{n_states}-state_model.pkl")
    with open(infile, "rb") as handle:
        mod = pickle.load(handle)

    # Extract the parameters
    mod_ep = mod.emissionprob_
    mod_tp = mod.transmat_

    # Initialize the mchmm model
    state_labs = [str(s) for s in range(n_states)]
    hmm_mod = mchmm.HiddenMarkovModel(observations=['A', 'C', 'G', 'T'],
                                      states=state_labs,
                                      tp=mod_tp, ep=mod_ep)

    # Make and save the graph representation
    gname = f"{n_states}-state_model_graph"
    graph = hmm_mod.graph_make(filename=gname, directory=outdir, format="png")
    graph.render()


# ----------------------------------------------------------------------------
def plot_states(n_states, obs_seq, modir, outdir):
    '''
        Function to fit a mchmm model with the best model parameters
        and plot the sequence of states.
        ------------------------------------------------------------------
        n_states: Number of states. This will be used to select the
                  corresponding model.

        modir: The directory where the models are stored.

        outdir: Directory for the output plot.
    '''
    # Load the corresponding model
    infile = os.path.join(modir, f"best_{n_states}-state_model.pkl")
    with open(infile, "rb") as handle:
        mod = pickle.load(handle)

    # Decode the observed DNA sequence
    vs = list(mod.decode(obs_seq)[1])

    # Plot the states
    outfile = os.path.join(outdir, f"sequence_{n_states}-state_model.png")
    plt.plot(range(1, len(obs_seq) + 1), vs)
    plt.title(f"Sequence of hidden states for the {n_states}-state model",
              fontsize=18)
    plt.xlabel("Position in the DNA sequence", fontsize=16)
    plt.ylabel("", fontsize=14)
    plt.savefig(outfile)
    plt.clf()
    plt.close()


# Set the output directories for results
res_dir = "/home/jorge/Documents/Maestria/" + \
    "Computational and Applied Mathematics/" + \
    "Discrete Mathematical Modeling/Project/Results/"

modir = os.path.join(res_dir, "models")
plotdir = os.path.join(res_dir, "plots")
graphdir = os.path.join(res_dir, "graphs")

# Set the path for the FASTA file
fasta_file = "/home/jorge/Documents/Maestria/" + \
    "Computational and Applied Mathematics/" + \
    "Discrete Mathematical Modeling/Project/Data/Sequence/" + \
    "GCF_000008865.2_ASM886v2_genomic.fna"

# Read the records
records = list(SeqIO.parse(fasta_file, "fasta"))
rec = records[1]

# Encode the DNA sequence
encoder = LabelEncoder()
obs_seq = encoder.fit_transform([x for x in str(rec.seq)]).reshape(-1, 1)

# Parallelize the model construction
jlb.Parallel(n_jobs=4)(jlb.delayed(n_state_mod)(obs_seq,
                                                s,
                                                modir)
                       for s in range(2, 10))

# Plot the convergence for each of the n-state models
jlb.Parallel(n_jobs=4)(jlb.delayed(plot_convergence)(s,
                                                     modir,
                                                     plotdir)
                       for s in range(2, 10))

# Make the graph representations for the HMMs
jlb.Parallel(n_jobs=4)(jlb.delayed(make_digraphs)(s,
                                                  modir, graphdir)
                       for s in range(2, 10))

# Plot the sequence of states
jlb.Parallel(n_jobs=4)(jlb.delayed(plot_states)(s,
                                                obs_seq,
                                                modir,
                                                plotdir)
                       for s in range(2, 10))
