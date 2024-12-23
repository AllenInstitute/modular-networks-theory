import os
import argparse
import time
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

from rnn import RNNAdaptive
from visualizations import plot_nested_stats_pairs, plot_eigenvalues
from utils import ld2da
from analysis_adaptive import adaptation_analysis

jax.config.update("jax_enable_x64", True)

####################################################
parser = argparse.ArgumentParser(
                    prog='Lyapunov RNN analyzer (adaptive neural net)',
                    description='Adapt and simulate an RNN in order to compute Lyapunov exponents and their statistics'
                    )
parser.add_argument('-e', '--seed', default=0 )
parser.add_argument('-d', '--debug', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-l', '--load', default=True, action=argparse.BooleanOptionalAction)
args  = parser.parse_args()
seed  = int(args.seed)
debug = args.debug
load = args.load
###################################################
plt.rcParams.update({'font.size': 16})

results_dir = 'results/adaptation'
sigma10 = 0.0
sigmaL0 = 5.0

if debug:
    qtab    = np.linspace(0.3, 0.9, 20)
    P_seqs  = ( 
                (1_000,),
                (32,32),
                (10,10,10)
              )
    steps_lyap = 50
    steps_sim = 700
    steps_adapt = 1_000
    eta = 0.2
else:
    qtab    = np.linspace(0.3, 0.9, 50)
    P_seqs  = ( 
                (10_000,),
                (100,100),
                (22,22,22)
              )
    steps_lyap = 200
    steps_sim = 700
    steps_adapt = 1_000
    eta = 0.2

key = jax.random.PRNGKey(seed)
net = RNNAdaptive()

stats_all = {}
stats_all2 = {}

stats_all['$q_L$'] = {}
stats_all2['$q_L$'] = {}

os.makedirs(results_dir, exist_ok=True)
fname_pkl = os.path.join(results_dir, 'adapt_lyap.pkl')

if load and os.path.exists(fname_pkl):
    print(f"Loading data from {fname_pkl}...")
    with open(fname_pkl, 'rb') as f:
        data_loaded = pickle.load(f)
        stats_all = data_loaded['stats_all']
        stats_all2 = data_loaded['stats_all2']
else:
    print("Simulating...")
    for Ps in P_seqs:
        start = time.time()
        L = len(Ps)
        print(f"Levels: {L}")
        label = f"L = {L}"
        qLtab_real = []
        qLtab_mf = []
        lyap_mf  = []
        sigmaL = []
        sigmaL_mf = []
        stats_arrays = []
        if L==1:  # make sure that in a 1-level system the initial sigma is non-zero.
            sigma10_=sigmaL0
        else:
            sigma10_=sigma10

        for qL in tqdm(qtab):
            key, subkey = jax.random.split(key)
            stats, qs, sigmas = adaptation_analysis(subkey, net, 
                                            save_pickle=False, 
                                            verbose=False, 
                                            figures=False, 
                                            stats_names=('MLE',),
                                            sigma1=sigma10_,
                                            sigmaL=sigmaL0,
                                            Ps=Ps, qL=qL, eta=eta, steps_lyap=steps_lyap, 
                                            steps_sim=steps_sim, steps_adapt=steps_adapt)
            stats_arrays.append(stats)
            qLtab_real.append(qs[-1])
            sigmaL.append(sigmas[-1])

            desired_qs = jnp.linspace(qL/L, qL, num=L)
            predicted_sigma2, predicted_R2 = net.mf_implicit(desired_qs)
            lyap_mf.append(jnp.log(np.max(predicted_R2))/2) # mf prediction: MAX Lyap. exp 
            sigmaL_mf.append(np.sqrt(predicted_sigma2[-1])) # mf prediction: sigmaL

        stats_arrays = ld2da(stats_arrays)
        print(stats_arrays)
        for k, value in stats_arrays.items():
            if k not in stats_all:
                stats_all[k] = {}
                stats_all2[k] = {}
            stats_all[k][label] = (qtab, value)
            stats_all2[k][label] = (sigmaL, value)

        stats_all2['$q_L$'][label]  = (sigmaL, qLtab_real)
        stats_all2['$q_L$']['MF: '+label]  = (sigmaL_mf, qtab)
        stats_all['MLE']['MF: '+label] = (qtab, lyap_mf)   
        stats_all['$q_L$'][label] = (qtab, qLtab_real)

        stop = time.time()
        print(f"Elapsed time: {stop-start:.2f}")

    data_all = {'stats_all': stats_all,
                'stats_all2': stats_all2,
                'P_seqs': P_seqs,
                'steps_lyap': steps_lyap,
                'steps_sim': steps_sim,
                'steps_adapt': steps_adapt,
                'eta': eta
               }
    with open(fname_pkl, 'wb') as f:
        pickle.dump(data_all, f)

    
fig, axs = plt.subplots(1, 3, figsize=(15, 3.5), constrained_layout=True)

for key, value in stats_all['$q_L$'].items():
    axs[0].plot(value[0], value[1], '.-', lw=2, label=key)
axs[0].set_xlabel('$\hat{q}_L$')
axs[0].set_ylabel('$q_L$')
axs[0].legend()

for key, value in stats_all2['$q_L$'].items():
    if key.startswith('MF:'):
        axs[1].plot(value[1], value[0], 'k--', alpha=0.7, lw=2)
    else:
        axs[1].plot(value[1], value[0], '.-', lw=2, label=key)
#axs[1].set_xlabel('$\hat{q}_L$')
axs[1].set_xlabel('$q_L$')
axs[1].set_ylabel('$\\sigma_L$')
axs[1].axhline(0, ls='--', color='black', alpha=0.5, lw=1)

for key, value in stats_all['MLE'].items():
    if key.startswith('MF:'):
        axs[2].plot(value[0], value[1], 'k--', alpha=0.7, lw=2)
    else:
        axs[2].plot(value[0], value[1], '.-', lw=2, label=key)
axs[2].set_xlabel('$\hat{q}_L$')
axs[2].set_ylabel('$\\lambda_{max}$')
axs[2].axhline(0, ls='--', color='black', alpha=0.5, lw=1)

fname_fig = os.path.join(results_dir, 'adapt_lyap.pdf')
plt.savefig(fname_fig, bbox_inches='tight')
