import os
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

from rnn import RNN2L
from utils import ld2da
from visualizations import plot_nested_stats
jax.config.update("jax_enable_x64", True)
fontsize = 25
plt.rcParams.update({'font.size': fontsize})
#plt.rcParams.update({'legend.labelspacing': 0.25})
####################################################
parser = argparse.ArgumentParser(
                    prog='Modular networks simulator',
                    description='Simulate an RNN in order to compute various statistics and compare to MF predictions.',
                    epilog='Finished')
parser.add_argument('-m', '--mu', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-t', '--thresholded', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-s', '--sigma', default='0.0')
parser.add_argument('-e', '--seed', default=np.random.randint(1e6) )
parser.add_argument('-d', '--debug', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-l', '--legend', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()
mu = args.mu
thresholded = args.thresholded
sigma = float(args.sigma)
seed  = int(args.seed)
debug = args.debug
legend = args.legend
###################################################
if debug:
    gtab = jnp.linspace(0.2, 10.0, 30)
    N0 = 30
    P  = 30
    steps_mf = 300
    steps_sim = 500
    steps_lyap = 100
    skip_init_steps = 100
else:
    gtab = jnp.linspace(0.2, 10.0, 50)
    N0 = 100
    P  = 100
    steps_mf = 3_000
    steps_sim = 22_000
    steps_lyap = 500         # only use last `steps_lyap` steps in the Lyapunov exponents computations
    skip_init_steps = 2_000    # avoid transient (for calculating q, qm, and d_PR)
symmetric_mu = False
symmetric_random = False

min_lyap_plot = -0.2
max_lyap_plot = 1.0
###################################################
key = jax.random.PRNGKey(seed)
net = RNN2L(thresholded=thresholded)

lyap_stats_all = []
q_all = []
qm_all = []
dpr_all = []
def generate_weights(key, g):
    key, subkey = jax.random.split(key)
    if mu: 
        J = net.generate_weights(subkey, 
                                    N0=N0, 
                                    P=P, 
                                    sigma=g, 
                                    sigma_mu=sigma,
                                    symmetric_mu=symmetric_mu,
                                    symmetric_random=symmetric_random)
    else:
        J = net.generate_weights(subkey, 
                                    N0=N0, 
                                    P=P, 
                                    sigma=sigma, 
                                    sigma_mu=g,
                                    symmetric_mu=symmetric_mu,
                                    symmetric_random=symmetric_random)
    return key, J

print("Simulations...")
start = time.time()
for g in tqdm(gtab):
    key, W = generate_weights(key, g)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape = (W.shape[0],), dtype=jnp.float64)
    _, Xall = net.evolve(W, x, steps_sim, save_trajectory=True)
    lyap_exps = net.lyapunov_exponents(W, Xall, steps=steps_lyap)
    
    lyap_stats = net.lyapunov_summary_stats(lyap_exps, stats = ('MLE', 
                                                           '2MLE', 
                                                           '3MLE', 
                                                           'KS entropy', 
                                                           'KY dimension'),
                                                       vmappable=True)
    lyap_stats_all.append(lyap_stats)
    
    Mall = net.population_activities(Xall, 
                                     N0, 
                                     P)
    q, qm, m = net.qs_evolution(Xall, 
                             Mall) 
    q, qm, m = net.steady_state_stats(q, 
                                   qm,
                                   m,
                                   skip_init_steps=skip_init_steps)
    q_all.append(q)
    qm_all.append(qm)
    dpr = net.participation_ratio_dimension(Xall, 
                                          skip_init_steps=skip_init_steps)
    dpr_all.append(dpr)
    
stop = time.time()
print(f"Elapsed time: {stop-start:.2f} seconds.")
###################################################
lyap_stats_all = ld2da(lyap_stats_all)
q = jnp.array(q_all)
qm = jnp.array(qm_all)
dpr_all = jnp.array(dpr_all)

if mu:
    sigmas_MF = gtab
    sigmas_mu_MF = gtab*0 + sigma
else:
    sigmas_MF = gtab*0 + sigma
    sigmas_mu_MF = gtab

print("Mean field...")
key, subkey = jax.random.split(key)
qs_MF, lyaps_MF = net.solve_iteratively_mf(subkey, 
                                         sigmas_MF, 
                                         sigmas_mu_MF, 
                                         steps = steps_mf, 
                                         mc = False,
                                         use_tqdm = True)
print("DONE")
stats_all = {
    'Activity levels':
    {
        '$q$': q,
        '$q_m$': qm,
        '$q-q_m$': q-qm,
        '$q$ (theory)': qs_MF[:,1],
        '$q_m$ (theory)': qs_MF[:,0],
        '$q-q_m$ (theory)': qs_MF[:,1]-qs_MF[:,0],
    },
    'Lyapunov exponents': 
    {
        '$\\lambda_{max}$': lyap_stats_all['MLE'],
        '$\\lambda_{coherent}$ (MF)': lyaps_MF[:,0],
        '$\\lambda_{random}$ (MF)': lyaps_MF[:,1],
        #'theory (bound)': lyaps_upper_bound_MF
    },
    'Dimensionality':
    {
     '': 0*gtab - 1, # just to shift the colors
    'PR': dpr_all/(N0*P),
    'KY': lyap_stats_all['KY dimension']/(N0*P),
    #'Slow components': lyap_stats_all['# of slow']
    }#,
    #'Kolmogorov-Sinai entropy' : lyap_stats_all['KS entropy']
}
print(dpr_all)
print(dpr_all/(N0*P) )
# find zeroes (transitions)
def find_first_zero(X, Y):
    for i in range(len(X)-1):
        if Y[i]*Y[i+1] < 0:
            return (X[i]+X[i+1])/2
    return None

coherent_transition = find_first_zero(gtab, lyaps_MF[:,0])
random_transition   = find_first_zero(gtab, lyaps_MF[:,1])

plot_nested_stats(stats_all, 
                  gtab, 
                  mu, 
                f'N0_{N0}-P_{P}-sigma_{sigma:.2f}-mu_{mu}-steps_sim_{steps_sim}-steps_lyap_{steps_lyap}- symm_mu_{symmetric_mu}-symm_random_{symmetric_random}-th_{thresholded}_seed_{seed}',
                  limits = {'Lyapunov exponents': 
                            {'xlim': (None, None), 'ylim': (min_lyap_plot, max_lyap_plot)},
                            'Activity levels': 
                            {'xlim': (None, None), 'ylim': (None, 1)},
                            'Dimensionality':
                            {'xlim': (None, None), 'ylim': (-0.01, 0.45)},
                           },
                  fontsize=fontsize,
                  vertical_lines=[
                                  {'location': coherent_transition,
                                   'style': '-'},
                                  {'location': random_transition,
                                   'style': '--'},
                                 ],
                  horizontal=False,
                  legend=legend
                  #, ylogscale = ['Attractor dimensions',]
                 )