import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tqdm import tqdm

from rnn import RNN2L
from visualizations import plot_eigenvalues

jax.config.update("jax_enable_x64", True)

plt.rcParams.update({'font.size': 40})

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

seed = 2
N0_eigvals = 5
P_eigvals  = 200
N0 = 100
P = 200
steps_init = 200
steps_sim = 20

sigma_all = [0.5, 5.0, 5.0, 1.0]
sigma_mu_all = [0.5, 0.5, 8.0, 5.0]

labels = ['quiescent', '$\\mu$', '$\\mu + M$', '$M$']
lw = 5
ylim = [-1.05, 1.05]
ylim_diff = [-2.05, 2.05]
N_plot = 5
colors = plt.cm.Dark2.colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    
net = RNN2L()
#key = jax.random.PRNGKey(seed)

def decompose(X):
    M = net.population_activities(X, N0, P)
    O = jnp.ones( (N0,) )
    Xm = jnp.kron(M, O)
    X = X - Xm
    return Xm, X

fig, axs = plt.subplots(3, len(sigma_all), figsize=(len(sigma_all)*5, 12), constrained_layout=True)
i = 0

for sigma, sigma_mu, label in tqdm( zip(sigma_all, sigma_mu_all, labels) ): 
    # Keep the same seed in each column for consistent phase transition [?]
    key = jax.random.PRNGKey(seed) 
    ### Eigenvalues
    # Generate weights
    key, subkey = jax.random.split(key)
    J = net.generate_weights(subkey, 
                            N0=N0_eigvals, 
                            P=P_eigvals, 
                            sigma=sigma, 
                            sigma_mu=sigma_mu)

    plot_eigenvalues(J, color='black', color_circle='red', title=None, ax=axs[0,i])
    axs[0,i].axis('off')
    axs[0,i].set_title(label)
    
    ### Activities and perturbations
    # Generate weights
    key, subkey = jax.random.split(key)
    J = net.generate_weights(subkey, 
                            N0=N0, 
                            P=P, 
                            sigma=sigma, 
                            sigma_mu=sigma_mu)
    ### Activities
    N = J.shape[0]
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, shape = (N,), dtype=jnp.float64)
    ## Equilibrate
    x = net.evolve(J, x0, steps_init, save_trajectory=False)
    
    ## Evolve the original replica:
    _, Xall = net.evolve(J, x, steps_sim, save_trajectory=True)
    # Plot activities of individual neurons
    axs[1,i].plot(Xall[:,:N_plot], lw=lw, alpha=0.7)
    axs[1,i].set_ylim(ylim)
    # Plot activities of populations
    Mall = net.population_activities(Xall, N0, P)
    axs[2,i].plot(Mall[:,:N_plot], lw=lw, alpha=0.7)
    axs[2,i].set_ylim(ylim)
    
    i += 1
    
axs[1,0].set_ylabel('$x^{1}_i$')    
axs[2,0].set_ylabel('$m^{\\alpha}$')

fig.supxlabel('$t$')

plt.savefig(os.path.join(results_dir, 'eig_and_activities.pdf'), bbox_inches='tight')