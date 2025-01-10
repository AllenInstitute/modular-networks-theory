import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
N0_matrix = 8
P_matrix  = 8

N0 = 100
P = 200
steps_init = 200
steps_sim = 20

#sigma_all = [0.5, 5.0, 5.0, 1.0]
#sigma_mu_all = [0.5, 0.5, 8.0, 5.0]
sigma_all = [0.5, 4.0, 4.0, 1.0]
sigma_mu_all = [0.5, 0.5, 6.0, 5.0]

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

fig, axs = plt.subplots(4, len(sigma_all), figsize=(len(sigma_all)*5, 17), constrained_layout=True)
i = 0
#gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1.5, 1])

def custom_axis_off(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

for sigma, sigma_mu, label in tqdm( zip(sigma_all, sigma_mu_all, labels) ): 
    axs[0,i].set_title(label)
    # Keep the same seed in each column for consistent phase transition [?]
    key = jax.random.PRNGKey(seed) 
    ### Matrix visualization
    # Generate weights
    key, subkey = jax.random.split(key)
    J = net.generate_weights(subkey, 
                            N0=N0_matrix, 
                            P=P_matrix, 
                            sigma=sigma, 
                            sigma_mu=sigma_mu)
    # Visualize weights
    axs[0,i].imshow(J, cmap='plasma', vmin=-1.5, vmax=1.5)
    axs[0,i].axis('off');
    
    ### Eigenvalues
    # Generate weights
    #key, subkey = jax.random.split(key)
    J = net.generate_weights(subkey, 
                            N0=N0_eigvals, 
                            P=P_eigvals, 
                            sigma=sigma, 
                            sigma_mu=sigma_mu)
    # Plot eigenvalues
    plot_eigenvalues(J, color='black', color_circle='red', title=None, ax=axs[1,i])
    axs[1,i].axis('off')
    #axs[1,i].set_position(axs[1,i].get_position())

    #custom_axis_off(axs[1,i])     
    ### Activities and perturbations
    # Generate weights
    key, subkey = jax.random.split(key)
    J = net.generate_weights(subkey, 
                            N0=N0, 
                            P=P, 
                            sigma=sigma, 
                            sigma_mu=sigma_mu)
    ## Activities
    N = J.shape[0]
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, shape = (N,), dtype=jnp.float64)
    ## Equilibrate
    x = net.evolve(J, x0, steps_init, save_trajectory=False)
    
    ## Evolve the original replica:
    _, Xall = net.evolve(J, x, steps_sim, save_trajectory=True)
    # Plot activities of individual neurons
    axs[2,i].plot(Xall[:,:N_plot], lw=lw, alpha=0.7)
    axs[2,i].set_ylim(ylim)
    # Plot activities of populations
    Mall = net.population_activities(Xall, N0, P)
    axs[3,i].plot(Mall[:,:N_plot], lw=lw, alpha=0.7)
    axs[3,i].set_ylim(ylim)
    #if i>0:
    #    axs[2,i].set_yticklabels([])
    #    axs[3,i].set_yticklabels([])
    
    i += 1

#axs[0,0].set_ylabel('$J$')    
#axs[1,0].set_ylabel('Eigenvalues of $J$')
axs[2,0].set_ylabel('$x^{1}_i$')    
axs[3,0].set_ylabel('$m^{\\alpha}$')

for i in range(4):
    axs[3, i].set_xlabel('$t$')
#fig.supxlabel('$t$')

plt.savefig(os.path.join(results_dir, 'eig_and_activities.pdf'), bbox_inches='tight')