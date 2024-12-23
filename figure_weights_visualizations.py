import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from visualizations import plot_eigenvalues
from rnn import RNN2L

jax.config.update("jax_enable_x64", True)

results_dir = 'results'
sigma = 0.3
sigma_mu = 1.0
symmetric_mu = False
symmetric_random = False

net = RNN2L()
os.makedirs(results_dir, exist_ok=True)

# ===== WEIGHTS =====
N0 = 10
P = 10
key = jax.random.PRNGKey(1)
J = net.generate_weights(key, 
                    N0=N0, 
                    P=P, 
                    sigma=sigma, 
                    sigma_mu=sigma_mu,
                    symmetric_mu=symmetric_mu,
                    symmetric_random=symmetric_random)

plt.figure(figsize=(6,6))
plt.imshow(J, cmap='plasma');
plt.axis('off');
plt.savefig(os.path.join(results_dir, 'weights.png'), bbox_inches='tight')
plt.savefig(os.path.join(results_dir, 'weights.pdf'), bbox_inches='tight')

# ===== EIGENVALUES =====
N0 = 5
P = 200
key = jax.random.PRNGKey(2)
J = net.generate_weights(key, 
                    N0=N0, 
                    P=P, 
                    sigma=sigma, 
                    sigma_mu=sigma_mu,
                    symmetric_mu=symmetric_mu,
                    symmetric_random=symmetric_random)

plot_eigenvalues(J, title=None, color='black')

plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'eigenvalues.pdf'), bbox_inches='tight');