import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rnn import RNNAdaptive
from analysis_adaptive import adaptation_analysis
jax.config.update("jax_enable_x64", True)

sigma10 = 0.0
sigmaL0 = 5.0
qL = 0.8
seeds = (0, 1, 2, 3, 4)
Ps = (100, 100)

net = RNNAdaptive()

for seed in seeds:
    print(f"Seed: {seed}")
    start = time.time()
    key = jax.random.PRNGKey(seed)   
    adaptation_analysis(key, net, Ps=Ps, qL=qL, sigma1=sigma10, sigmaL=sigmaL0, fname_prefix=f'seed{seed}', calculate_lyap=False)
    stop = time.time()
    print(f"Elapsed time: {stop-start:.2f}")
    
print("DONE")