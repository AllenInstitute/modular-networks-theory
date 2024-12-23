import jax
import jax.numpy as jnp

from .rnn_base import RNN

class RNNAdaptive(RNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def evolve_adapt(self, Zs, sigmas, Ps, x, steps, 
                     #save_x_trajectory=False, # NOT IMPLEMENTED
                     save_sigma_trajectory=True,
                     save_q_trajectory=True,
                     qL = 0.8,
                     eta=1e-1,
                     include_q0=False):

        L = len(Ps)
        desired_vs = jnp.array( (qL/L,)*L, dtype=self.precision)  
        if save_sigma_trajectory:
            sigmas_all = [sigmas, ]
        if save_q_trajectory:
            qs_all = []
            
        @jax.jit
        def step(x, sigmas):
            J = self.from_lists_to_J(Zs, sigmas, Ps)
            x = self.step(J, x)
            qs, vs = self.calculate_qs_vs(x, Ps)
            return x, qs, sigmas + eta*(desired_vs - vs)
        
        for i in range(steps): # probably better to rewrite with scan/fori_loop
            x, qs, sigmas = step(x, sigmas)   
            if save_sigma_trajectory:
                sigmas_all.append(sigmas)
            if save_q_trajectory:
                start = 0 if include_q0 else 1
                qs_all.append(qs[start:])
                
        return_list = [sigmas, ]
        
        if save_sigma_trajectory:
            return_list.append(sigmas_all)
        if save_q_trajectory:
            return_list.append(qs_all)
            
        return return_list
    