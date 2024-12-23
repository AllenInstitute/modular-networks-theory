import jax
import numpy as np
import jax.numpy as jnp

from .rnn_base import RNN

class RNN2L(RNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def mf_max_lyapunov_exponents(self, 
                                  sigma, 
                                  sigma_mu, 
                                  q,
                                  qm):
        '''
        L=2;
        Lyapunov exponents from sigmas and qs.
        '''
        s2 = sigma**2
        sm2 = sigma_mu**2
        A = jnp.pi*(s2*q + sm2*qm)
        B = jnp.pi**2*s2*q*(s2*q + 2*sm2*qm)/4
        
        random = 0.5*jnp.log( s2/jnp.sqrt(1 + A) )
        coherent = 0.5*jnp.log( sm2/jnp.sqrt(1 + A + B) )
        
        return random, coherent
    
    def population_activities(self, Xall, N0, P):
        #Mall = [jnp.mean(Xall[:, i*N0:(i+1)*N0], axis=1) for i in range(P)]
        #return jnp.stack(Mall, axis=1)
        Mall = [jnp.mean(Xall[..., i*N0:(i+1)*N0], axis=-1) for i in range(P)]
        return jnp.stack(Mall).T
        
    def qs_evolution(self, Xall, Mall):
        '''
        Given neural and population activity trajectories, 
        calculate evolution of q, qm, and m.
        '''
        return jnp.mean(Xall**2, axis=1), jnp.mean(Mall**2, axis=1), jnp.mean(Mall, axis=1)
    
    def steady_state_stats(self, q_all, qm_all, m_all, skip_init_steps=0):
        q_all, qm_all, m_all = q_all[skip_init_steps:], qm_all[skip_init_steps:], m_all[skip_init_steps:], 
        return q_all.mean(), qm_all.mean(), m_all.mean()

    def generate_weights(self, 
                         key, 
                         N0=10, 
                         P=10, 
                         sigma=1, 
                         sigma_mu=0,
                         symmetric_random=False,
                         symmetric_mu=False):
        N = N0*P
        key, subkey = jax.random.split(key)
        M = sigma_mu*jax.random.normal(subkey, 
                                       shape=(P,P), 
                                       dtype=self.precision)/jnp.sqrt(P)
        if symmetric_mu:
            M = (M + M.T)/jnp.sqrt(2)
        
        W = sigma*jax.random.normal(key, 
                                    shape=(N,N), 
                                    dtype=self.precision)/jnp.sqrt(N)
        if symmetric_random:
            W = (W + W.T)/jnp.sqrt(2)
            
        O = jnp.ones((N0, N0),
                     dtype=self.precision)/N0
        W = jnp.kron(M, O) + W
        return W
        
    def solve_iteratively_mf(self, key, sigmas, sigmas_mu, steps = 100, mc = False, use_tqdm=False):
        qs_all = []
        R2s_all = []
        if use_tqdm:
            from tqdm import tqdm
            G = tqdm(zip(sigmas, sigmas_mu))
        else:
            G = zip(sigmas, sigmas_mu)
            
        for sigma, sigma_mu in G:
            key, subkey = jax.random.split(key)
            qs, R2s = self.mf(subkey, (sigma_mu, sigma), mc=mc, steps=steps)
            qs_all.append(qs)
            R2s_all.append(R2s)

        R2s_all = jnp.array(R2s_all, dtype=self.precision)
        return jnp.array(qs_all), jnp.log( R2s_all )/2

    def mf_from_qs(self, q, qm, eps = 1e-12):
        A1 = (jnp.tan(jnp.pi*(qm+1)/4))**2
        A2 = (jnp.tan(jnp.pi*(q+1)/4))**2
        
        s_mu_general = jnp.sqrt( (A1-1)*(A2+1)/((1+A1)*jnp.pi*(qm+eps)) )
        s_mu_0 = jnp.sqrt( (1+A2)/2 )
        
        sigma_mu = s_mu_general*(qm>0) + (s_mu_0)*(qm==0) 
        sigma = jnp.sqrt( 2*(A2-A1)/((1+A1)*jnp.pi*(q+eps)) )
        
        C = 1 + jnp.pi*sigma**2*q/2
        D = jnp.pi*sigma_mu**2*qm
        lyap_coherent = jnp.log( sigma_mu**2/jnp.sqrt(C*(C+D)) )/2
        lyap_random = jnp.log(sigma**2/jnp.sqrt(A2))/2
        
        return sigma, sigma_mu, lyap_random, lyap_coherent
    
    def mf_from_qs2(self, q, qm, eps = 1e-12):
        ''' Untested! '''
        qm = qm + eps
        q = q + eps
        
        g1 = 2/(jnp.pi*qm)
        g2 = 2/(jnp.pi*q)
        s1 = jnp.sin(jnp.pi*qm/2)
        s2 = jnp.sin(jnp.pi*q/2)
        c1 = jnp.cos(jnp.pi*qm/2)
        c2 = jnp.cos(jnp.pi*q/2)
        t1 = jnp.tan(jnp.pi*qm/2)
        
        sigma_mu = jnp.sqrt( g1*s1/(1-s2) )
        sigma = jnp.sqrt( g2*(s2-s1)/(1-s2) )
        
        lyap_coherent = jnp.log( g1*t1 )/2
        lyap_random = jnp.log( g2*(s2-s1)/c2 )/2
        
        return sigma, sigma_mu, lyap_random, lyap_coherent