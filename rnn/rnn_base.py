import jax
import numpy as np
import jax.numpy as jnp

from tqdm import trange

class RNN():
    def __init__(self, thresholded = False, precision = jnp.float64):
        self.precision = precision
        if thresholded:
            self.phi = self.phi_thresholded
        else:
            self.phi = self.phi_symmetric
        f = lambda z: self.phi(z)
        self.dphi = jax.vmap(jax.grad(f)) # elementwise derivative of the activation function 

    def phi_symmetric(self, x):
        return jax.lax.erf(jnp.sqrt(jnp.pi)*x/2)
    
    def phi_thresholded(self, x):
       #return jax.lax.erf(jax.nn.relu(jnp.sqrt(jnp.pi)*x-1))
        return jax.lax.erf(jax.nn.relu(jnp.sqrt(jnp.pi)*x))
    
    def step(self, W, x):
        return self.phi(W@x)
    
    def evolve(self, W, x, steps, save_trajectory=False):
        if save_trajectory:
            Xall = []
            Xall.append(x)
            
        for i in range(steps):
            x = self.step(W, x)
            if save_trajectory:
                Xall.append(x)
        
        if save_trajectory:
            return x, jnp.stack(Xall)
        else:
            return x
        
    def jacobian_auto(self, W, x): # A bit slower? But should work also with more complex steps
        return jax.jacfwd(lambda z: self.step(W, z))(x)
    
    def jacobian(self, W, x): 
        return (W.T*self.dphi(W@x)).T     # multiply each row of W by the corresponding element of f(W@x)
##########################################################################
## Weights
##########################################################################   
    def from_lists_to_J(self, Zs, sigmas, Ps):
        J = 0
        L = len(Ps)
        for i in range(L):
            P = Ps[i]
            O = jnp.ones((P, P),dtype=self.precision)/P
            Z = Zs[i]
            J = jnp.kron(J, O) + sigmas[i]*Z
        return J
    def generate_weights(self,
                          key, 
                          P=2,        # number of subpopulations within each population
                          L=11,       # number of levels
                          Ps = None,  # list of Ps. If specified, it takes priority over P and L.
                          sigma1=1.0, # standard deviation (deepest level)
                          sigmaL=1.0, # standard deviation (outer/most shallow level)
                         ):
        key, subkey = jax.random.split(key)
        if Ps is not None:
            L = len(Ps)
        sigmas = jnp.linspace(sigma1, sigmaL, num=L)

        O = jnp.ones((P, P),dtype=self.precision)/P
        J = 0 
        n = 1
        for i in range(L):
            if Ps is not None:
                P = Ps[i]
                O = jnp.ones((P, P),dtype=self.precision)/P
            n = n*P
            Z = jax.random.normal(subkey, shape=(n,n), dtype=self.precision)/jnp.sqrt(n)
            J = jnp.kron(J, O) + sigmas[i]*Z
        return J
    def generate_weights_split(self,
                              key, 
                              P=2,        # number of subpopulations within each population
                              L=11,       # number of levels
                              Ps = None,  # list of Ps. If specified, it takes priority over P and L.
                              sigma1=1.0, # standard deviation (deepest level)
                              sigmaL=1.0, # standard deviation (outer/most shallow level)
                             ):
        key, subkey = jax.random.split(key)
        if Ps is not None:
            L = len(Ps)
        sigmas = jnp.linspace(sigma1, sigmaL, num=L)
        Zs = []
        
        O = jnp.ones((P, P),dtype=self.precision)/P
        J = 0 
        n = 1
        for i in range(L):
            if Ps is not None:
                P = Ps[i]
                O = jnp.ones((P, P),dtype=self.precision)/P
            n = n*P
            Z = jax.random.normal(subkey, shape=(n,n), dtype=self.precision)/jnp.sqrt(n)
            Zs.append(Z)
            J = jnp.kron(J, O) + sigmas[i]*Z
        if Ps is None:
            Ps = jnp.array( [P,]*L )
        return J, Zs, sigmas, Ps
##########################################################################
## Analysis (simulations)
##########################################################################   
    def calculate_qs_vs(self, X, Ps):
        qs = []
        new_shape = X[...,0].shape + Ps  # reshape into a convenient tensor representation
        M = jnp.reshape(X, new_shape)
        for P in Ps[::-1]:
            qs.append( (M**2).mean())
            M = jnp.mean(M, axis=-1)
        qs.append( (M**2).mean()) # should be zero at the end if phi odd and no bias
        qs = jnp.array(qs, dtype=self.precision)[::-1]
        return qs, jnp.diff(qs)
    
    def participation_ratio_dimension(self, Xall, skip_init_steps=0, division_eps=1e-12):
        #eigvals, _ = jnp.linalg.eigh(jnp.cov(Xall[skip_init_steps:,:], rowvar=False))
        #return (eigvals.sum())**2/( (eigvals**2).sum() + division_eps) 
        ## This is faster:
        C = jnp.cov(Xall, rowvar=False)
        return jnp.trace(C)**2/( (C*C).sum() + division_eps )

    def jacobian_eigvals_multistep(self, W, Xall, steps = 5): 
        ''' Calculate eigenvalues of Jacobians at last `steps` time steps. '''
        Eigvals = []
        
        for i in range(steps):
            J = self.jacobian(W, Xall[-steps+i])
            vals = jnp.linalg.eigvals(J)
            Eigvals.append( vals )
        
        return Eigvals
    
    def lyapunov_exponents(self, W, Xall, steps = 5):  
        X = Xall[-steps:]
        Q = jnp.identity(W.shape[0], dtype=self.precision)
        lambdas = jnp.zeros(W.shape[0], dtype=self.precision)
        
        for x in X:
            J = self.jacobian(W, x)
            Q = J@Q
            Q, R = jnp.linalg.qr(Q)
            lambdas = lambdas + jnp.log( jnp.abs(R.diagonal()) )/steps
            
        return lambdas

    def lyapunov_summary_stats(self,
                               lyap_exps, 
                               sort=True,
                               stats = ('MLE', '2MLE', '3MLE', 'KY dimension', 'KS entropy', '# of LEs > 0'),
                               eps_slow = 0.1,
                               verbose = False,
                               vmappable = True
                              ):
        results = {}

        if sort:
            lyap_exps = jnp.flip(jnp.sort(lyap_exps))
        if verbose:
            print(lyap_exps)

        if 'MLE' in stats:
            results['MLE'] = lyap_exps[0]
        if '2MLE' in stats:
            results['2MLE'] = lyap_exps[1]
        if '3MLE' in stats:
            results['3MLE'] = lyap_exps[2]
        if '# of slow' in stats:
            results['# of slow'] = (jnp.abs(lyap_exps) < eps_slow).sum()
        if 'KS entropy' in stats:
            results['KS entropy'] = ( (lyap_exps>0)*lyap_exps ).sum()
        if '# of LEs > 0' in stats:
            results ['# of LEs > 0'] = (lyap_exps>0).sum()
        if 'KY dimension' in stats:
            c_le = jnp.cumsum(lyap_exps)
            k = (c_le >= 0).sum()
            if verbose:
                print(c_le)
                print(k)
            if vmappable:
                j = (k-1)*(k>0) # if k==0, j=0, not -1
                if_chaotic = k>0
                # this assumes k<len(lyap_exps):
                results['KY dimension'] = k + if_chaotic*c_le[j]/jnp.abs(lyap_exps[k]) 
            else:
                if k > 0 and k < len(lyap_exps):
                    results['KY dimension'] = k + c_le[k-1]/jnp.abs(lyap_exps[k])
                else:
                    results['KY dimension'] = jnp.nan # ?
        return results
##########################################################################
## Mean field
##########################################################################
    def mf(self, 
            key, 
            sigmas,
            steps=100,
            samples=int(1e6),
            verbose=False,
            keep_history=False,
            use_trange=False,
            mc = True
              ):
        ''' 
        Solve MF self-consistent equations with sampling based integration 
        or with a closed-form self-consistent formula. 
        Works properly only with the symmetric (odd) `erf` activation function.
        '''
        L = len(sigmas)
        key, subkey = jax.random.split(key)
        qs = jax.random.uniform(subkey, shape=(L,), dtype=self.precision) 
        sigmas2 = jnp.array( sigmas, dtype=self.precision )**2
        if keep_history:
            qs_history = []
            qs_history.append(qs.copy())
            
        I = trange(steps) if use_trange else range(steps)
        
        @jax.jit
        def single_step_update_mc(key_, q_):
            Z = jax.random.normal(key_, shape=(L,samples) )
            a2 = jnp.cumsum(sigmas2*q_)
            a2L = a2[-1]
            C = jnp.sqrt( a2/(1+jnp.pi*(a2L - a2)/2) ) # elementwise
            for k in range(L):
                Z = Z.at[k,:].set(Z[k,:]*C[k])    
            return jnp.mean( ( self.phi( Z ) )**2, axis=1)
        
        @jax.jit
        def single_step_update_closed_form(key_, q_):
            a2 = jnp.cumsum(sigmas2*q_)
            a2L = a2[-1]
            C2 = a2/(1+jnp.pi*(a2L - a2)/2)  # elementwise
            return 4*jnp.arctan(jnp.sqrt(jnp.pi*C2+1))/jnp.pi - 1
        
        single_step_update = single_step_update_mc if mc else single_step_update_closed_form
        
        for i in I:
            if verbose:
                print('qs first')
                print(qs)
            key, subkey = jax.random.split(key)
            qs = single_step_update(key, qs)
            if keep_history:
                qs_history.append(qs.copy())
            if verbose:
                print("sigmas2*qs")
                print(sigmas2*qs)
                print("a2")
                print(a2)
                print("C")
                print(C)
                print("Cb")
                print(Cb)
                print('qs last')
                print(qs)
        ## calculate R2s    
        a2 = jnp.cumsum(sigmas2*qs)
        a2L = a2[-1]
        b2affine = 1 + (a2L - a2)*jnp.pi/2
        
        R2s = sigmas2/jnp.sqrt(b2affine*(b2affine + jnp.pi*a2))
        if keep_history:
            return qs, R2s, qs_history
        return qs, R2s
    
    
    def mf_implicit(self,
                        qs
                       ):
        ''' Equations (21) and (22) in the paper. '''
        sin_qs = jnp.sin(jnp.pi*qs/2)
        diff_sin_qs = jnp.diff(sin_qs, prepend=0)
        C0 = 2*diff_sin_qs/(np.pi*qs)  # common factor
        sigma2s = C0/(1-sin_qs[-1])
        R2s     = C0/jnp.cos(jnp.pi*qs/2)
        return sigma2s, R2s