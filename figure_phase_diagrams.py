import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rnn import RNN2L
jax.config.update("jax_enable_x64", True)

fontsize = 22
plt.rcParams.update({'font.size': fontsize})

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

key = jax.random.PRNGKey(1)   
net = RNN2L()

color_by_array = ['max_lyap', 'max_lyap_coherent', 'max_lyap_random']#, 'qm']

show_colorbar = True
lw = 2
sigma_max = 10
sigma_mu_max = 10
q_max = 0.97
Nq =  100
qm = 0    # transition line

q_constant = 0.9

def find_zero_lyap(net, q, qm_min = 0, qm_max = 1, eps_q=1e-4, eps_lyap=1e-8, steps=100, random=True):
    if qm_max == 1:
        qm_max = q - eps_q
    
    qm = (qm_min + qm_max)/2
    
    sigma, sigma_mu, lyap_r, lyap_c = net.mf_from_qs(q, qm)
    lyap = lyap_r if random else lyap_c
    
    if np.abs(lyap) < eps_lyap or steps<=0:
        return sigma, sigma_mu, qm, steps, lyap
    
    if (random and lyap < 0) or ( not random and lyap > 0): 
        return find_zero_lyap(net, q, qm_min=qm_min, qm_max=qm, eps_q=eps_q, eps_lyap=eps_lyap, random=random, steps=steps-1)
    
    return find_zero_lyap(net, q, qm_min=qm, qm_max=qm_max, eps_q=eps_q, eps_lyap=eps_lyap, random=random, steps=steps-1)

def lyap_max_mesh(net, 
                  q_max = 0.9, 
                  q_min = 0.01,
                  Nq = 1000, 
                  Nqm = 500, 
                  N_smu = 100, 
                  N_sigmas=50, 
                  min_sigma=0.1,
                  eps_q = 1e-4,
                  subspace = 'max' 
                 ):
    q_tab = jnp.linspace(q_min, q_max, num=Nq, dtype=jnp.float64)
    sigmas_all = []
    sigmas_mu_all = []
    max_lyaps_all = []
    
    sigmas = jnp.linspace(min_sigma, 1, N_sigmas) 
    for s in sigmas:
        sigmas_array = 0*sigmas + s
        sigmas_mu_array = sigmas
        if subspace == 'max':
            max_lyaps = jnp.log(jnp.maximum(sigmas_array, sigmas_mu_array))
        elif subspace == 'random':
            max_lyaps = jnp.log(sigmas_array)
        elif subspace == 'coherent':
            max_lyaps = jnp.log(sigmas_mu_array)
        else:
            raise Exception("Unknown subspace...")
            
        sigmas_all.append(sigmas_array)
        sigmas_mu_all.append(sigmas_mu_array)
        max_lyaps_all.append(max_lyaps)
        
    for q in q_tab:
        qm_array = jnp.linspace(0, q-eps_q, num=Nqm)
        q_array = q + 0*qm_array
        sigmas, sigmas_mu, lyap_random, lyap_coherent = net.mf_from_qs(q_array, qm_array)
        sigmas_all.append(sigmas)
        sigmas_mu_all.append(sigmas_mu)
        if subspace == 'max':
            max_lyaps_all.append(jnp.maximum(lyap_random, lyap_coherent))
        elif subspace == 'random':
            max_lyaps_all.append(lyap_random)
        else:
            max_lyaps_all.append(lyap_coherent)
        # bottom right corner
        new_sigmas = jnp.zeros(N_smu, dtype=jnp.float64) + sigmas[0]
        new_sigmas_mu = jnp.linspace(0, sigmas_mu[0], num=N_smu, dtype=jnp.float64)
        if subspace == 'coherent':
            new_lyaps = lyap_coherent[0]  - jnp.log(sigmas_mu[0]) + jnp.log(new_sigmas_mu + 1e-6)
        else:
            new_lyaps = jnp.zeros(N_smu, dtype=jnp.float64) + lyap_random[0]
        sigmas_all.append(new_sigmas)
        sigmas_mu_all.append(new_sigmas_mu)
        max_lyaps_all.append(new_lyaps)
    
    return np.concatenate(sigmas_all), np.concatenate(sigmas_mu_all), np.concatenate(max_lyaps_all)

def qm_mesh(net, 
          q_max = 0.9,  
          q_min = 0.1,
          Nq = 1000, 
          Nqm = 500, 
          N_smu = 100, 
          N_sigmas=50, 
          min_sigma=0.1,
          eps_q = 1e-4):
    q_tab = jnp.linspace(q_min, q_max, num=Nq, dtype=jnp.float64)
    sigmas_all = []
    sigmas_mu_all = []
    qm_all = []
    
    sigmas = jnp.linspace(min_sigma, 1, N_sigmas) 
    for s in sigmas:
        sigmas_array = 0*sigmas + s
        sigmas_mu_array = sigmas
        qm_array = 0*sigmas
        sigmas_all.append(sigmas_array)
        sigmas_mu_all.append(sigmas_mu_array)
        qm_all.append(qm_array)
        
    for q in q_tab:
        qm_array = jnp.linspace(0, q-eps_q, num=Nqm)
        q_array = q + 0*qm_array
        sigmas, sigmas_mu, _, _ = net.mf_from_qs(q_array, qm_array)
        sigmas_all.append(sigmas)
        sigmas_mu_all.append(sigmas_mu)
        qm_all.append(qm_array)
        # bottom right corner
        new_sigmas = jnp.zeros(N_smu, dtype=jnp.float64) + sigmas[0]
        new_sigmas_mu = jnp.linspace(0, sigmas_mu[0], num=N_smu, dtype=jnp.float64)
        new_qm = jnp.zeros(N_smu, dtype=jnp.float64) + qm_array[0]
        sigmas_all.append(new_sigmas)
        sigmas_mu_all.append(new_sigmas_mu)
        qm_all.append(new_qm)
    
    return np.concatenate(sigmas_all), np.concatenate(sigmas_mu_all), np.concatenate(qm_all)

for color_by in color_by_array:
    print("Working on: " + color_by)
    
    plt.figure(figsize=(5+1*show_colorbar,4.5))

    q_tab = jnp.linspace(0, q_max, num=Nq, dtype=jnp.float64)
    qm_tab = qm + 0*q_tab
    sigmas, sigmas_mu, lyap_random, lyap_coherent = net.mf_from_qs(q_tab, qm_tab)
    if color_by == 'max_lyap' or color_by == 'max_lyap_coherent':
        plt.plot(sigmas, sigmas_mu, 'k-', lw=lw, label='$\lambda_{coherent}=0$'); # transition to non-zero qm

    sigmas_HD_EoC = []
    sigmas_mu_HD_EoC = []
    sigmas_LD_EoC = []
    sigmas_mu_LD_EoC = []

    qm_lambda_coh0 = []
    q_lambda_coh0 = []

    for q in q_tab:
        if q<0.01:
            continue
        sigma, sigma_mu, _, _, _ = find_zero_lyap(net, q)
        sigmas_HD_EoC.append(sigma)
        sigmas_mu_HD_EoC.append(sigma_mu)

    if color_by == 'max_lyap' or color_by == 'max_lyap_random':
        plt.plot(sigmas_HD_EoC, sigmas_mu_HD_EoC, 'k--', lw=lw, label='$\lambda_{random}=0$')
        plt.plot([1, 1], [0, 1], 'k--', lw=lw)
    
    plt.xlabel('$\sigma$')
    plt.ylabel('$\sigma_{\mu}$');
    plt.xlim([0, sigma_max])
    plt.ylim([0, sigma_mu_max]);

    cmap = 'Oranges' if color_by=='qm' else 'BrBG'
    vmax = 1.0 if color_by=='qm' else 1.0
    vmin = 0.0 if color_by=='qm' else -1.0

    if color_by == 'max_lyap':
        s, s_mu, mle = lyap_max_mesh(net, q_max=q_max)
    elif color_by == 'max_lyap_random':
        s, s_mu, mle = lyap_max_mesh(net, q_max=q_max, subspace='random')
    elif color_by == 'max_lyap_coherent':
        s, s_mu, mle = lyap_max_mesh(net, q_max=q_max, subspace='coherent')
    elif color_by == 'qm':
        s, s_mu, mle = qm_mesh(net, q_max=q_max)

    sc = plt.scatter(s, s_mu, c=mle, cmap=cmap, vmin=vmin, vmax=vmax, s=16);

    if show_colorbar:
        texts = {'max_lyap': '$\lambda_{max}$',
             'max_lyap_random': '$\lambda_{random}$',
             'max_lyap_coherent': '$\lambda_{coherent}$',
             'qm': '$q_m$'
            }
        clb=plt.colorbar(sc, label=texts[color_by])

    plt.xticks( [0, 2, 4, 6, 8, 10] )
    plt.yticks( [0, 2, 4, 6, 8, 10] )
    
    if color_by == 'max_lyap':
        c = 'purple'
        plt.text(2.8, 1.3, '$\\mu$', color=c)
        plt.text(3.9, 8, '$\\mu + M$', color=c)
        plt.text(1.3, 6, '$M$', color=c)        
    else:
        plt.legend(loc='lower right');

    plt.savefig(os.path.join(results_dir, f'phase_diagram-{color_by}.png'), bbox_inches='tight', dpi=1200)
    plt.close()