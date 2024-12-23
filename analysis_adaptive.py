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
from visualizations import plot_nested_stats_pairs, plot_trajectory, plot_eigenvalues
from utils import ld2da

def adaptation_analysis(key, 
                        net,
                        steps_lyap = 100, 
                        steps_sim = 200,
                        steps_adapt = 2_000,
                        steps_init = 100,      
                        Ps = [500,],
                        eta = 0.2,
                        qL=0.8,
                        analyze_initial=False,
                        verbose=True,
                        figures=True,
                        return_weights=False,
                        sigma1=0.0,
                        sigmaL=5.0,
                        fontsize=16,
                        results_dir='results/adaptation',
                        fname_prefix='',
                        calculate_lyap=True,
                        save_pickle=True,
                        stats_names=('MLE', 'KY dimension'),
                        include_q0=False
                       ):
    
    plt.rcParams.update({'font.size': fontsize})
    os.makedirs(results_dir, exist_ok=True)
    fname_prefix += f"_L{len(Ps)}_"
    fpath_prefix=os.path.join(results_dir, fname_prefix)
    
    key, subkey = jax.random.split(key)
    L = len(Ps)
    desired_qs = jnp.linspace(qL/L, qL, num=L)
    predicted_sigma2s, predicted_R2s = net.mf_implicit(desired_qs)
    
    if verbose:
        print(f"Desired qs: {desired_qs}")
        print(f"Predicted sigmas: {np.sqrt(predicted_sigma2s)}")
    
    J, Zs, sigmas, Ps =  net.generate_weights_split(subkey, 
                                 Ps = Ps,
                                 sigma1 = sigma1,
                                 sigmaL = sigmaL)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(key, shape = (J.shape[0],), dtype=jnp.float64)
    
    if analyze_initial:
        x, Xall = net.evolve(J, x, steps_sim, save_trajectory=True)

        qs, vs = net.calculate_qs_vs(x, Ps)
        lyap_exps = net.lyapunov_exponents(J, Xall, steps=steps_lyap)    
        stats = net.lyapunov_summary_stats(lyap_exps, stats = stats_names, vmappable=True)
        if verbose:
            print("qs and vs before adaptation:")
            print(qs)
            print(vs)
            print(f"sigmas before adaptation: {sigmas}")
            print(stats)
        if figures:
            plot_trajectory(Xall, ends_lengths = (60, 30))
            plt.savefig(fpath_prefix+'x_pre.pdf', bbox_inches='tight')
            
    else: # equilibration phase to to avoid nonstandard initial conditions
        x = net.evolve(J, x, steps_init, save_trajectory=False)

    sigmas, sigmas_all, qs_all = net.evolve_adapt(Zs, sigmas, Ps, x, steps_adapt, eta=eta, qL=qL, include_q0=include_q0)

    J = net.from_lists_to_J(Zs, sigmas, Ps)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(key, shape = (J.shape[0],), dtype=jnp.float64)
    x, Xall = net.evolve(J, x, steps_sim, save_trajectory=True)
    qs, vs = net.calculate_qs_vs(Xall[int(steps_sim/2):,:], Ps)
    if calculate_lyap:
        lyap_exps = net.lyapunov_exponents(J, Xall, steps=steps_lyap)    
        stats = net.lyapunov_summary_stats(lyap_exps, stats = stats_names, vmappable=True)
    else:
        stats = None
    if verbose:
        print("qs and vs after adaptation:")
        print(qs)
        print(vs)
        print(f"sigmas after adaptation: {sigmas}")
        print(stats)

    if figures:
        colors = plt.cm.Dark2.colors
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    #    plot_eigenvalues(J, title='J eigenvalues')
        plot_trajectory(Xall, ends_lengths = (60, 30))
        plt.savefig(fpath_prefix+'x_after.pdf', bbox_inches='tight')
        
        plt.figure(figsize=(4.5,4))
        for i, s in enumerate(np.array(sigmas_all).T):
            plt.plot(s, label=f"$\\sigma_{i+1}$")
        for y in np.sqrt(predicted_sigma2s):
            print(f"plotting a horizontal line at {y:.2f}")
            plt.axhline(y, color='black', alpha=0.5, ls='--', lw=2)
        plt.xlabel('step #')
        plt.ylabel('Control parameters');
        plt.legend()
        plt.savefig(fpath_prefix+'sigmas.pdf', bbox_inches='tight')
        
        plt.figure(figsize=(4.5,4))
        for i, q in enumerate(np.array(qs_all).T):
            k = i if include_q0 else i+1
            plt.plot(q, label=f"$q_{k}$")
            
        for y in desired_qs:
            plt.axhline(y, color='black', alpha=0.5, ls='--', lw=2)
            #plt.axhline(y, alpha=0.8, ls='--', lw=2)
        plt.xlabel('step #')
        plt.ylabel('Order parameters');
        plt.legend()
        plt.savefig(fpath_prefix+'qs.pdf', bbox_inches='tight')
    if save_pickle:
        data_dict = {'Ps': Ps,
                     'steps_lyap': steps_lyap, 
                     'steps_sim': steps_sim,
                     'steps_adapt': steps_adapt,
                     'steps_init': steps_init,      
                     'eta': eta,
                     'qL': qL,
                     'qs_all': qs_all,
                     'sigmas_all': sigmas_all,
                     'stats': stats
                    }
        with open(fpath_prefix+'.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            
    if return_weights:
        return stats, qs, sigmas, J
    else:
        return stats, qs, sigmas
    
if __name__ == '__main__':
    '''CURRENTLY, THIS IS NOT USED ANYMORE!'''
    jax.config.update("jax_enable_x64", True)

    ####################################################
    parser = argparse.ArgumentParser(
                        prog='Lyapunov RNN analyzer (adaptive neural net)',
                        description='Adapt and simulate an RNN in order to compute Lyapunov exponents and their statistics',
                        epilog='Finished')
    parser.add_argument('-e', '--seed', default=np.random.randint(1e6) )
    parser.add_argument('-d', '--debug', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-m', '--mf', default=True, action=argparse.BooleanOptionalAction)
    args  = parser.parse_args()
    seed  = int(args.seed)
    debug = args.debug
    mf    = args.mf
    ###################################################

    if debug:
        qtab    = np.linspace(0.3, 0.9, 20)
        P_seqs  = ( 
                    (1_000,),
                    (32,32)
                  )
        steps_lyap = 10
        steps_sim = 20
        steps_adapt = 10
        samples_mf = int(1e7)
        steps_mf   = 100
        eta = 0.8
    else:
#        qtab    = np.linspace(0.2, 0.95, 50)
        qtab    = np.linspace(0.3, 0.9, 50)
        P_seqs  = ( 
                    (10_000,),
                    (100,100),
                    #(50,20,10),
                    (22,22,22),
                    #(10,20,50),
                  )
        steps_lyap = 200
        steps_sim = 700
        steps_adapt = 3_000
        samples_mf = int(1e7)
        steps_mf   = 300
        eta = 0.3
            
    key = jax.random.PRNGKey(seed)
    net = RNNAdaptive()
    
    stats_all = {}
    stats_all2 = {}
    
    stats_all2['$q_L$'] = {}
    
    for Ps in P_seqs:
        start = time.time()
        L = len(Ps)
        print(f"Levels: {L}")
        label = f"L = {L}; P = {Ps}"
        qLtab_real = []
        qLtab_mf = []
        lyap_mf  = []
        sigmaL = []
        stats_arrays = []
        for qL in tqdm(qtab):
            key, subkey = jax.random.split(key)
            stats, qs, sigmas = adaptation_analysis(subkey, net, verbose=False, figures=False, 
                                           Ps=Ps, qL=qL, eta=eta, steps_lyap=steps_lyap, steps_sim=steps_sim, steps_adapt=steps_adapt)
            stats_arrays.append(stats)
            qLtab_real.append(qs[-1])
            sigmaL.append(sigmas[-1])
            if mf:
                key, subkey = jax.random.split(key)
                qs_mf, R2mf = net.mf(subkey, sigmas, samples=samples_mf, steps=steps_mf)
                qLtab_mf.append(qs_mf[-1])
                lyap_mf.append( jnp.log(jnp.max(R2mf))/2 )
                
        stats_arrays = ld2da(stats_arrays)
        for k, value in stats_arrays.items():
            if k not in stats_all:
                stats_all[k] = {}
                stats_all2[k] = {}
            stats_all[k][label] = (qLtab_real, value)
            stats_all2[k][label] = (sigmaL, value)

        stats_all2['$q_L$'][label]  = (sigmaL, qLtab_real)
        if mf:
            stats_all2['$q_L$']['MF: '+label]  = (sigmaL, qLtab_mf)
            stats_all2['MLE']['MF: '+label] = (sigmaL, lyap_mf)  
            stats_all['MLE']['MF: '+label] = (qLtab_mf, lyap_mf)   
    
        stop = time.time()
        print(f"Elapsed time: {stop-start:.2f}")
        
    plot_nested_stats_pairs(stats_all, 
              f'q_steps_sim_{steps_sim}-steps_lyap_{steps_lyap}-steps_adapt_{steps_adapt}-seed_{seed}',
              xlabel = '$q_L$'
             )
    plot_nested_stats_pairs(stats_all2, 
              f'sigma_steps_sim_{steps_sim}-steps_lyap_{steps_lyap}-steps_adapt_{steps_adapt}-seed_{seed}',
              xlabel = '$\sigma_L$'
             )