import os

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from utils import ld2da
    
def plot_eigenvalues(W, color=None, color_circle='black', title='W eigenvalues', ax=None):
    W_eig = np.linalg.eig(W)
    W_eig_real = np.real(W_eig[0])
    W_eig_imag = np.imag(W_eig[0])

    if ax is None:
        plt.figure(figsize=(5,5))
        ax = plt.gca()
        
    ax.set_title(title)
    ax.scatter(W_eig_real, W_eig_imag, c=color);

    phi = np.linspace(0, 2*np.pi, 1000)
    ax.plot(np.cos(phi), np.sin(phi), '--', color=color_circle);
    ax.set_xlabel('$Re(\lambda)$');
    ax.set_ylabel('$Im(\lambda)$');
    ax.axis('equal');
    
def plot_nested_stats(stats, gtab, mu, name, lw=4, ms=10, alpha=0.7, 
                      limits={}, ylogscale=[], xlabel = None, fontsize=16, 
                     vertical_lines=None, horizontal=True, legend=True):
    
    Nstats = len(stats)
    
    plt.rcParams.update({'font.size': fontsize})
    #plt.rcParams.update({'legend.labelspacing': 0.25})
    if horizontal:
        fig, axs = plt.subplots(1, Nstats, figsize=(Nstats*5.2, 4.5))
    else:
        fig, axs = plt.subplots(Nstats, 1, figsize=(7, Nstats*4.5))
    ind = 0
    for key, val in stats.items():
        #axs[ind].spines['right'].set_visible(False)
        #axs[ind].spines['top'].set_visible(False)
        #axs[ind].spines['left'].set_visible(True)
        #axs[ind].spines['bottom'].set_visible(True)
        axs[ind].set_ylabel(key)
        if vertical_lines is not None:
            for vl in vertical_lines:
                axs[ind].axvline(vl['location'], 
                                 linestyle=vl['style'],
                                 color='black',
                                 alpha=0.4,
                                 lw=lw)
        axs[ind].axhline(0.0, ls='--', color='gray')
        if isinstance(val, dict): # nested (multiple plots)
            if key in ylogscale:
                for subkey, subval in val.items():
                    if 'theory' in subkey:
                        axs[ind].semilogy(gtab, subval, 'k--', alpha=alpha, linewidth=lw, markersize=ms)
                    elif 'MF' in subkey:
                        axs[ind].semilogy(gtab, subval, '--', label=subkey, alpha=alpha, linewidth=lw, markersize=ms)
                    else:
                        print(subval)
                        axs[ind].semilogy(gtab, subval, '.', label=subkey, linewidth=lw, markersize=ms)
            else:
                for subkey, subval in val.items():
                    if 'theory' in subkey:
                        axs[ind].plot(gtab, subval, 'k--', alpha=alpha, linewidth=lw, markersize=ms)
                    elif 'MF' in subkey:
                        axs[ind].plot(gtab, subval, '--', label=subkey, alpha=alpha, linewidth=lw, markersize=ms)
                    else:
                        axs[ind].plot(gtab, subval, '.', label=subkey, linewidth=lw, markersize=ms)

            if legend:
                axs[ind].legend(fontsize=fontsize-4)
        else: # flat
            axs[ind].plot(gtab, val, 'k.-', linewidth=lw, markersize=ms)
        if key in limits:
            axs[ind].set_xlim(limits[key]['xlim'])
            axs[ind].set_ylim(limits[key]['ylim'])
        ind += 1

    if horizontal:
        for ax in axs:
            if xlabel is None:
                if mu:
                    ax.set_xlabel("$\sigma$")
                else:
                    ax.set_xlabel("$\sigma_{\mu}$")
            else:
                ax.set_xlabel(xlabel)
    else:
        if xlabel is None:
            if mu:
                axs[-1].set_xlabel("$\sigma$")
            else:
                axs[-1].set_xlabel("$\sigma_{\mu}$")
        else:
            axs[-1].set_xlabel(xlabel)

    plt.tight_layout()
    fname = f'{name}.pdf'
    fpath = os.path.join('results', fname)
    os.makedirs('results', exist_ok=True)
    plt.savefig(fpath, bbox_inches='tight')
    
def plot_nested_stats_pairs(stats, name, lw=2, ms=7, limits = {}, xlabel = None, results_path='results'):
    
    Nstats = len(stats)
    fig, axs = plt.subplots(1, Nstats, figsize=(Nstats*5, 4))
    
    ind = 0
    for key, val in stats.items():
        axs[ind].set_ylabel(key)
        axs[ind].set_xlabel(xlabel)
        axs[ind].axhline(0.0, ls='--', color='gray')
        if isinstance(val, dict): # nested (multiple plots)
            for subkey, subval in val.items():
                if subkey.startswith('MF'):
                    axs[ind].plot(subval[0], subval[1], 'k--', alpha=0.6, linewidth=lw, markersize=ms)
                else:
                    axs[ind].plot(subval[0], subval[1], '.-', label=subkey, linewidth=lw, markersize=ms)
            axs[ind].legend()
        else: # flat
            axs[ind].plot(val[0], val[1], 'k.-', linewidth=lw, markersize=ms)
        if key in limits:
            axs[ind].set_xlim(limits[key]['xlim'])
            axs[ind].set_ylim(limits[key]['ylim'])
        ind += 1
        
    plt.tight_layout()
    fname = f'stats-{name}.pdf'
    fpath = os.path.join(results_path, fname)
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(fpath, bbox_inches='tight')


def plot_trajectory(Xall, ends_lengths = (200, 50) ):
    qs = [(z**2).mean() for z in Xall]
    ms = [z.mean() for z in Xall]
    
    T = len(Xall)
    L = len(ends_lengths)
    
    fig, axs = plt.subplots(1, L+1, figsize=((L+1)*5,4))
    
    axs[0].plot(range(T), qs, label='q (mean sq. activity)')
    axs[0].plot(range(T), ms, label='m (mean activity)')
    axs[0].set_xlabel('step')
    axs[0].set_ylabel('activity stats')
    axs[0].legend()
    
    if L>0:
        for i, l in enumerate(ends_lengths):
            axs[i+1].plot(-l + np.arange(l), qs[-l:])
            axs[i+1].plot(-l + np.arange(l), ms[-l:])
            axs[i+1].set_xlabel('step')