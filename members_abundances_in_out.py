import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from astropy.table import Table, join
from os import chdir
from scipy.stats import norm as gauss_norm


def _prepare_pdf_data(means, stds, range, norm=True):
    x_vals = np.linspace(range[0], range[1], 250)
    y_vals = np.zeros_like(x_vals)
    # create and sum all PDF of stellar abundances
    for d_m, d_s in zip(means, stds):
        if np.isfinite([d_m, d_s]).all():
            y_vals += gauss_norm.pdf(x_vals, loc=d_m, scale=d_s)
    # return normalized summed pdf of all stars
    if norm and np.nansum(y_vals) > 0.:
        y_vals = 1. * y_vals/np.nanmax(y_vals)
    return x_vals, y_vals


def _prepare_hist_data(d, bins, range, norm=True):
    heights, edges = np.histogram(d, bins=bins, range=range)
    width = np.abs(edges[0] - edges[1])
    if norm:
        heights = 1.*heights / np.nanmax(heights)
    return edges[:-1], heights, width


simulation_dir = '/shared/data-camelot/cotar/'
data_dir_clusters = simulation_dir+'GaiaDR2_open_clusters_1907_GALAH_CGmembers/'

data_dir = '/shared/ebla/cotar/'
USE_DR3 = True

if USE_DR3:
    cannon_data = Table.read(data_dir+'GALAH_iDR3_main_alpha_190529.fits')
    fe_col = 'fe_h'
    teff_col = 'teff'
    suffix = '_DR3'
else:
    cannon_data = Table.read(data_dir+'sobject_iraf_iDR2_180325_cannon.fits')
    gaia_data = Table.read(data_dir+'sobject_iraf_53_gaia.fits')['source_id', 'sobject_id']
    cannon_data = join(cannon_data, gaia_data, keys='sobject_id', join_type='inner')
    fe_col = 'Fe_H_cannon'
    teff_col = 'Teff_cannon'
    suffix = '_DR2'

# cluster_dir = data_dir_clusters + 'Cluster_orbits_Gaia_DR2_/'

# detemine all posible simulation subdirs
chdir(data_dir_clusters)
for cluster_dir in glob('Cluster_orbits_Gaia_DR2_*'):
    chdir(cluster_dir)
    print 'Working on clusters in ' + cluster_dir

    for sub_dir in glob('*'):

        if '.png' in sub_dir:
            continue

        print sub_dir
        chdir(sub_dir)

        try:
            g_init = Table.read('members_init_galah.csv', format='ascii', delimiter='\t')['source_id']
        except:
            g_init = list([])

        try:
            g_in = Table.read('possible_ejected-step1_galah.csv', format='ascii', delimiter='\t')['source_id']
        except:
            g_in = list([])

        try:
            g_out = Table.read('possible_outside-step1_galah.csv', format='ascii', delimiter='\t')['source_id']
        except:
            g_out = list([])

        chdir('..')

        if len(g_init) == 0 or len(g_in) == 0 or len(g_out) == 0:
            print ' Some Galah lists are missing'

        idx_init = np.in1d(cannon_data['source_id'], g_init)
        idx_in = np.in1d(cannon_data['source_id'], g_in)
        idx_out = np.in1d(cannon_data['source_id'], g_out)

        if USE_DR3:
            abund_cols = [c for c in cannon_data.colnames if '_fe' in c and 'nr_' not in c and 'e_' not in c and ('I' in c or 'II' in c or 'III' in c)]
        else:
            abund_cols = [c for c in cannon_data.colnames if '_abund' in c and len(c.split('_')) == 3]

        rg = (-1.2, 1.2)
        bs = 40

        x_cols_fig = 7
        y_cols_fig = 5

        # NEW: plot with PDF
        fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(15, 10))
        for i_c, col in enumerate(abund_cols):
            print col
            x_p = i_c % x_cols_fig
            y_p = int(1. * i_c / x_cols_fig)
            # idx_val = cannon_data['flag_'+col] == 0
            x_c_field, y_c_field = _prepare_pdf_data(cannon_data[col][idx_out], cannon_data['e_'+col][idx_out], rg)
            x_c_init, y_c_init = _prepare_pdf_data(cannon_data[col][idx_init], cannon_data['e_'+col][idx_init], rg)
            x_c_in, y_c_in = _prepare_pdf_data(cannon_data[col][idx_in], cannon_data['e_'+col][idx_in], rg)

            ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
            ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
            ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected')

            ax[y_p, x_p].set(ylim=(0, 1.02), title=col.split('_')[0], xlim=rg, xticks=[-1., -0.5, 0, 0.5, 1.], xticklabels=['-1.', '', '0', '', '1.'])
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')
            if i_c == 0:
                ax[y_p, x_p].legend()

        x_p = -1
        y_p = -1
        x_c_field, y_c_field = _prepare_pdf_data(cannon_data[fe_col][idx_out], cannon_data['e_' + fe_col][idx_out], rg)
        x_c_init, y_c_init = _prepare_pdf_data(cannon_data[fe_col][idx_init], cannon_data['e_' + fe_col][idx_init], rg)
        x_c_in, y_c_in = _prepare_pdf_data(cannon_data[fe_col][idx_in], cannon_data['e_' + fe_col][idx_in], rg)

        ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
        ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
        ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected')

        ax[y_p, x_p].set(ylim=(0, 1.02), title='Fe/H', xlim=rg)
        ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98,hspace=0.3, wspace=0.3)
        # plt.show()
        plt.savefig('p_abundances_' + sub_dir + '' + suffix + '.png', dpi=250)
        plt.close(fig)

        # OLD: plot with histogram bins
        fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(15, 10))
        for i_c, col in enumerate(abund_cols):
            print col
            x_p = i_c % x_cols_fig
            y_p = int(1. * i_c / x_cols_fig)
            # idx_val = cannon_data['flag_'+col] == 0
            idx_val = np.isfinite(cannon_data[col])

            # plots
            h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[col][np.logical_and(idx_out, idx_val)], bs, rg)
            ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.2, facecolor='C2', edgecolor=None, label='Field')
            ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C2', label='')

            h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[col][np.logical_and(idx_init, idx_val)], bs, rg)
            ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.3, facecolor='C0', edgecolor=None, label='Initial')
            ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C0', label='')

            h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[col][np.logical_and(idx_in, idx_val)], bs, rg)
            ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.3, facecolor='C1', edgecolor=None, label='Ejected')
            ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C1', label='')

            ax[y_p, x_p].set(ylim=(0, 1.02), title=col.split('_')[0], xlim=rg, xticks=[-1.5, -0.75, 0, 0.75, 1.5], xticklabels=['-1.5', '', '0', '', '1.5'])
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')
            if i_c == 0:
                ax[y_p, x_p].legend()

        rg = (-1.5, 1.5)
        # idx_val = cannon_data['flag_cannon'] == 0
        # col = 'Fe_H_cannon'
        idx_val = np.isfinite(cannon_data[teff_col])
        x_p = -1
        y_p = -1
        h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[fe_col][np.logical_and(idx_out, idx_val)], bs, rg)
        ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.2, facecolor='C2', edgecolor=None, label='Field')
        ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C2', label='')

        h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[fe_col][np.logical_and(idx_init, idx_val)], bs, rg)
        ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.3, facecolor='C0', edgecolor=None, label='Initial')
        ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C0', label='')

        h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[fe_col][np.logical_and(idx_in, idx_val)], bs, rg)
        ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.3, facecolor='C1', edgecolor=None, label='Ejected')
        ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C1', label='')

        ax[y_p, x_p].set(ylim=(0, 1.02), title='Fe/H', xlim=rg)
        ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98,
                            hspace=0.3, wspace=0.3)

        # plt.show()
        plt.savefig('h_abundances_' + sub_dir + '' + suffix + '.png', dpi=250)
        plt.close(fig)

    # go to the directory with all simulations
    chdir('..')
