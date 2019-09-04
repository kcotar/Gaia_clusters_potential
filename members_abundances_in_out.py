import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from astropy.table import Table, join
from os import chdir, system
from scipy.stats import norm as gauss_norm
from sys import argv
from getopt import getopt


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
data_dir_clusters = simulation_dir+'GaiaDR2_open_clusters_1907_GALAH_CGmebers/'

data_dir = '/shared/ebla/cotar/'
USE_DR3 = True
Q_FLAGS = False
suffix = ''

if len(argv) > 1:
    # parse input options
    opts, args = getopt(argv[1:], '', ['dr3=', 'suffix=', 'flags='])
    # set parameters, depending on user inputs
    print opts
    for o, a in opts:
        if o == '--dr3':
            USE_DR3 = int(a) > 0
        if o == '--suffix':
            suffix += str(a)
        if o == '--flags':
            Q_FLAGS = int(a) > 0

CG_data = Table.read(data_dir+'clusters/Cantat-Gaudin_2018/members.fits')

if USE_DR3:
    cannon_data = Table.read(data_dir+'GALAH_iDR3_main_alpha_190529.fits')
    fe_col = 'fe_h'
    teff_col = 'teff'
    q_flag = 'flag_sp'
    suffix += '_DR3'
else:
    cannon_data = Table.read(data_dir+'sobject_iraf_iDR2_180325_cannon.fits')
    gaia_data = Table.read(data_dir+'sobject_iraf_53_gaia.fits')['source_id', 'sobject_id']
    cannon_data = join(cannon_data, gaia_data, keys='sobject_id', join_type='inner')
    fe_col = 'Fe_H_cannon'
    teff_col = 'Teff_cannon'
    q_flag = 'flag_cannon'
    suffix += '_DR2'

if Q_FLAGS:
    suffix += '_flag0'

# detemine all posible simulation subdirs
chdir(data_dir_clusters)
for cluster_dir in glob('Cluster_orbits_Gaia_DR2_*'):
    chdir(cluster_dir)
    print 'Working on clusters in ' + cluster_dir

    for sub_dir in glob('*'):

        if '.png' in sub_dir or 'individual-abund' in sub_dir:
            continue

        print sub_dir
        chdir(sub_dir)

        try:
            g_init = Table.read('members_init_galah.csv', format='ascii', delimiter='\t')
            idx_init = np.in1d(cannon_data['source_id'], g_init['source_id'])
        except:
            idx_init = np.full(len(cannon_data), False)

        try:
            g_in = Table.read('possible_ejected-step1_galah.csv', format='ascii', delimiter='\t')
            # further refinement of results to be plotted here
            g_in = g_in[np.logical_and(g_in['time_in_cluster'] >= 1.5,  # [Myr] longest time (of all incarnations) inside cluster
                                       g_in['in_cluster_prob'] >= 68.)]  # percentage of reincarnations inside cluster
            idx_in = np.in1d(cannon_data['source_id'], g_in['source_id'])
            idx_in_no_CG = np.logical_and(idx_in,
                                          np.logical_not(np.in1d(cannon_data['source_id'], CG_data['source_id'])))
        except:
            idx_in = np.full(len(cannon_data), False)
            idx_in_no_CG = np.full(len(cannon_data), False)

        try:
            g_out = Table.read('possible_outside-step1_galah.csv', format='ascii', delimiter='\t')
            # further refinement of results to be plotted here
            g_out = g_out[np.logical_and(g_out['time_in_cluster'] <= 0,
                                         g_out['in_cluster_prob'] <= 0)]
            idx_out = np.in1d(cannon_data['source_id'], g_out['source_id'])
        except:
            idx_out = np.full(len(cannon_data), False)

        chdir('..')

        if np.sum(idx_init) == 0 or np.sum(idx_in) == 0 or np.sum(idx_out) == 0:
            print ' Some Galah lists are missing'

        if USE_DR3:
            abund_cols = [c for c in cannon_data.colnames if '_fe' in c and 'nr_' not in c and 'e_' not in c and ('I' in c or 'II' in c or 'III' in c)]
        else:
            abund_cols = [c for c in cannon_data.colnames if '_abund' in c and len(c.split('_')) == 3]

        # ------------------------------------------------------------------------------
        # NEW: plot with PDF
        # ------------------------------------------------------------------------------
        rg = (-1.0, 1.0)
        bs = 40

        x_cols_fig = 7
        y_cols_fig = 5

        fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(15, 10))
        for i_c, col in enumerate(abund_cols):
            print col
            x_p = i_c % x_cols_fig
            y_p = int(1. * i_c / x_cols_fig)

            idx_val = np.isfinite(cannon_data[col])
            if Q_FLAGS and not USE_DR3:
                # Elements in DR3 (MgI, SiI, CaI, TiI, TiII ...) are only computed if at least one unflagged line was available
                idx_val = np.logical_and(idx_val, cannon_data['flag_' + col] == 0)

            idx_u1 = np.logical_and(idx_out, idx_val)
            idx_u2 = np.logical_and(idx_init, idx_val)
            idx_u3 = np.logical_and(idx_in, idx_val)
            x_c_field, y_c_field = _prepare_pdf_data(cannon_data[col][idx_u1], cannon_data['e_'+col][idx_u1], rg)
            x_c_init, y_c_init = _prepare_pdf_data(cannon_data[col][idx_u2], cannon_data['e_'+col][idx_u2], rg)
            x_c_in, y_c_in = _prepare_pdf_data(cannon_data[col][idx_u3], cannon_data['e_'+col][idx_u3], rg)

            ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
            ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
            ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected')

            label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
            ax[y_p, x_p].set(ylim=(0, 1.02), title=col.split('_')[0] + label_add, xlim=rg, xticks=[-1., -0.5, 0, 0.5, 1.], xticklabels=['-1.', '', '0', '', '1.'])
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')
            if i_c == 0:
                ax[y_p, x_p].legend()

        rg = (-1.7, 0.5)
        idx_val = np.isfinite(cannon_data[teff_col])
        if Q_FLAGS:
            idx_val = np.logical_and(idx_val, cannon_data[q_flag] == 0)

        x_p = -1
        y_p = -1

        idx_u1 = np.logical_and(idx_out, idx_val)
        idx_u2 = np.logical_and(idx_init, idx_val)
        idx_u3 = np.logical_and(idx_in, idx_val)
        x_c_field, y_c_field = _prepare_pdf_data(cannon_data[fe_col][idx_u1], cannon_data['e_' + fe_col][idx_u1], rg)
        x_c_init, y_c_init = _prepare_pdf_data(cannon_data[fe_col][idx_u2], cannon_data['e_' + fe_col][idx_u2], rg)
        x_c_in, y_c_in = _prepare_pdf_data(cannon_data[fe_col][idx_u3], cannon_data['e_' + fe_col][idx_u3], rg)

        ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
        ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
        ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected')

        label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
        ax[y_p, x_p].set(ylim=(0, 1.02), title='Fe/H' + label_add, xlim=rg)
        ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98, hspace=0.3, wspace=0.3)
        # plt.show()
        plt.savefig('p_abundances_' + sub_dir + '' + suffix + '.png', dpi=250)
        plt.close(fig)


        # ------------------------------------------------------------------------------
        # NEW: plot with PDF - excluding known cluster stars from ejected population
        # ------------------------------------------------------------------------------
        rg = (-1.0, 1.0)
        bs = 40

        x_cols_fig = 7
        y_cols_fig = 5

        fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(15, 10))
        for i_c, col in enumerate(abund_cols):
            print col
            x_p = i_c % x_cols_fig
            y_p = int(1. * i_c / x_cols_fig)

            idx_val = np.isfinite(cannon_data[col])
            if Q_FLAGS and not USE_DR3:
                # Elements in DR3 (MgI, SiI, CaI, TiI, TiII ...) are only computed if at least one unflagged line was available
                idx_val = np.logical_and(idx_val, cannon_data['flag_' + col] == 0)

            idx_u1 = np.logical_and(idx_out, idx_val)
            idx_u2 = np.logical_and(idx_init, idx_val)
            idx_u3 = np.logical_and(idx_in_no_CG, idx_val)
            x_c_field, y_c_field = _prepare_pdf_data(cannon_data[col][idx_u1], cannon_data['e_'+col][idx_u1], rg)
            x_c_init, y_c_init = _prepare_pdf_data(cannon_data[col][idx_u2], cannon_data['e_'+col][idx_u2], rg)
            x_c_in, y_c_in = _prepare_pdf_data(cannon_data[col][idx_u3], cannon_data['e_'+col][idx_u3], rg)

            ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
            ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
            ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected - CG memb')

            label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
            ax[y_p, x_p].set(ylim=(0, 1.02), title=col.split('_')[0] + label_add, xlim=rg, xticks=[-1., -0.5, 0, 0.5, 1.], xticklabels=['-1.', '', '0', '', '1.'])
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')
            if i_c == 0:
                ax[y_p, x_p].legend()

        rg = (-1.7, 0.5)
        idx_val = np.isfinite(cannon_data[teff_col])
        if Q_FLAGS:
            idx_val = np.logical_and(idx_val, cannon_data[q_flag] == 0)

        x_p = -1
        y_p = -1

        idx_u1 = np.logical_and(idx_out, idx_val)
        idx_u2 = np.logical_and(idx_init, idx_val)
        idx_u3 = np.logical_and(idx_in_no_CG, idx_val)
        x_c_field, y_c_field = _prepare_pdf_data(cannon_data[fe_col][idx_u1], cannon_data['e_' + fe_col][idx_u1], rg)
        x_c_init, y_c_init = _prepare_pdf_data(cannon_data[fe_col][idx_u2], cannon_data['e_' + fe_col][idx_u2], rg)
        x_c_in, y_c_in = _prepare_pdf_data(cannon_data[fe_col][idx_u3], cannon_data['e_' + fe_col][idx_u3], rg)

        ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
        ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
        ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected - CG memb')

        label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
        ax[y_p, x_p].set(ylim=(0, 1.02), title='Fe/H' + label_add, xlim=rg)
        ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98, hspace=0.3, wspace=0.3)
        # plt.show()
        if 'CGmebers' not in data_dir_clusters:
            plt.savefig('p_abundances_' + sub_dir + '' + suffix + '_noCGmemb.png', dpi=250)
        plt.close(fig)



        # ------------------------------------------------------------------------------
        # OLD: plot with histogram bins
        # ------------------------------------------------------------------------------
        rg = (-1.0, 1.0)
        bs = 40

        x_cols_fig = 7
        y_cols_fig = 5

        fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(15, 10))
        for i_c, col in enumerate(abund_cols):
            print col
            x_p = i_c % x_cols_fig
            y_p = int(1. * i_c / x_cols_fig)

            idx_val = np.isfinite(cannon_data[col])
            if Q_FLAGS and not USE_DR3:
                # Elements in DR3 (MgI, SiI, CaI, TiI, TiII ...) are only computed if at least one unflagged line was available
                idx_val = np.logical_and(idx_val, cannon_data['flag_'+col] == 0)

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

            label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(np.logical_and(idx_out, idx_val)),
                                                           np.sum(np.logical_and(idx_init, idx_val)),
                                                           np.sum(np.logical_and(idx_in, idx_val)))
            ax[y_p, x_p].set(ylim=(0, 1.02), title=col.split('_')[0] + label_add, xlim=rg, xticks=[-1., -0.5, 0, 0.5, 1.], xticklabels=['-1.', '', '0', '', '1.'])
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')
            if i_c == 0:
                ax[y_p, x_p].legend()

        rg = (-1.7, 0.5)
        idx_val = np.isfinite(cannon_data[teff_col])
        if Q_FLAGS:
            idx_val = np.logical_and(idx_val, cannon_data[q_flag] == 0)

        x_p = -1
        y_p = -1
        h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[fe_col][np.logical_and(idx_out, idx_val)], bs, rg)
        ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.2, facecolor='C2', edgecolor=None, label='Field ({:.0f})'.format(np.sum(np.logical_and(idx_out, idx_val))))
        ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C2', label='')

        h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[fe_col][np.logical_and(idx_init, idx_val)], bs, rg)
        ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.3, facecolor='C0', edgecolor=None, label='Initial ({:.0f})'.format(np.sum(np.logical_and(idx_init, idx_val))))
        ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C0', label='')

        h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[fe_col][np.logical_and(idx_in, idx_val)], bs, rg)
        ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.3, facecolor='C1', edgecolor=None, label='Ejected ({:.0f})'.format(np.sum(np.logical_and(idx_in, idx_val))))
        ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C1', label='')

        label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(np.logical_and(idx_out, idx_val)),
                                                       np.sum(np.logical_and(idx_init, idx_val)),
                                                       np.sum(np.logical_and(idx_in, idx_val)))
        ax[y_p, x_p].set(ylim=(0, 1.02), title='Fe/H' + label_add, xlim=rg)
        ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98,
                            hspace=0.3, wspace=0.3)

        # plt.show()
        plt.savefig('h_abundances_' + sub_dir + '' + suffix + '.png', dpi=250)
        plt.close(fig)

        # create new subdirectory with individual star plots
        new_sub_cluster_dir = sub_dir + '_individual-abund'
        system('mkdir ' + new_sub_cluster_dir)
        chdir(new_sub_cluster_dir)

        for i_star, id_star in enumerate(g_in['source_id']):

            rg = (-0.8, 0.8)
            if USE_DR3:
                rg = (-0.6, 0.6)

            x_cols_fig = 7
            y_cols_fig = 5

            fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(15, 10))
            for i_c, col in enumerate(abund_cols):
                print col
                x_p = i_c % x_cols_fig
                y_p = int(1. * i_c / x_cols_fig)

                idx_val = np.isfinite(cannon_data[col])
                if Q_FLAGS and not USE_DR3:
                    # Elements in DR3 (MgI, SiI, CaI, TiI, TiII ...) are only computed if at least one unflagged line was available
                    idx_val = np.logical_and(idx_val, cannon_data['flag_' + col] == 0)

                idx_u1 = np.logical_and(idx_out, idx_val)
                idx_u2 = np.logical_and(idx_init, idx_val)
                idx_u3 = np.logical_and(cannon_data['source_id'] == id_star, idx_val)
                x_c_field, y_c_field = _prepare_pdf_data(cannon_data[col][idx_u1], cannon_data['e_' + col][idx_u1], rg)
                x_c_init, y_c_init = _prepare_pdf_data(cannon_data[col][idx_u2], cannon_data['e_' + col][idx_u2], rg)
                x_c_in, y_c_in = _prepare_pdf_data(cannon_data[col][idx_u3], cannon_data['e_' + col][idx_u3], rg)

                ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
                ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
                ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected')

                label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
                if USE_DR3:
                    ax[y_p, x_p].set(ylim=(0, 1.02), title=col.split('_')[0] + label_add, xlim=rg,
                                     xticks=[-0.6, -0.3, 0, 0.3, 0.6], xticklabels=['-0.6', '', '0', '', '0.6'])
                else:
                    ax[y_p, x_p].set(ylim=(0, 1.02), title=col.split('_')[0] + label_add, xlim=rg,
                                     xticks=[-0.8, -0.4, 0, 0.4, 0.8], xticklabels=['-0.8', '', '0', '', '0.8'])

                ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')
                if i_c == 0:
                    ax[y_p, x_p].legend()

            rg = (-1.5, 0.4)
            idx_val = np.isfinite(cannon_data[teff_col])
            if Q_FLAGS:
                idx_val = np.logical_and(idx_val, cannon_data[q_flag] == 0)

            x_p = -1
            y_p = -1

            idx_u1 = np.logical_and(idx_out, idx_val)
            idx_u2 = np.logical_and(idx_init, idx_val)
            idx_u3 = np.logical_and(cannon_data['source_id'] == id_star, idx_val)
            x_c_field, y_c_field = _prepare_pdf_data(cannon_data[fe_col][idx_u1], cannon_data['e_' + fe_col][idx_u1], rg)
            x_c_init, y_c_init = _prepare_pdf_data(cannon_data[fe_col][idx_u2], cannon_data['e_' + fe_col][idx_u2], rg)
            x_c_in, y_c_in = _prepare_pdf_data(cannon_data[fe_col][idx_u3], cannon_data['e_' + fe_col][idx_u3], rg)

            ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
            ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
            ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected')

            label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
            ax[y_p, x_p].set(ylim=(0, 1.02), title='Fe/H' + label_add, xlim=rg)
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

            plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98, hspace=0.3, wspace=0.3)
            # plt.show()
            plt.savefig('p_abundances_' + sub_dir + '' + suffix + '_' + str(id_star) + '.png', dpi=250)
            plt.close(fig)

        chdir('..')

    # go to the directory with all simulations
    chdir('..')


