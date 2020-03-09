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

# turn off polyfit ranking warnings
import warnings
warnings.filterwarnings('ignore')


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


def _evaluate_abund_trend_fit(orig, fit, idx, sigma_low, sigma_high):
    # diffence to the original data
    diff = orig - fit
    std_diff = np.nanstd(diff[idx])
    # select data that will be fitted
    idx_outlier = np.logical_or(diff < (-1. * std_diff * sigma_low),
                                diff > (std_diff * sigma_high))
    return np.logical_and(idx, ~idx_outlier)


def fit_abund_trend(p_data, a_data,
                    steps=3, sigma_low=2.5, sigma_high=2.5,
                    order=5, window=10, n_min_perc=10.,func='poly'):

    idx_fit = np.logical_and(np.isfinite(p_data), np.isfinite(a_data))
    data_len = np.sum(idx_fit)

    n_fit_points_prev = np.sum(idx_fit)
    if data_len <= order + 1:
        return None, None
    p_offset = np.nanmedian(p_data)

    for i_f in range(steps):  # number of sigma clipping steps
        if func == 'cheb':
            coef = np.polynomial.chebyshev.chebfit(p_data[idx_fit] - p_offset, a_data[idx_fit], order)
            f_data = np.polynomial.chebyshev.chebval(p_data - p_offset, coef)
        if func == 'legen':
            coef = np.polynomial.legendre.legfit(p_data[idx_fit] - p_offset, a_data[idx_fit], order)
            f_data = np.polynomial.legendre.legval(p_data - p_offset, coef)
        if func == 'poly':
            coef = np.polyfit(p_data[idx_fit] - p_offset, a_data[idx_fit], order)
            f_data = np.poly1d(coef)(p_data - p_offset)
        if func == 'spline':
            coef = splrep(p_data[idx_fit] - p_offset, a_data[idx_fit], k=order, s=window)
            f_data = splev(p_data - p_offset, coef)

        idx_fit = _evaluate_abund_trend_fit(a_data, f_data, idx_fit, sigma_low, sigma_high)
        n_fit_points = np.sum(idx_fit)
        if 100.*n_fit_points/data_len < n_min_perc:
            break
        if n_fit_points == n_fit_points_prev:
            break
        else:
            n_fit_points_prev = n_fit_points

    a_std = np.nanstd(a_data - f_data)
    return [coef, p_offset], a_std


def eval_abund_trend(p_data, m_data, func='poly'):
    coef, p_offset = m_data

    if func == 'cheb':
        f_data = np.polynomial.chebyshev.chebval(p_data - p_offset, coef)
    if func == 'legen':
        f_data = np.polynomial.legendre.legval(p_data - p_offset, coef)
    if func == 'poly':
        f_data = np.poly1d(coef)(p_data - p_offset)
    if func == 'spline':
        f_data = splev(p_data - p_offset, coef)

    return f_data


simulation_dir = '/shared/data-camelot/cotar/'
data_dir_clusters = simulation_dir+'GaiaDR2_open_clusters_2001_GALAH/'

data_dir = '/shared/ebla/cotar/'
USE_DR3 = True
Q_FLAGS = True
P_INDIVIDUAL = False
suffix = ''

if len(argv) > 1:
    # parse input options
    opts, args = getopt(argv[1:], '', ['dr3=', 'suffix=', 'flags=', 'individual='])
    # set parameters, depending on user inputs
    print(opts)
    for o, a in opts:
        if o == '--dr3':
            USE_DR3 = int(a) > 0
        if o == '--suffix':
            suffix += str(a)
        if o == '--flags':
            Q_FLAGS = int(a) > 0
        if o == '--individual':
            P_INDIVIDUAL = int(a) > 0

CG_data = Table.read(data_dir+'clusters/Cantat-Gaudin_2018/members.fits')
tails_data = Table.read(data_dir+'clusters/cluster_tails/members_open_gaia_tails.fits')

# remove cluster members from tails data
print('Cluster members all:', len(CG_data), len(tails_data))
idx_not_in_cluster = np.in1d(tails_data['source_id'], CG_data['source_id'], invert=True)
tails_data = tails_data[idx_not_in_cluster]
print('Cluster members all:', len(CG_data), len(tails_data))

if USE_DR3:
    # cannon_data = Table.read(data_dir+'GALAH_iDR3_main_alpha_190529.fits')
    cannon_data = Table.read(data_dir+'GALAH_iDR3_main_191213.fits')
    fe_col = 'fe_h'
    teff_col = 'teff'
    q_flag = 'flag_sp'
    suffix += '_DR3'
else:
    pass

if Q_FLAGS:
    suffix += '_flag0'

# determine all possible simulation subdirs
chdir(data_dir_clusters)
for cluster_dir in glob('Cluster_orbits_GaiaDR2_*'):
    chdir(cluster_dir)
    print('Working on clusters in ' + cluster_dir)

    for sub_dir in glob('*'):

        current_cluster = '_'.join(sub_dir.split('_')[0:2])
        source_id_cg = CG_data[CG_data['cluster'] == current_cluster]['source_id']
        source_id_tail = tails_data[tails_data['cluster'] == current_cluster]['source_id']
        idx_cg_memb = np.in1d(cannon_data['source_id'], np.array(source_id_cg))
        idx_tail = np.in1d(cannon_data['source_id'], np.array(source_id_tail))

        if '.png' in sub_dir or 'individual-abund' in sub_dir:
            continue

        print(' ')
        print(sub_dir)
        chdir(sub_dir)

        try:
            g_init = Table.read('members_init_galah.csv', format='ascii', delimiter='\t')
            idx_init = np.in1d(cannon_data['source_id'], g_init['source_id'])
        except:
            idx_init = np.full(len(cannon_data), False)

        try:
            g_in_all = Table.read('possible_ejected-step1.csv', format='ascii', delimiter='\t')
            g_in = Table.read('possible_ejected-step1_galah.csv', format='ascii', delimiter='\t')
            # further refinement of results to be plotted here
            g_in_all = g_in_all[np.logical_and(g_in_all['time_in_cluster'] >= 1.,  # [Myr] longest time (of all incarnations) inside cluster
                                               g_in_all['in_cluster_prob'] >= 68.)]  # percentage of reincarnations inside cluster
            g_in = g_in[np.logical_and(g_in['time_in_cluster'] >= 1.,
                                       g_in['in_cluster_prob'] >= 68.)]
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
            print(' Some Galah lists are missing')

        if USE_DR3:
            abund_cols = [c for c in cannon_data.colnames if '_fe' in c and 'nr_' not in c and 'diff_' not in c and 'e_' not in c and 'Li' not in c and 'alpha' not in c]  # and ('I' in c or 'II' in c or 'III' in c)]
        else:
            abund_cols = [c for c in cannon_data.colnames if '_abund' in c and len(c.split('_')) == 3]

        # abund_cols = ['e_' + cc for cc in abund_cols]
        # rg = (0., 0.35)
        # yt = [0., 0.1, 0.2, 0.3]
        # medfix = '-snr-sigma_'

        abund_cols = ['diff_' + cc for cc in abund_cols]
        rg = (-0.45, 0.45)
        yt = [-0.3, -0.15, 0.0, 0.15, 0.3]
        medfix = '-detrended-snr_'

        # ------------------------------------------------------------------------------
        # NEW: plot with parameter dependency trends
        # ------------------------------------------------------------------------------
        bs = 40

        x_cols_fig = 7
        y_cols_fig = 5

        param_lims = {'snr_c2_iraf': [5, 175], 'age': [0., 14.], 'teff': [3000, 7000], 'logg': [0.0, 5.5], 'fe_h': [-1.2, 0.5]}
        for param in ['snr_c2_iraf']: #list(param_lims.keys()):
            cannon_data['abund_det'] = 0
            cannon_data['abund_det_elems'] = 0
            print('Estimating membership using parameter', param)
            fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(15, 10))
            for i_c, col in enumerate(abund_cols):
                # print(col)
                x_p = i_c % x_cols_fig
                y_p = int(1. * i_c / x_cols_fig)

                fit_x_param = 'teff'
                cur_abund_col = '_'.join(col.split('_')[1:])

                cannon_data['diff_' + cur_abund_col] = cannon_data[cur_abund_col]

                idx_val = np.isfinite(cannon_data[col])
                if Q_FLAGS:
                    idx_val = np.logical_and(idx_val, cannon_data[q_flag] == 0)

                idx_u1 = np.logical_and(idx_out, idx_val)
                idx_u2 = np.logical_and(idx_init, idx_val)
                idx_u3 = np.logical_and(idx_in, idx_val)
                idx_u4 = np.logical_and(idx_cg_memb, idx_val)
                idx_u5 = np.logical_and(idx_tail, idx_val)

                fit_model, col_std = fit_abund_trend(cannon_data[fit_x_param][idx_u2],
                                                     cannon_data[cur_abund_col][idx_u2],
                                                     order=3, steps=2, func='poly',
                                                     sigma_low=2.5, sigma_high=2.5, n_min_perc=10.)

                if fit_model is not None:
                    cannon_data['diff_' + cur_abund_col] = cannon_data[cur_abund_col] - eval_abund_trend(cannon_data[fit_x_param], fit_model, func='poly')
                else:
                    cannon_data['diff_' + cur_abund_col] = np.nan

                ax[y_p, x_p].scatter(cannon_data[param][idx_u1], cannon_data[col][idx_u1],
                                     lw=0, s=3, color='C2', label='Field')
                ax[y_p, x_p].scatter(cannon_data[param][idx_u2], cannon_data[col][idx_u2],
                                     lw=0, s=3, color='C0', label='Initial')
                ax[y_p, x_p].scatter(cannon_data[param][idx_u3], cannon_data[col][idx_u3],
                                     lw=0, s=3, color='C1', label='Ejected')
                if np.sum(idx_u5) > 0:
                    print('Ejected in tail:', np.sum(np.logical_and(idx_u3, idx_u5)))
                    ax[y_p, x_p].scatter(cannon_data[param][idx_u5], cannon_data[col][idx_u5],
                                         lw=0, s=3, color='C4', label='Tail')

                label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
                ax[y_p, x_p].set(xlim=param_lims[param], title=' '.join(col.split('_')[:2]) + label_add,
                                 ylim=rg,
                                 yticks=yt,)
                ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

            rg = (-0.6, 0.6)
            idx_val = np.isfinite(cannon_data[teff_col])
            if Q_FLAGS:
                idx_val = np.logical_and(idx_val, cannon_data[q_flag] == 0)

            x_p = -1
            y_p = -1

            idx_u1 = np.logical_and(idx_out, idx_val)
            idx_u2 = np.logical_and(idx_init, idx_val)
            idx_u3 = np.logical_and(idx_in, idx_val)
            idx_u5 = np.logical_and(idx_tail, idx_val)

            sl1 = ax[y_p, x_p].scatter(cannon_data[param][idx_u1], cannon_data[fe_col][idx_u1],
                                 lw=0, s=3, color='C2', label='Field')
            sl2 = ax[y_p, x_p].scatter(cannon_data[param][idx_u2], cannon_data[fe_col][idx_u2],
                                 lw=0, s=3, color='C0', label='Initial')
            sl3 = ax[y_p, x_p].scatter(cannon_data[param][idx_u3], cannon_data[fe_col][idx_u3],
                                 lw=0, s=3, color='C1', label='Ejected')

            fit_model, col_std = fit_abund_trend(cannon_data[param][idx_u2], cannon_data[fe_col][idx_u2],
                                                 order=3, steps=2, sigma_low=2.5, sigma_high=2.5, n_min_perc=10.,
                                                 func='poly')

            if np.sum(idx_u5) > 0:
                sl5 = ax[y_p, x_p].scatter(cannon_data[param][idx_u5], cannon_data[fe_col][idx_u5],
                                     lw=0, s=3, color='C4', label='Tail')
                ax[-1, -3].legend(handles=[sl1, sl1, sl3, sl5])
            else:
                ax[-1, -3].legend(handles=[sl1, sl1, sl3])

            label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
            ax[y_p, x_p].set(ylim=rg, title='Fe/H' + label_add, xlim=param_lims[param])
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

            x_p = -2
            y_p = -1

            ax[y_p, x_p].scatter(cannon_data['age'][idx_u1], cannon_data[param][idx_u1],
                                 lw=0, s=3, color='C2', label='Field')
            ax[y_p, x_p].scatter(cannon_data['age'][idx_u2], cannon_data[param][idx_u2],
                                 lw=0, s=3, color='C0', label='Initial')
            ax[y_p, x_p].scatter(cannon_data['age'][idx_u3], cannon_data[param][idx_u3],
                                 lw=0, s=3, color='C1', label='Ejected')

            if np.sum(idx_u5) > 0:
                ax[y_p, x_p].scatter(cannon_data['age'][idx_u5], cannon_data[param][idx_u5],
                                     lw=0, s=3, color='C4', label='Tail')

            label_add = ' = {:.0f}, {:.0f}, {:.0f}'.format(np.sum(idx_u1), np.sum(idx_u2), np.sum(idx_u3))
            ax[y_p, x_p].set(ylim=param_lims[param], title='age' + label_add, xlim=[0., 14.])
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

            plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98, hspace=0.3, wspace=0.3)
            # plt.show()
            plt.savefig('p_' + param + '_abundances' + medfix + sub_dir + '' + suffix + '.png', dpi=250)
            plt.close(fig)

    chdir('..')
