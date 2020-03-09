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
from copy import copy, deepcopy

# turn off polyfit ranking warnings
import warnings
warnings.filterwarnings('ignore')

def _prepare_pdf_data(means, stds, range, bins=2000, norm=True):
    x_vals = np.linspace(range[0], range[1], bins)
    y_vals = np.zeros_like(x_vals)
    # create and sum all PDF of stellar abundances
    for d_m, d_s in zip(means, stds):
        if np.isfinite([d_m, d_s]).all():
            y_vals += gauss_norm.pdf(x_vals, loc=d_m, scale=d_s)
    # return normalized summed pdf of all stars
    if norm and np.nansum(y_vals) > 0.:
        y_vals = 1. * y_vals/np.nanmax(y_vals)
    return x_vals, y_vals

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
tails_data_all = deepcopy(tails_data)

for i_l in range(len(CG_data)):
    CG_data['cluster'][i_l] = str(CG_data['cluster'][i_l]).lstrip().rstrip()

# remove cluster members from tails data
print('Cluster members all:', len(CG_data), len(tails_data))
idx_not_in_cluster = np.in1d(tails_data['source_id'], CG_data['source_id'], invert=True)
tails_data = tails_data[idx_not_in_cluster]
print('Cluster members all:', len(CG_data), len(tails_data))

cannon_data = Table.read(data_dir+'GALAH_iDR3_main_191213.fits')
cannon_data_valid = cannon_data[cannon_data['flag_sp'] == 0]

# determine all possible simulation subdirs
chdir(data_dir_clusters)
for cluster_dir in glob('Cluster_orbits_GaiaDR2_*'):
    chdir(cluster_dir)
    print('Working on clusters in ' + cluster_dir)

    for sub_dir in glob('*'):

        current_cluster = '_'.join(sub_dir.split('_')[0:2])
        source_id_cg = CG_data[CG_data['cluster'] == current_cluster]['source_id']
        source_id_tail = tails_data[tails_data['cluster'] == current_cluster]['source_id']
        source_id_tail_all = tails_data_all[tails_data_all['cluster'] == current_cluster]['source_id']

        idx_cg_memb = np.in1d(cannon_data['source_id'], np.array(source_id_cg))
        idx_tail = np.in1d(cannon_data['source_id'], np.array(source_id_tail))

        if '.png' in sub_dir or 'individual-abund' in sub_dir:
            continue

        print(' ')
        print(sub_dir)
        chdir(sub_dir)

        gaia_data_used = Table.read('gaia_query_data_combined_used.fits')
        g_tails = gaia_data_used[np.in1d(gaia_data_used['source_id'], np.array(source_id_tail))]

        try:
            g_init = Table.read('members_init.csv', format='ascii', delimiter='\t')
            idx_init = np.in1d(cannon_data['source_id'], g_init['source_id'])
        except:
            idx_init = np.full(len(cannon_data), False)

        try:
            g_in = Table.read('possible_ejected-step1.csv', format='ascii', delimiter='\t')
            g_in = g_in[np.logical_and(g_in['time_in_cluster'] >= 1.,
                                       g_in['in_cluster_prob'] >= 68.)]
            idx_in = np.in1d(cannon_data['source_id'], g_in['source_id'])
            idx_in_no_CG = np.logical_and(idx_in,
                                          np.logical_not(np.in1d(cannon_data['source_id'], CG_data['source_id'])))
        except:
            idx_in = np.full(len(cannon_data), False)
            idx_in_no_CG = np.full(len(cannon_data), False)

        try:
            g_out = Table.read('possible_outside-step1.csv', format='ascii', delimiter='\t')
            g_out = g_out[np.logical_and(g_out['time_in_cluster'] <= 0,
                                         g_out['in_cluster_prob'] <= 0)]
            idx_out = np.in1d(cannon_data['source_id'], g_out['source_id'])
        except:
            idx_out = np.full(len(cannon_data), False)

        chdir('..')

        if np.sum(idx_init) == 0 or np.sum(idx_in) == 0 or np.sum(idx_out) == 0:
            print(' Some kinematics lists are missing')

        if 'parallax' not in g_out.colnames:
            print(' Kinematics data not found in the supplied data set')
            continue

        if len(source_id_tail_all) > 0:
            print('Stars in tail reference:', len(source_id_tail_all))
            print('Without used cluster members:', np.sum(np.in1d(source_id_tail_all, source_id_cg, invert=True)))
            n_c_w_e = np.sum(np.in1d(g_in['source_id'], source_id_tail_all))
            print('Common with ejected:', n_c_w_e, n_c_w_e/len(g_in['source_id'])*100)
            idx_tail_valid = np.in1d(cannon_data_valid['source_id'], source_id_tail)
            idx_ejected_valid = np.in1d(cannon_data_valid['source_id'], g_in['source_id'])
            n_w_u_p = np.sum(idx_tail_valid)
            print('With unflagged GALAH parameters:', n_w_u_p, np.sum(idx_tail_valid * idx_ejected_valid)/n_w_u_p*100)

        # ------------------------------------------------------------------------------
        # NEW: plot with PDF
        # ------------------------------------------------------------------------------
        rg = (-1.0, 1.0)
        bs = 40

        x_cols_fig = 2
        y_cols_fig = 2

        plot_cols = {
            'parallax': np.nanmedian(g_init['parallax']) + np.array([-4., 4.]),
            'rv': np.nanmedian(g_init['rv']) + np.array([-25., 25.]),
            'pmra': np.nanmedian(g_init['pmra']) + np.array([-25., 25.]),
            'pmdec': np.nanmedian(g_init['pmdec']) + np.array([-25., 25.]),
        }

        fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(7, 7))
        for i_c, col in enumerate(list(plot_cols.keys())):
            print(col)
            x_p = i_c % x_cols_fig
            y_p = int(1. * i_c / x_cols_fig)

            rg = plot_cols[col]

            x_c_field, y_c_field = _prepare_pdf_data(g_out[col], g_out[col + '_error'], rg)
            x_c_init, y_c_init = _prepare_pdf_data(g_init[col], g_init[col + '_error'], rg)
            x_c_in, y_c_in = _prepare_pdf_data(g_in[col], g_in[col + '_error'], rg)

            ax[y_p, x_p].plot(x_c_field, y_c_field, lw=1, color='C2', label='Field')
            ax[y_p, x_p].plot(x_c_init, y_c_init, lw=1, color='C0', label='Initial')
            ax[y_p, x_p].plot(x_c_in, y_c_in, lw=1, color='C1', label='Ejected')

            if len(g_tails) > 0:
                x_c_tail, y_c_tail = _prepare_pdf_data(g_tails[col], g_tails[col + '_error'], rg)
                ax[y_p, x_p].plot(x_c_tail, y_c_tail, lw=1, color='C4', label='Tail')
            else:
                y_c_tail = np.full_like(y_c_in, fill_value=0.)

            y_total = y_c_init + y_c_in + y_c_tail
            y_limits = np.where(y_total > 0.001)[0]
            x_limits = [x_c_in[y_limits[0]], x_c_in[y_limits[-1]]]

            label_add = ' = {:.0f}, {:.0f}, {:.0f}, {:.0f}'.format(len(g_out), len(g_init), len(g_in), len(g_tails))
            ax[y_p, x_p].set(ylim=(0, 1.02),
                             title=col.split('_')[0] + label_add,
                             xlim=x_limits,
                             # xticks=[-1., -0.5, 0, 0.5, 1.],
                             # xticklabels=['-1.', '', '0', '', '1.'],
                             )
            ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')
            if i_c == 0:
                ax[y_p, x_p].legend()

        plt.tight_layout()
        # plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98, hspace=0.3, wspace=0.3)
        # plt.show()
        plt.savefig('p_kinematics_' + sub_dir + '.png', dpi=250)
        plt.close(fig)

    # go to the directory with all simulations
    chdir('..')


