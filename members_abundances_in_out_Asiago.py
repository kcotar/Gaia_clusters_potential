import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from astropy.table import Table, join
from os import chdir


def _prepare_hist_data(d, bins, range, norm=True):
    heights, edges = np.histogram(d, bins=bins, range=range)
    width = np.abs(edges[0] - edges[1])
    if norm:
        heights = 1.*heights / np.nanmax(heights)
    return edges[:-1], heights, width


root_dir = '/data4/cotar/'
data_dir = root_dir + 'Asiago_reduced_data/NGC6940_params_R19000/'
data_dir_clusters = root_dir+'Gaia_open_clusters_analysis_September/Cluster_orbits_Gaia_DR2__05/NGC_6940_orbits/'

cannon_data = Table.read(data_dir + 'NGC6940_reduced_params.fits')

chdir(data_dir_clusters)

try:
    g_init = Table.read('members_init.csv', format='ascii', delimiter='\t')
    g_in = Table.read('possible_ejected-step1.csv', format='ascii', delimiter='\t')
    g_out = Table.read('possible_outside-step1.csv', format='ascii', delimiter='\t')
except:
    print ' Some Galah lists are missing'
    chdir('..')
    raise SystemExit

idx_init = np.in1d(cannon_data['source_id'], g_init['source_id'])
idx_in = np.in1d(cannon_data['source_id'], g_in['source_id'])
idx_out = np.in1d(cannon_data['source_id'], g_out['source_id'])

abund_cols = [c for c in cannon_data.colnames if len(c) <= 2 and c not in ['rv', 'Li']]

rg = (-1.75, 1.75)
bs = 40

x_cols_fig = 6
y_cols_fig = 5
fig, ax = plt.subplots(y_cols_fig, x_cols_fig, figsize=(15, 10))
for i_c, col in enumerate(abund_cols):
    print col
    x_p = i_c % x_cols_fig
    y_p = int(1. * i_c / x_cols_fig)
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

chdir('..')
col = 'feh'
x_p = -1
y_p = -1
idx_val = np.isfinite(cannon_data[col])

h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[col][np.logical_and(idx_out, idx_val)], bs, rg)
ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.2, facecolor='C2', edgecolor=None, label='Field')
ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C2', label='')

h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[col][np.logical_and(idx_init, idx_val)], bs, rg)
ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.3, facecolor='C0', edgecolor=None, label='Initial')
ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C0', label='')

h_edg, h_hei, h_wid = _prepare_hist_data(cannon_data[col][np.logical_and(idx_in, idx_val)], bs, rg)
ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, alpha=0.3, facecolor='C1', edgecolor=None, label='Ejected')
ax[y_p, x_p].step(h_edg, h_hei, where='mid', lw=0.6, alpha=1., color='C1', label='')

ax[y_p, x_p].set(ylim=(0, 1.02), title='Fe/H', xlim=rg)
ax[y_p, x_p].grid(ls='--', alpha=0.2, color='black')

plt.subplots_adjust(top=0.97, bottom=0.02, left=0.04, right=0.98,
                    hspace=0.3, wspace=0.3)

# plt.show()
plt.savefig('abundances_NGC_6583_orbits_Asiago.png', dpi=200)
plt.close(fig)




