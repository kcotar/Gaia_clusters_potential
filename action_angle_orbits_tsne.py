from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE_multi
from helper_functions import move_to_dir
from copy import deepcopy
from sklearn.cluster import DBSCAN
from abundances_analysis import *

coord_cols = ['X', 'Y', 'Z']
vel_cols = ['V_X', 'V_Y', 'V_Z']
action_cols = ['J_R', 'L_Z', 'J_Z']
angle_cols = ['O_R', 'O_phi', 'O_z']
phase_cols = ['Th_R', 'Th_phi', 'Th_z']
orig_cols = ['ra', 'dec', 'pmra', 'pmdec', 'parsec']

COMPUTE_TSNE = True         # Step 1 - create tsne projection
EXOPLORE_RESULTS = False     # Step 2 - manual browsing the projection
AUTOMATIC_ANALYSIS = True  # Step 3 - automatic analysis of projection and its overdenseties

perp = 4
theta = 0.5
n_cpu_tsne = 30
n_cpu_dbscan = 20

galah_dir = '/home/klemen/data4_mount/'
data_dir = galah_dir+'TGAS_data_set/'
gaia_data_all = Table.read(data_dir + 'TgasSource_all_with_rv_feh.fits')
idx_gaia_use = np.logical_and(gaia_data_all['parallax'] > 0,     # positive parallaxes
                              1e3/gaia_data_all['parallax'] <= 750)  # inside a defined radius
source_id_use = gaia_data_all['source_id'][idx_gaia_use]

# # original Gaia data
# file_in = 'TgasSource_all_with_rv_feh.fits'
# suffix = 'origall'
# use_cols = orig_cols

# # reduced orbital data
# file_in = 'TgasSource_all_with_rv_feh_orbits-rvonly.fits'
# suffix = 'ac-an-ph-co'
# use_cols = list(np.hstack([coord_cols, action_cols, angle_cols, phase_cols]))

# reduced cartesian data
file_in = 'TgasSource_all_with_rv_feh_cartesian.fits'
suffix = 'cartesian'
use_cols = list(np.hstack([coord_cols, vel_cols]))

data = Table.read(data_dir + file_in)
if 'parallax' in data.colnames:
    data['parsec'] = 1./data['parallax']

# filter by source_ids
data = data[np.in1d(data['source_id'], source_id_use)]

# final destination of outputs
suffix += '_per{:2.0f}_th{:0.2}'.format(perp, theta)
move_to_dir('ORBITS_tsne_projections')
move_to_dir(suffix)
file_out = file_in[:-5]

if COMPUTE_TSNE:
    # filter out possible bad rows
    data_used_temp = data[use_cols].to_pandas().values
    idx_ok_rows = np.logical_and(np.isfinite(data_used_temp),               # filter out inf and nan values
                                 data_used_temp != 9999.99).all(axis=1)     # filter out bad galpy values
    n_bad = np.sum(np.logical_not(idx_ok_rows))
    if n_bad > 0:
        print ' Removed rows:', n_bad
        data = data[idx_ok_rows]
    data_used_temp = None
    data_orig = deepcopy(data)

    for col in use_cols:
        if col == 'Th_phi':
            data[col][data[col] > np.pi] -= 2*np.pi
        median = np.median(data[col])
        std = np.std(data[col])
        std_norm = np.std(data[col][np.abs(data[col] - median) < 3.*std])
        data[col] = (data[col]-median) / std_norm

    project_data = data[use_cols].to_pandas().values
    # project_data = normalize(project_data, axis=1)

    # run tSNE
    print 'Running tSNE projection'

    # tsne_class = TSNE(n_components=2, perplexity=perp, n_iter=1000, n_iter_without_progress=350, init='random', verbose=1,
    #                           method='barnes_hut', angle=theta)
    tsne_class = TSNE_multi(n_components=2, perplexity=perp, n_iter=1200, n_iter_without_progress=350, init='random', verbose=1,
                            method='barnes_hut', angle=theta, n_jobs=n_cpu_tsne)
    tsne_res = tsne_class.fit_transform(project_data)

    data['tsne_axis_1'] = tsne_res[:, 0]
    data['tsne_axis_2'] = tsne_res[:, 1]

    tsne_data = data['source_id', 'tsne_axis_1', 'tsne_axis_2']
    tsne_data.write(file_out+'_tsne.fits', overwrite=True)

    plt.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=1)
    plt.savefig(file_out+'_tsne.png', dpi=350)
    plt.close()

    # output plots with colours
    for c_c in use_cols:
        c_data = data_orig[c_c]
        plt.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=1, c=c_data, cmap='viridis',
                    vmin=np.percentile(c_data, 2), vmax=np.percentile(c_data, 98))
        plt.colorbar()
        plt.savefig(file_out + '_tsne_'+c_c+'.png', dpi=350)
        plt.close()

    clusters_path = '/home/klemen/data4_mount/clusters/Gaia_2017/tabled.csv'
    clusters_data = Table.read(clusters_path)

    for clust in np.unique(clusters_data['Cluster']):
        print 'Plot for:', clust
        idx_tsne_clust = np.in1d(tsne_data['source_id'], clusters_data['Source'][clusters_data['Cluster'] == clust])
        if np.sum(idx_tsne_clust) < 5:
            continue
        plt.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=0.75, c='black')
        plt.scatter(tsne_data['tsne_axis_1'][idx_tsne_clust], tsne_data['tsne_axis_2'][idx_tsne_clust], lw=0, s=1, c='red')
        plt.xlim(-40, 40)
        plt.ylim(-40, 40)
        plt.savefig(file_out + '_tsne_'+clust+'.png', dpi=350)
        plt.close()

if EXOPLORE_RESULTS:
    print 'Starting projection explorer'
    tsne_data = Table.read(file_out+'_tsne.fits')

    class PointSelector:
        def __init__(self, axis, tsne_data_in):
            self.tsne_data = deepcopy(tsne_data_in)
            self.lasso = LassoSelector(axis, self.determine_points)

        def determine_points(self, vertices):
            if len(vertices) > 0:
                vert_path = Path(vertices)
                # determine objects in region
                selected = np.array([tsne_row['source_id'] for tsne_row in self.tsne_data if
                                     vert_path.contains_point((tsne_row['tsne_axis_1'], tsne_row['tsne_axis_2']))], dtype='int64')
                n_selected = len(selected)
                if n_selected > 0:
                    # merge datasets
                    print ','.join([str(s) for s in selected])
                    print 'Selected: ', n_selected
                    data_selection_input = data[np.in1d(data['source_id'], selected)]
                    data_selection_gaia = gaia_data_all[np.in1d(gaia_data_all['source_id'], selected)]
                    print data_selection_input
                    print data_selection_gaia['source_id', 'ra','dec','rv','pmra','pmdec','parallax','feh']
                    # plt.scatter(data_selection_gaia['ra'],data_selection_gaia['dec'])
                    # plt.show()
                    fig = plt.figure()
                    ax = Axes3D(fig)
                    ax.scatter(data_selection_input['X'], data_selection_input['Y'], data_selection_input['Z'])
                    plt.show(fig)
                    plt.close(fig)
                else:
                    print 'Number of points in region is too small'
            else:
                print 'Number of vertices in selection is too small'

    fig, ax = plt.subplots(1, 1)
    ax.scatter(tsne_data['tsne_axis_1'],
               tsne_data['tsne_axis_2'], c='black', lw=0, s=1)
    selector = PointSelector(ax, tsne_data)
    plt.title('tSNE projection of Gaia data or its derivates')
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.show()
    plt.close()

if AUTOMATIC_ANALYSIS:

    match_data = Table.read(galah_dir + 'galah_tgas_xmatch_20180222.csv')
    cannon_data = Table.read(galah_dir + 'sobject_iraf_iDR2_180108_cannon.fits')

    print 'Starting automatic analysis of projection'
    tsne_data = Table.read(file_out + '_tsne.fits')
    print ' DBSCAN'

    # for esp_val in np.arange(0.04, 0.25, 0.02):

    esp_val = 0.04
    db_labels = DBSCAN(eps=esp_val, min_samples=10, n_jobs=n_cpu_dbscan).fit_predict(tsne_data['tsne_axis_1', 'tsne_axis_2'].to_pandas().values)

    idx_plot = db_labels >= 0
    # plot those labels
    plt.scatter(tsne_data['tsne_axis_1'][idx_plot], tsne_data['tsne_axis_2'][idx_plot], lw=0, s=1, c=db_labels[idx_plot], cmap='viridis')
    plt.colorbar()
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.savefig(file_out + '_tsne_dbscan'+str(esp_val)+'.png', dpi=350)
    plt.close()

    # analyse every labeled cluster on its own
    min_obj_label = 5
    max_obj_label = 100
    u_labels = np.unique(db_labels[db_labels >= 0])
    for label in u_labels:
        idx_obj_sel = (db_labels == label)
        n_obj_sel = np.sum(idx_obj_sel)
        print ' Label:', label, 'objects:', n_obj_sel

        if n_obj_sel < min_obj_label or min_obj_label > max_obj_label:
            # TEMP: process only smaller clusters at this time
            continue

        selected_source_ids = tsne_data['source_id'][idx_obj_sel]
        prefix = '{:03.0f}'.format(label)

        # subset of Gaia data
        gaia_data_selected = gaia_data_all[np.in1d(gaia_data_all['source_id'], selected_source_ids)]

        plt.scatter(gaia_data_selected['ra'], gaia_data_selected['dec'], lw=0, s=5, c='black')
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.savefig(prefix+'_1_pos.png', dpi=250)
        plt.close()

        # output list of objects
        csv_out_file = prefix+'_0_raw.csv'
        csv_out_cols = ['source_id', 'ra', 'dec', 'rv', 'pmra', 'pmdec', 'feh', 'phot_g_mean_mag']
        gaia_data_selected[csv_out_cols].write(csv_out_file, format='ascii', comment=False, delimiter='\t',
                                               overwrite=True, fill_values=[(ascii.masked, 'nan')])
        # add some additional data
        txt_out = open(csv_out_file, 'a')
        txt_out.write(' \n')
        txt_out.write(','.join([str(sid) for sid in selected_source_ids]))
        txt_out.close()

        # output analysis plots - orbits
        plot_orbits(gaia_data_selected, path=prefix+'_2_orbits.png')

        # output analysis plots - abundances
        idx_use = np.in1d(match_data['source_id'], selected_source_ids)
        if np.sum(idx_use) > 0:
            selected_sobject_ids = match_data['sobject_id'][idx_use]
            cannon_data_sel = cannon_data[np.in1d(cannon_data['sobject_id'], selected_sobject_ids)]
            plot_abundances_histograms(cannon_data_sel, other_data=None, use_flag=True, path=prefix+'_3_abund.png')
