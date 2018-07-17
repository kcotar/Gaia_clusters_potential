import os, imp
from sklearn.externals import joblib
from astropy.table import Table
from isochrones_class import *
from cluster_class import *
from cluster_members_class import *
from abundances_analysis import *
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from sklearn import mixture
from matplotlib.colors import LogNorm
from gaia_data_queries import *
imp.load_source('hr_class', '../Binaries_clusters/HR_diagram_class.py')
from hr_class import *

# step 1 of the analysis
MEMBER_DETECTION = True  # Step 1
ORBITS_ANALYSIS = False  # Step 2
USE_UPDATED_KHAR = False

# step 2 of the analysis
RV_USE = True
RV_ONLY = False
NO_INTERACTIONS = False
REVERSE_VEL = True
GALAXY_POTENTIAL = True
QUERY_DATA = True

data_dir = '/home/klemen/data4_mount/'
khar_dir = data_dir + 'clusters/Kharchenko_2013/'

csv_out_cols_init = ['source_id', 'ra', 'dec', 'rv', 'pmra', 'pmdec', 'phot_g_mean_mag']
csv_out_cols = ['source_id', 'ra', 'dec', 'rv', 'pmra', 'pmdec', 'phot_g_mean_mag', 'time_in_cluster', 'ang_dist', '3d_dist']

# read Kharachenko clusters data
if USE_UPDATED_KHAR:
    clusters = Table.read(khar_dir + 'catalog_tgas_update.csv')
else:
    clusters = Table.read(khar_dir + 'catalog.csv')

print 'Reading additional data'
galah_data = Table.read(data_dir+'sobject_iraf_53_reduced_20180327.fits')
gaia_galah_xmatch = Table.read(data_dir+'sobject_iraf_53_gaia.fits')['sobject_id', 'source_id']
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova_Gaia/isochrones_all.fits', photo_system='Gaia')

selected_clusters_Asiago = ['NGC_2264', 'NGC_2281', 'NGC_2301', 'NGC_2548', 'NGC_2632', 'NGC_2682',
                            'Melotte_111', 'Mamajek_2', 'IC_4665', 'Collinder_350', 'Collinder_359',
                            'NGC_6633', 'IC_4756', 'Stephenson_1',
                            'NGC_6738', 'NGC_6793', 'Stock_1',
                            'Turner_9', 'Roslund_1',
                            'NGC_6828', 'Roslund_5', 'NGC_6882',
                            'Roslund_6',
                            'NGC_6940', 'FSR_0251', 'Alessi_12', 'Roslund_7', 'FSR_0261',
                            'NGC_6991A', 'NGC_6997', 'Basel_15', 'NGC_7058',
                            'NGC_7063', 'NGC_7092', 'IC_1396', 'IC_5146', 'NGC_7160',
                            'NGC_7243', 'Teutsch_39']

cluster_fits_out = 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits'

output_dir = 'Khar_cluster_initial_Gaia_DR2_500pc'
os.system('mkdir '+output_dir)
os.chdir(output_dir)


# ------------------------------------------
# ----------------  Functions  -------------
# ------------------------------------------
def fill_table(in_data, cluster, cols, cols_data):
    out_data = deepcopy(in_data)
    idx_l = np.where(out_data['cluster'] == cluster)[0]
    for i_v, col in enumerate(cols):
        out_data[col][idx_l] = cols_data[i_v]
    return out_data

# ------------------------------------------
# ----------------  STEP 1  ----------------
# ------------------------------------------
cluster_params_table_fits = os.getcwd() + '/cluster_params.fits'
if os.path.isfile(cluster_params_table_fits):
    # read existing table
    cluster_params_table = Table.read(cluster_params_table_fits)
else:
    # create new table with cluster parameters
    cluster_params_table = Table(names=('cluster', 'ra_c', 'e_ra_c', 'dec_c', 'e_dec_c', 'pmra', 'e_pmra', 'pmdec', 'e_pmdec', 'th_pmra', 'rv', 'e_rv', 'dist', 'e_dist'),
                                 dtype=np.hstack(('S25', np.full(13, 'float64'))))

if MEMBER_DETECTION:
    out_dir_suffix = '_member_sel'
    cluster_obj_found_out = Table(names=('source_id', 'cluster'), dtype=('int64', 'S30'))

    # iterate over (pre)selected clusters
    # for obs_cluster in np.unique(clusters['Cluster']):
    for obs_cluster in selected_clusters_Asiago[:5]:
    # for obs_cluster in ['NGC_1252','NGC_6994','NGC_7772','NGC_7826','NGC_1901']:
        print 'Working on:', obs_cluster

        if np.sum(cluster_params_table['cluster'] == obs_cluster) > 0:
            print 'Already processed'
            continue
        else:
            # add dummy row to the data that will be fill during the analysis
            row_empty = [obs_cluster]
            for ire in range(len(cluster_params_table.colnames)-1):
                row_empty.append(np.nan)
            cluster_params_table.add_row(row_empty)

        out_dir = obs_cluster + out_dir_suffix

        idx_cluster_pos = np.where(clusters['Cluster'] == obs_cluster)[0]
        if len(idx_cluster_pos) == 0:
            continue
        clust_data = clusters[idx_cluster_pos]
        print ' Basic info -->', 'r1:', clust_data['r1'].data[0], 'r2:', clust_data['r2'].data[0], 'pmra:', clust_data['pmRAc'].data[0], 'pmdec:', clust_data['pmDEc'].data[0]

        clust_center = coord.ICRS(ra=clust_data['RAdeg'] * un.deg,
                                  dec=clust_data['DEdeg'] * un.deg,
                                  distance=clust_data['d'] * un.pc)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        os.chdir(out_dir)

        if QUERY_DATA:
            uotput_file = 'gaia_query_data.csv'
            if os.path.isfile(uotput_file):
                gaia_data = Table.read(uotput_file)
            else:
                print ' Sending QUERY to download Gaia data'
                # limits to retrieve Gaia data
                gaia_data = get_data_subset(clust_data['RAdeg'].data[0], clust_data['DEdeg'].data[0],
                                            clust_data['r2'].data[0] * 2.,
                                            clust_data['d'].data[0], dist_span=500)
                if len(gaia_data) == 0:
                    os.chdir('..')
                    continue
                gaia_data.write(uotput_file)
            gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                                     dec=gaia_data['dec'] * un.deg,
                                     distance=1e3 / gaia_data['parallax'] * un.pc)

        # processing limits
        idx_possible_r2 = gaia_ra_dec.separation(clust_center) < clust_data['r2'] * 1.5 * un.deg
        gaia_cluster_sub_r2 = gaia_data[idx_possible_r2]
        idx_distance = np.abs(1e3/gaia_cluster_sub_r2['parallax'] - clust_data['d']) < 400  # for now because of uncertain distances
        gaia_cluster_sub_r2 = gaia_cluster_sub_r2[idx_distance]

        n_in_selection = len(gaia_cluster_sub_r2)
        if n_in_selection < 40:
            print ' WARNING: Not enough objects in selection ('+str(n_in_selection)+')'
            cluster_obj_found_out.write(cluster_params_table_fits, overwrite=True)
            os.chdir('..')
            continue

        # create cluster class and plot all data
        find_members_class = CLUSTER_MEMBERS(gaia_cluster_sub_r2, clust_data)
        find_members_class.plot_on_sky(path='cluster_pos.png', mark_objects=False)

        print ' Multi radius Gaussian2D density fit'
        pm_median_all = [np.nanmedian(gaia_data['pmra']), np.nanmedian(gaia_data['pmdec'])]
        for c_rad in [np.float64(clust_data['r2'])]:  # np.linspace(np.float64(clust_data['r1']), np.float64(clust_data['r2']), 2):
            cluster_density_param = find_members_class.perform_selection_density(c_rad, suffix='_{:.3f}'.format(c_rad), n_runs=2)

        # check if cluster was detected
        if np.sum(np.isfinite(cluster_density_param)) == 0:
            print ' WARNING: Cluster not recognized from PM data'
            cluster_obj_found_out.write(cluster_params_table_fits, overwrite=True)
            os.chdir('..')
            continue
        else:
            # fill table with relevant data
            cluster_params_table = fill_table(cluster_params_table, obs_cluster,
                                              ['pmra', 'pmdec', 'e_pmra', 'e_pmdec', 'th_pmra'],
                                              cluster_density_param[1:])

        # continue with the processing
        find_members_class.plot_selection_density_pde(path='cluster_pos_pde.png')

        print cluster_params_table
        os.chdir('..')
        continue

        find_members_class.plot_members(25, path='cluster_pm_multi_sel.png')
        find_members_class.plot_members(25, path='cluster_pm_multi_n.png', show_n_sel=True)
        find_members_class.plot_selected_hist(path='selection_hist.png')
        clust_ok = find_members_class.refine_distances_selection(out_plot=True, path='cluster_parsec.png')
        if clust_ok:
            # elipse fitting to the data and search for additional members inside it, discard distant stars in hull
            find_members_class.include_iniside_hull(distance_limits=True, manual_hull=False)
            find_members_class.plot_members(path='cluster_pm_multi_sel_final.png', show_final=True)
            # possible RV refinements
            clust_ok = find_members_class.refine_distances_selection_RV(out_plot=True, path='cluster_rv.png')
            # get and store member results
            members_source_id = find_members_class.get_cluster_members(recompute=False, idx_only=False)
            find_members_class.plot_on_sky(path='cluster_pos_final.png', mark_objects=True)
            # add members to the resulting file
            if np.isscalar(members_source_id):
                os.chdir('..')
                continue

            for m_s_id in members_source_id:
                cluster_obj_found_out.add_row([m_s_id, obs_cluster])
            # final members selection
            gaia_cluster_members_final = gaia_cluster_sub_r2[find_members_class.get_cluster_members(recompute=False, idx_only=True)]

            # create HR diagram
            output_list_objects(gaia_cluster_members_final, clust_center, csv_out_cols_init, 'memebers_data.csv')
            cluster_hr = HR_DIAGRAM(gaia_cluster_members_final, clust_data[clust_data['Cluster'] == obs_cluster], isochrone=iso, photo_system='Gaia')
            cluster_hr.plot_HR_diagram(include_isocrone=True, include_rv=True, path='hr_diagram.png')

        os.chdir('..')
        print ''  # nicer looking output with blank lines

    # save cluster results
    cluster_obj_found_out.write(cluster_fits_out, format='fits', overwrite=True)
    # save only ra/dec information for determined objects
    gaia_data[np.in1d(gaia_data['source_id'], cluster_obj_found_out['source_id'])]['source_id', 'ra', 'dec'].write(cluster_fits_out[:-5]+'_pos.fits', format='fits', overwrite=True)

