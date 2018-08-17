import os, imp
from sklearn.externals import joblib
from astropy.table import Table
from isochrones_class import *
#from cluster_class import *
from cluster_members_class import *
from abundances_analysis import *
from sklearn import mixture
from gaia_data_queries import *
imp.load_source('hr_class', '../Binaries_clusters/HR_diagram_class.py')
from hr_class import *
from sys import argv
from getopt import getopt


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
# ----------------  INPUTS  ----------------
# ------------------------------------------
selected_clusters = ['NGC_6940']
out_dir_suffix = '_11'
rerun = True
if len(argv) > 1:
    # parse input options
    opts, args = getopt(argv[1:], '', ['clusters=', 'suffix=', 'rerun='])
    # set parameters, depending on user inputs
    print opts
    for o, a in opts:
        if o == '--clusters':
            selected_clusters = a.split(',')
        if o == '--suffix':
            out_dir_suffix = str(a)
        if o == '--rerun':
            rerun = int(a) > 0



# ------------------------------------------
# ----------------  SETTINGS  --------------
# ------------------------------------------
# step 1 of the analysis
MEMBER_DETECTION = True  # Step 1
QUERY_DATA = True

data_dir = '/home/klemen/data4_mount/'
khar_dir = data_dir + 'clusters/Kharchenko_2013/'

# read Kharachenko clusters data
clusters = Table.read(khar_dir + 'catalog.csv')

print 'Reading additional data'
galah_data = Table.read(data_dir+'sobject_iraf_53_reduced_20180327.fits')
gaia_galah_xmatch = Table.read(data_dir+'sobject_iraf_53_gaia.fits')['sobject_id', 'source_id']
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova_Gaia/isochrones_all.fits', photo_system='Gaia')

cluster_fits_out = 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits'

output_dir = 'Khar_cluster_initial_Gaia_DR2_'+out_dir_suffix
os.system('mkdir '+output_dir)
os.chdir(output_dir)

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
    cluster_obj_found_out = Table(names=('source_id', 'cluster', 'ra', 'dec', 'd'), dtype=('int64', 'S30', 'float64', 'float64', 'float64'))

    # iterate over (pre)selected clusters
    for obs_cluster in selected_clusters:
        print 'Working on:', obs_cluster

        if np.sum(cluster_params_table['cluster'] == obs_cluster) > 0:
            print 'Already processed to some point'
            if not rerun:
                continue
        else:
            # add dummy row to the data that will be filled during the analysis
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
                                            clust_data['d'].data[0], dist_span=1e3)
                if len(gaia_data) == 0:
                    os.chdir('..')
                    continue
                gaia_data.write(uotput_file)
            gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                                     dec=gaia_data['dec'] * un.deg,
                                     distance=1e3 / gaia_data['parallax'] * un.pc)

        # processing limits
        idx_possible_r2 = gaia_ra_dec.separation(clust_center) < clust_data['r2'] * 1.3 * un.deg
        gaia_cluster_sub_r2 = gaia_data[idx_possible_r2]
        idx_distance = np.abs(1e3/gaia_cluster_sub_r2['parallax'] - clust_data['d']) < 900  # for now because of uncertain distances
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

        # check if proper motion properties of the cluster were already established
        print ' Multi radius Gaussian2D density fit'
        pm_median_all = [np.nanmedian(gaia_data['pmra']), np.nanmedian(gaia_data['pmdec'])]
        for c_rad in [np.float64(clust_data['r2'])]:
            cluster_density_param = find_members_class.perform_selection_density(c_rad, suffix='_{:.3f}'.format(c_rad), n_runs=7)

        # check if cluster was detected
        if np.sum(np.isfinite(cluster_density_param)) == 0:
            print ' WARNING: Cluster not recognizable from PM data'
            cluster_params_table.write(cluster_params_table_fits, overwrite=True)
            os.chdir('..')
            continue
        else:
            # fill table with relevant data
            cluster_params_table = fill_table(cluster_params_table, obs_cluster,
                                              ['pmra', 'pmdec', 'e_pmra', 'e_pmdec', 'th_pmra'],
                                              cluster_density_param[1:])

        # continue with the processing
        print 'PM0:', find_members_class.cluster_g2d_params
        find_members_class.plot_cluster_members_pmprob(path='cluster_sel_1_pm.png', max_sigma=1.)
        find_members_class.plot_cluster_members_pmprob(path='cluster_sel_1_pm_stdreg.png', plot_std_regions=True)
        p_m, p_s = find_members_class.initial_distance_cut(path='cluster_sel_2_parasec.png', max_sigma=3.)
        # TODO: Decision based oon p_s

        n_sel_s1 = np.sum(find_members_class.selected_final)
        print 'PM selected in first step:', n_sel_s1
        if n_sel_s1 < 10:
            print ' WARNING: Cluster has a low number of probable members'
            os.chdir('..')
            continue

        find_members_class.update_selection_parameters(sel_limit=None)
        find_members_class.plot_selection_density_pde(path='cluster_pos_pde_pm.png')
        # TODO: possible to determine min_perc using histogram analysis, something similar to Otsu algorithm

        # complete selection plot four in one
        find_members_class.plot_selected_complete(path='params_0_1.png')
        find_members_class.plot_selected_complete_dist(path='params_0_2.png')

        n_stars_init = np.sum(find_members_class.selected_final)
        d_stars_init = 1e5
        sel_sigma = 1.5
        for i_i in range(1, 50):
            find_members_class.update_selection_parameters(prob_sigma=sel_sigma)  # sel_limit=sel_thr)
            # complete selection plot four in one
            n_stars_curr = np.sum(find_members_class.selected_final)
            if n_stars_curr <= 0:
                break
            find_members_class.plot_selected_complete(path='params_'+str(i_i)+'_1.png')
            find_members_class.plot_selected_complete_dist(path='params_'+str(i_i)+'_2.png')
            print 'PM :', find_members_class.cluster_g2d_params
            d_stars_curr = n_stars_init - n_stars_curr
            print '  ', i_i, n_stars_curr, d_stars_curr, sel_sigma

            if d_stars_curr == 0:
                # algorithm converged to a solution
                print '  Converge to a constant number of stars'
                break
            if 1.*d_stars_curr/n_stars_curr < -0.1:
                if 1.*d_stars_curr/n_stars_curr < -0.2:
                    # algorithm starts adding too many new stars to the cluster
                    print '  Too many new stars added to the cluster'
                    break
                else:
                    sel_sigma += 0.1
                    sel_sigma = min(sel_sigma, 2.0)

            d_stars_init = d_stars_curr
            n_stars_init = n_stars_curr

        # save members to a Table file that will written out
        for m_s_id in find_members_class.data[find_members_class.selected_final]:
            cluster_obj_found_out.add_row([m_s_id['source_id'], obs_cluster, m_s_id['ra'], m_s_id['dec'], m_s_id['parsec']])

        os.chdir('..')
        # save cluster results
        cluster_obj_found_out.write(cluster_fits_out, format='fits', overwrite=True)
        # save only ra/dec information for determined objects
        gaia_data[np.in1d(gaia_data['source_id'], cluster_obj_found_out['source_id'])]['source_id', 'ra', 'dec'].write(
            cluster_fits_out[:-5] + '_pos.fits', format='fits', overwrite=True)
        # TODO: add final cluster parameters to the parameters table and fits file
        cluster_params_table.write(cluster_params_table_fits, overwrite=True)
        # cluster_params_table = fill_table(cluster_params_table, obs_cluster,
        #                                   ['pmra', 'pmdec', 'e_pmra', 'e_pmdec', 'th_pmra'],
        #                                   cluster_density_param[1:])

