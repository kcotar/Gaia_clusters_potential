import os, imp
from sklearn.externals import joblib
from astropy.table import Table, join
from isochrones_class import *
from cluster_class import *
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
selected_clusters = ['NGC_6811']
out_dir_suffix = '_1'
rerun = False
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

csv_out_cols_init = ['source_id', 'ra', 'dec', 'rv', 'pmra', 'pmdec', 'phot_g_mean_mag']
csv_out_cols = ['source_id', 'ra', 'dec', 'rv', 'pmra', 'pmdec', 'phot_g_mean_mag', 'time_in_cluster', 'ang_dist', '3d_dist', 'out_cluster_prob']



# ------------------------------------------
# ----------------  SETTINGS  --------------
# ------------------------------------------
# step 1 of the analysis
RV_ZERO = False
RV_MEAN_CLESTER_MEMB = False
SIMULATE_ORBITS = True  # Step 1
QUERY_DATA = True
NO_INTERACTIONS = True
REVERSE_VEL = True
GALAXY_POTENTIAL = True
PKL_SAVE = False
USE_GALPY = True

data_dir = '/data4/cotar/'
cluster_memb_dir = data_dir+'Gaia_open_clusters_analysis_September/'

# read Kharachenko clusters data
cluster_members = Table.read(cluster_memb_dir + 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits')

print 'Reading additional data'
galah_data = Table.read(data_dir+'sobject_iraf_53_reduced_20180327.fits')
gaia_galah_match = Table.read(data_dir+'sobject_iraf_53_gaia.fits')['sobject_id', 'source_id']
galah_data = join(galah_data, gaia_galah_match, keys='sobject_id', join_type='left')
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova_Gaia/isochrones_all.fits', photo_system='Gaia')

output_dir = cluster_memb_dir+'Cluster_orbits_Gaia_DR2_'+out_dir_suffix
os.system('mkdir '+output_dir)
os.chdir(output_dir)

# ------------------------------------------
# ----------------  STEP 1  ----------------
# ------------------------------------------
if SIMULATE_ORBITS:
    out_dir_suffix = '_orbits'
    if RV_ZERO:
        out_dir_suffix += '_zeroRV'
    if RV_MEAN_CLESTER_MEMB:
        out_dir_suffix += '_meanRV'

    # iterate over (pre)selected clusters
    for obs_cluster in selected_clusters:
        print 'Working on:', obs_cluster

        out_dir = obs_cluster + out_dir_suffix

        idx_cluster_pos = np.where(cluster_members['cluster'] == obs_cluster)[0]
        print ' Members found in previous step:',  len(idx_cluster_pos)
        if len(idx_cluster_pos) == 0:
            continue
        clust_data = cluster_members[idx_cluster_pos]

        clust_center = coord.ICRS(ra=np.nanmedian(clust_data['ra']) * un.deg,
                                  dec=np.nanmedian(clust_data['dec']) * un.deg,
                                  distance=np.nanmedian(clust_data['d']) * un.pc)
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
                gaia_data = get_data_subset(np.nanmedian(clust_data['ra']), np.nanmedian(clust_data['dec']),
                                            3.5,
                                            np.nanmedian(clust_data['d']), dist_span=800)
                if len(gaia_data) == 0:
                    os.chdir('..')
                    continue
                gaia_data.write(uotput_file)

        print 'Gaia all:', len(gaia_data)
        if RV_ZERO:
            gaia_data['rv'] = 0.
            idx_members = np.in1d(gaia_data['source_id'], clust_data['source_id'])
            # rough filtering on pm values
            mean_pmra = np.nanmedian(gaia_data['pmra'])
            mean_pmdec = np.nanmedian(gaia_data['pmdec'])
            print 'Median PM values', mean_pmra, mean_pmdec
            idx_pm_use = np.logical_and(np.abs(gaia_data['pmra'] - mean_pmra)<3., np.abs(gaia_data['pmdec'] - mean_pmdec)<3.)
            print ' Will use PM ok objects:', np.sum(idx_pm_use)
            gaia_data = gaia_data[idx_pm_use]
        else:
            if RV_MEAN_CLESTER_MEMB:
                idx_members = np.in1d(gaia_data['source_id'], clust_data['source_id'])
                idx_members_rv = np.logical_and(np.abs(gaia_data['rv']) > 0., idx_members)
                rv_med = np.nanmedian(gaia_data['rv'][idx_members_rv])
                gaia_data['rv'][idx_members] = rv_med
                print 'Median cluster RV:', rv_med

            # complement Gaia data with GALAH RV measurements
            matched_obj = gaia_data[np.in1d(gaia_data['source_id'], galah_data['source_id'])]['source_id']
            print 'Galah objects:', len(matched_obj)
            if len(matched_obj) > 0:
                print 'Adding GALAH RV values to Gaia data'
                for sor_id in matched_obj:
                    idx_gaia = np.where(gaia_data['source_id'] == sor_id)[0]
                    # get and combine galah rv values
                    galah_data_sub = galah_data[galah_data['source_id'] == sor_id]
                    # compute rv statistics as object might have been observed multiple times in GALAH
                    gaia_data['rv'][idx_gaia] = np.nanmedian(galah_data_sub['rv_guess'])
                    gaia_data['rv_error'][idx_gaia] = np.nanmedian(galah_data_sub['e_rv_guess'])
                # output matched GALAH cluster init after RV addition/changes
                output_list_objects(gaia_data[np.logical_and(np.in1d(gaia_data['source_id'], clust_data['source_id']),
                                                             np.in1d(gaia_data['source_id'], galah_data['source_id']))],
                                    clust_center, csv_out_cols_init, 'members_init_galah.csv')

            gaia_data = gaia_data[np.abs(gaia_data['rv']) > 0.]
            print 'Gaia with RV:', len(gaia_data)

            idx_members = np.in1d(gaia_data['source_id'], clust_data['source_id'])
            print 'Gaia in cluster with RV:', np.sum(idx_members)
            print 'GALAH in cluster with RV:', np.sum(np.in1d(galah_data['source_id'], clust_data['source_id']))

            if np.sum(idx_members) < 5:  # at least 3 points are needed for the construction of cluster volume in the xyz coordinate space
                os.chdir('..')
                continue

        # starrt orbits analysis
        min_in_clust_time = 5e6
        pkl_file_test = 'cluster_simulation_run-step1.pkl'  # TEMP: for faster processing and testing
        if not os.path.isfile(pkl_file_test):
            gaia_data_members = gaia_data[idx_members]
            print 'RV members cuts'
            rv_clust_med = np.nanmedian(gaia_data_members['rv'])
            rv_clust_std = np.nanstd(gaia_data_members['rv'])
            plt.hist(gaia_data_members['rv'], bins=50)
            plt.axvline(rv_clust_med, color='black')
            plt.axvline(rv_clust_med - 5., color='black', ls='--')
            plt.axvline(rv_clust_med + 5., color='black', ls='--')
            plt.axvline(rv_clust_med - rv_clust_std, color='red', ls='--')
            plt.axvline(rv_clust_med + rv_clust_std, color='red', ls='--')
            plt.tight_layout()
            plt.savefig('members_init_rv.png', dpi=200)
            plt.close()
            # filter outlying rv values
            gaia_data_members = gaia_data_members[np.logical_and(gaia_data_members['rv'] >= rv_clust_med - 5.,
                                                                 gaia_data_members['rv'] <= rv_clust_med + 5.)]
            if len(gaia_data_members) < 5:
                print 'Not enough members after RV cuts'
                os.chdir('..')
                continue

            print 'Step 1 integration'
            output_list_objects(gaia_data_members, clust_center, csv_out_cols_init, 'members_init.csv')
            # TODO: age and meh for those clusters
            cluster_class = CLUSTER(meh=-0.1, age=200e6, isochrone=iso, id=obs_cluster, reverse=REVERSE_VEL)
            cluster_class.init_members(gaia_data_members)
            cluster_class.init_background(gaia_data[~idx_members])
            cluster_class.plot_cluster_xyz(path=obs_cluster + '_stanje_zac.png')

            gaia_test_stars_data = gaia_data[~idx_members]
            gaia_cluster_sub_ra_dec = coord.ICRS(ra=gaia_test_stars_data['ra'] * un.deg,
                                                 dec=gaia_test_stars_data['dec'] * un.deg,
                                                 distance=1e3 / gaia_test_stars_data['parallax'] * un.pc)

            # all stars in a sphere around cluster center
            idx_dist_cond = gaia_cluster_sub_ra_dec.separation_3d(clust_center) < 800 * un.pc
            # have low galactic velocity difference towards median galactic velocity
            vel_clust_gal = np.nanmedian(cluster_class.members['d_x','d_y','d_z'].to_pandas().values, axis=0)
            print vel_clust_gal
            vel_diff_test_gal = np.sqrt(np.sum((cluster_class.members_background['d_x','d_y','d_z'].to_pandas().values - vel_clust_gal)**2, axis=1))
            print 'median diff', np.mean(vel_diff_test_gal), np.median(vel_diff_test_gal)
            # plt.hist(vel_diff_test_gal, bins=200)
            # plt.axvline((25. * un.km / un.s).to(un.pc / un.yr).value, color='black')
            # plt.show()
            # plt.close()
            idx_vel_condition = vel_diff_test_gal <= (20. * un.km / un.s).to(un.pc / un.yr).value
            # combine conditions
            idx_test = np.logical_and(idx_dist_cond, idx_vel_condition)

            n_test = np.sum(idx_test)
            print 'Number of all stars in cluster vicinity:', len(cluster_class.members_background)
            print 'Number of test stars in cluster vicinity:', n_test
            if n_test <= 0:
                print '  WARNING: Not enough stars in vicinity to perform any orbit integration simulation.'
                os.chdir('..')
                continue

            cluster_class.init_test_particle(gaia_test_stars_data[idx_test])
            if USE_GALPY:
                # cluster_class.galpy_run_all(members=True, particles=True, total_time=-220e6, step_years=1e4)
                in_clust_prob = cluster_class.galpy_mutirun_all(members=True, particles=True, total_time=-220e6, step_years=3e4,
                                                n_runs=100, perc_in=25., min_in_time=min_in_clust_time)
            else:
                cluster_class.integrate_particle(220e6, step_years=1e4, include_galaxy_pot=GALAXY_POTENTIAL,
                                                 integrate_stars_pos=True, integrate_stars_vel=True,
                                                 disable_interactions=NO_INTERACTIONS)

            cluster_class.determine_orbits_that_cross_cluster()
            if PKL_SAVE:
                joblib.dump(cluster_class, pkl_file_test)
        else:
            cluster_class = joblib.load(pkl_file_test)
        #cluster_class.plot_cluster_xyz_movement(path='orbits_integration_all-step1.png')

        # output results to file
        idx_probable_in = cluster_class.particle['time_in_cluster'] * 1e6 > min_in_clust_time
        pos_crossing_particles = cluster_class.particle[idx_probable_in]
        print 'Number of possible crossing stars:', len(pos_crossing_particles)
        print 'Number of possible GALAH crossing stars:', np.sum(np.in1d(pos_crossing_particles['source_id'], clust_data['source_id']))

        # HR diagrams and filtering
        # cluster_hr = HR_DIAGRAM(gaia_data[idx_members], khar_clusters_sel, isochrone=iso, photo_system='Gaia')
        # cluster_hr.plot_HR_diagram(include_isocrone=True, include_rv=True, path='hr_diagram_1_members.png')
        # cluster_hr = HR_DIAGRAM(pos_crossing_particles, khar_clusters_sel, isochrone=iso, photo_system='Gaia')
        # cluster_hr.plot_HR_diagram(include_isocrone=True, include_rv=True, path='hr_diagram_2_possible.png')

        output_list_objects(pos_crossing_particles, clust_center, csv_out_cols, 'possible_ejected-step1.csv')
        output_list_objects(cluster_class.particle[~idx_probable_in], clust_center, csv_out_cols, 'possible_outside-step1.csv')
        #cluster_class.plot_cluster_xyz(path=obs_cluster + '_possible_ejected-step1.png', show_possible=True)
	os.chdir('..')
