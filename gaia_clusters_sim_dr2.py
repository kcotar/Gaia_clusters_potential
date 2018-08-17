import os, imp
from sklearn.externals import joblib
from astropy.table import Table
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
csv_out_cols = ['source_id', 'ra', 'dec', 'rv', 'pmra', 'pmdec', 'phot_g_mean_mag', 'time_in_cluster', 'ang_dist', '3d_dist']



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

data_dir = '/home/klemen/data4_mount/'
Cluster_memb_dir = '/home/klemen/Gaia_clusters_potential/'

# read Kharachenko clusters data
cluster_members = Table.read(Cluster_memb_dir + 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits')

print 'Reading additional data'
galah_data = Table.read(data_dir+'sobject_iraf_53_reduced_20180327.fits')
gaia_galah_xmatch = Table.read(data_dir+'sobject_iraf_53_gaia.fits')['sobject_id', 'source_id']
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova_Gaia/isochrones_all.fits', photo_system='Gaia')

output_dir = 'Cluster_orbits_Gaia_DR2_'+out_dir_suffix
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
                                            np.nanmedian(clust_data['d']), dist_span=550)
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
            gaia_data = gaia_data[np.abs(gaia_data['rv']) > 0.]
            print 'Gaia with RV:', len(gaia_data)

            idx_members = np.in1d(gaia_data['source_id'], clust_data['source_id'])
            print 'Gaia in cluster with RV:', np.sum(idx_members)

            if np.sum(idx_members) < 5:  # at least 3 points are needed for the construction of cluster volume in the xyz coordinate space
                os.chdir('..')
                continue

        # starrt orbits analysis
        pkl_file_test = 'cluster_simulation_run-step1.pkl'  # TEMP: for faster processing and testing
        if not os.path.isfile(pkl_file_test):
            print 'Step 1 integration'
            output_list_objects(gaia_data[idx_members], clust_center, csv_out_cols_init, 'members_init.csv')
            # TODO: age and meh for those clusters
            cluster_class = CLUSTER(meh=-0.1, age=200e6, isochrone=iso, id=obs_cluster, reverse=REVERSE_VEL)
            cluster_class.init_members(gaia_data[idx_members])
            cluster_class.init_background(gaia_data[~idx_members])
            cluster_class.plot_cluster_xyz(path=obs_cluster + '_stanje_zac.png')

            gaia_cluster_sub_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                                                 dec=gaia_data['dec'] * un.deg,
                                                 distance=1e3 / gaia_data['parallax'] * un.pc)

            idx_test = gaia_cluster_sub_ra_dec.separation_3d(
                clust_center) < 750 * un.pc  # all stars in a sphere around cluster center
            idx_test = np.logical_and(idx_test, ~idx_members)
            test_stars = gaia_data[idx_test]
            print 'Number of test stars in cluster vicinity:', len(test_stars)

            cluster_class.init_test_particle(test_stars)
            if USE_GALPY:
                cluster_class.galpy_run_all(members=True, particles=True, total_time=-220e6, step_years=1e4)
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

        min_in_clust_time = 4e6
        # output results to file
        pos_crossing_particles = cluster_class.particle[cluster_class.particle['time_in_cluster'] * 1e6 > min_in_clust_time]
        print 'Number of possible crossing stars:', len(pos_crossing_particles)

        # HR diagrams and filtering
        # cluster_hr = HR_DIAGRAM(gaia_data[idx_members], khar_clusters_sel, isochrone=iso, photo_system='Gaia')
        # cluster_hr.plot_HR_diagram(include_isocrone=True, include_rv=True, path='hr_diagram_1_members.png')
        # cluster_hr = HR_DIAGRAM(pos_crossing_particles, khar_clusters_sel, isochrone=iso, photo_system='Gaia')
        # cluster_hr.plot_HR_diagram(include_isocrone=True, include_rv=True, path='hr_diagram_2_possible.png')

        output_list_objects(pos_crossing_particles, clust_center, csv_out_cols, 'possible_ejected-step1.csv')
        #cluster_class.plot_cluster_xyz(path=obs_cluster + '_possible_ejected-step1.png', show_possible=True)
	os.chdir('..')
