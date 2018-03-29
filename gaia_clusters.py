import os
from sklearn.externals import joblib
from isochrones_class import *
from cluster_class import *
from abundances_analysis import *
from scipy.stats import circmean

RV_USE = True
RV_ONLY = False
NO_INTERACTIONS = True
REVERSE_VEL = True
GALAXY_POTENTIAL = True
PKL_SAVE = False
USE_GALPY = True
csv_out_cols = ['source_id', 'ra', 'dec', 'rv_orig', 'feh', 'pmra', 'pmdec', 'phot_g_mean_mag', 'time_in_cluster', 'ang_dist']

data_dir = '/home/klemen/data4_mount/'
# read Kharachenko clusters data
clusters = Table.read(data_dir+'clusters/Gaia_2017/tabled.csv')
# read Tgas data set
if RV_USE:
    gaia_data = Table.read(data_dir + 'TGAS_data_set/TgasSource_all_with_rv_feh.fits')
    if RV_ONLY:
        gaia_data = gaia_data[np.logical_and(gaia_data['rv'] != 0, gaia_data['e_rv'] != 0)]
else:
    gaia_data = Table.read(data_dir+'TGAS_data_set/TgasSource_all.fits')

# rv bacukp
gaia_data['rv_orig'] = gaia_data['rv']

# read cannon and galah data for the last stages of analysis
cannon_data = Table.read(data_dir+'sobject_iraf_iDR2_180108_cannon.fits')
gaia_galah_xmatch = Table.read(data_dir+'galah_tgas_xmatch_20171111.csv')['sobject_id', 'source_id']

# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova_Gaia/isochrones_all.fits')

gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                         dec=gaia_data['dec'] * un.deg,
                         distance=1e3/gaia_data['parallax'] * un.pc)

os.chdir('Gaia_cluster_stars')
# iterate over preselected clusters
for obs_cluster in np.unique(clusters['Cluster']):
    print 'Working on:', obs_cluster

    out_dir = obs_cluster
    if RV_USE:
        out_dir += '_with_rv'
        if RV_ONLY:
            out_dir += '_only'
    if NO_INTERACTIONS:
        out_dir += '_nointer'
    if REVERSE_VEL:
        out_dir += '_reversevel'
    if not GALAXY_POTENTIAL:
        out_dir += '_nogalpot'
    out_dir += '_GAIA-init'
    out_dir += '_my-integ'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    os.chdir(out_dir)

    clust_data = clusters[clusters['Cluster'] == obs_cluster]
    print 'N initial reference set:', len(clust_data)
    clust_center = coord.ICRS(ra=circmean(clust_data['RAdeg'], low=0, high=360) * un.deg,
                              dec=circmean(clust_data['DEdeg'], low=-90, high=90) * un.deg)

    print clust_center
    # define possible cluster stars
    idx_possible_r1 = gaia_ra_dec.separation(clust_center) < 10. * un.deg  # everything in radius 10 deg around cluster
    gaia_cluster_sub = gaia_data[idx_possible_r1]
    print np.sum(idx_possible_r1)
    # kinematics selection

    idx_members = np.in1d(gaia_cluster_sub['source_id'], clust_data['Source'])
    if np.sum(idx_members) == 0:
        print 'No match with input catalog'

    # refine selection if needed, add some plots
    gaia_cluster_memb = gaia_cluster_sub[idx_members]
    d_val_param = [5, 0.4, 3, 3]
    col_param = ['rv', 'feh', 'pmra', 'pmdec']

    # define objects that might be removed based on their parameters
    idx_memb_ok_params = np.full(len(gaia_cluster_memb), True)
    for i_c in range(len(col_param)):
        col = col_param[i_c]
        idx_ok_vals = np.isfinite(gaia_cluster_memb[col])
        median_val = np.nanmedian(gaia_cluster_memb[col][idx_ok_vals])
        if np.sum(idx_ok_vals) == 0:
            continue
        # possible outlyers
        lim_val = d_val_param[i_c]
        if 'pm' in col:
            if np.sum(idx_ok_vals) > 3:
                lim_val = 3*np.nanstd(gaia_cluster_memb[col][idx_ok_vals])

        idx_outlyer = np.logical_and(idx_ok_vals,
                                     np.abs(gaia_cluster_memb[col] - median_val) > lim_val)
        idx_memb_ok_params[idx_outlyer] = False
        # plots
        plt.hist(gaia_cluster_memb[col][idx_ok_vals], bins=50)
        plt.axvline(x=median_val, color='black', ls='-')
        plt.axvline(x=median_val-lim_val, color='black', ls='--')
        plt.axvline(x=median_val+lim_val, color='black', ls='--')
        plt.title(col+'     all:'+str(len(gaia_cluster_memb))+'     valid:'+str(np.sum(idx_ok_vals)))
        plt.savefig('memb_stat_'+col+'.png', dpi=200)
        plt.close()

    # refine using parameters selection
    if np.sum(np.logical_not(idx_memb_ok_params)) >= 1:
        gaia_cluster_memb[np.logical_not(idx_memb_ok_params)]['source_id', 'ra', 'dec', 'rv', 'feh', 'pmra', 'pmdec'].write('memb_removed_check.txt', format='ascii', comment=False, delimiter='\t', overwrite=True, fill_values=[(ascii.masked, 'nan')])
        gaia_cluster_memb = gaia_cluster_memb[idx_memb_ok_params]

    # redefine including the distance
    clust_center = coord.ICRS(ra=circmean(gaia_cluster_memb['ra'], low=0, high=360) * un.deg,
                              dec=circmean(gaia_cluster_memb['dec'], low=-90, high=90) * un.deg,
                              distance=np.median(1e3/gaia_cluster_memb['parallax']) * un.pc)

    idx_members = np.in1d(gaia_cluster_sub['source_id'], gaia_cluster_memb['source_id'])
    print 'N reference after limiting:', np.sum(idx_members)

    if len(gaia_cluster_memb) < 5:
        print ' Too few members'
        os.chdir('..')
        continue

    # determine Galah and cannon members data
    cluster_members_galah = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], gaia_cluster_sub[idx_members]['source_id'])]['sobject_id']
    cannon_cluster_data = cannon_data[np.in1d(cannon_data['sobject_id'], cluster_members_galah)]

    # processing without radial velocity - set rv to 0 for all stars
    gaia_cluster_sub['rv'] = 0.

    pkl_file_test = 'cluster_simulation_run-step1.pkl'  # TEMP: for faster processing and testing
    if not os.path.isfile(pkl_file_test):
        print 'Step 1 integration'
        # TODO: age and meh for those clusters
        cluster_class = CLUSTER(meh=0.0, age=10 ** 7, isochrone=iso, id=obs_cluster, reverse=REVERSE_VEL)
        cluster_class.init_members(gaia_cluster_sub[idx_members])
        cluster_class.init_background(gaia_cluster_sub[~idx_members])
        cluster_class.plot_cluster_xyz(path=obs_cluster + '_stanje_zac.png')

        gaia_cluster_sub_ra_dec = coord.ICRS(ra=gaia_cluster_sub['ra'] * un.deg,
                                             dec=gaia_cluster_sub['dec'] * un.deg,
                                             distance=1e3 / gaia_cluster_sub['parallax'] * un.pc)

        idx_test = gaia_cluster_sub_ra_dec.separation_3d(clust_center) < 200 * un.pc  # all stars in a sphere around cluster center
        idx_test = np.logical_and(idx_test, ~idx_members)
        test_stars = gaia_cluster_sub[idx_test]
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
    cluster_class.plot_cluster_xyz_movement(path='orbits_integration_all-step1.png')

    min_in_clust_time = 2e6
    # output results to file
    pos_crossing_particles = cluster_class.particle[cluster_class.particle['time_in_cluster']*1e6 > min_in_clust_time]
    output_list_objects(pos_crossing_particles, clust_center, csv_out_cols, 'possible_ejected-step1.csv')
    cluster_class.plot_cluster_xyz(path=obs_cluster + '_possible_ejected-step1.png', show_possible=True)

    # re-run everything with objects that have RV data
    gaia_cluster_sub['rv'] = gaia_cluster_sub['rv_orig']
    member_stars = gaia_cluster_sub[np.logical_and(idx_members,
                                                   np.isfinite(gaia_cluster_sub['rv_orig']))]
    observ_stars = gaia_cluster_sub[np.logical_and(np.in1d(gaia_cluster_sub['source_id'], pos_crossing_particles['source_id']),
                                                   np.isfinite(gaia_cluster_sub['rv_orig']))]
    print 'With rv - after step 1:', len(member_stars), len(observ_stars)
    if len(member_stars) >= 5 and len(observ_stars) >= 1:
        print 'Step 2 integration'
        cluster_class.init_members(member_stars)
        cluster_class.init_test_particle(observ_stars)
        if USE_GALPY:
            cluster_class.galpy_run_all(members=True, particles=True, total_time=-220e6, step_years=1e4)
        else:
            cluster_class.integrate_particle(220e6, step_years=1e4, include_galaxy_pot=GALAXY_POTENTIAL,
                                             integrate_stars_pos=True, integrate_stars_vel=True,
                                             disable_interactions=NO_INTERACTIONS)
        cluster_class.determine_orbits_that_cross_cluster()
        pos_crossing_particles = cluster_class.particle[cluster_class.particle['time_in_cluster'] > min_in_clust_time]
        # plot all orbits
        cluster_class.plot_cluster_xyz_movement(path='orbits_integration_all-step2.png')
        print ' Crossing in step 2:', len(pos_crossing_particles)
        output_list_objects(pos_crossing_particles, clust_center, csv_out_cols, 'possible_ejected-step2.csv')
        cluster_class.plot_cluster_xyz(path=obs_cluster + '_possible_ejected-step2.png', show_possible=True)


    # # video and plot outputs
    # for sou_id in possible_ejected:
    #
    #     galah_match_data = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], sou_id)]
    #     if len(galah_match_data) == 1:
    #         sob_id = galah_match_data['sobject_id'].data[0]
    #     else:
    #         sob_id = 0
    #
    #     suffix = str(sob_id) + '_' + str(sou_id)
    #     print 'Output results for:', suffix
    #     cluster_class.plot_cluster_xyz_movement(path=suffix + '_orbit.png', source_id=sou_id)
    #     # cluster_class.animate_particle_movement(path=suffix + '_video.mp4', source_id=sou_id, t_step=0.5e6)
    #
    #     # cannon_observed_data = cannon_data[np.in1d(cannon_data['sobject_id'], sob_id)]
    #     # if len(cannon_observed_data) == 1:
    #     #     # it is also possible to create an abundance plot for Galah stars in Cannon dataset
    #     #     cluster_class.animate_particle_movement(path=suffix + '_video.mp4', source_id=sou_id, t_step=0.5e6)
    #     #     plot_abundances_histograms(cannon_cluster_data, cannon_observed_data,
    #     #                                use_flag=True, path=suffix + '_abund.png')
    print ''
    print ''
    os.chdir('..')
