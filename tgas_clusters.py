import os
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

# step 1 of the analysis
MEMBER_DETECTION = True

# step 2 of the analysis
RV_USE = True
RV_ONLY = False
NO_INTERACTIONS = False
REVERSE_VEL = True
GALAXY_POTENTIAL = True

data_dir = '/home/klemen/data4_mount/'
# read Kharachenko clusters data
clusters = Table.read(data_dir+'clusters/Kharchenko_2013/catalog.csv')
# read Tgas data set
if RV_USE:
    gaia_data = Table.read(data_dir + 'TGAS_data_set/TgasSource_all_with_rv_feh.fits')
    if RV_ONLY:
        gaia_data = gaia_data[np.logical_and(gaia_data['rv'] != 0, gaia_data['e_rv'] != 0)]
else:
    gaia_data = Table.read(data_dir+'TGAS_data_set/TgasSource_all.fits')
galah_data = Table.read(data_dir+'sobject_iraf_52_reduced_20171111.fits')
cannon_data = Table.read(data_dir+'sobject_iraf_iDR2_180108_cannon.fits')
gaia_galah_xmatch = Table.read(data_dir+'galah_tgas_xmatch_20171111.csv')['sobject_id', 'source_id']
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova/isochrones_all.fits')

gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                         dec=gaia_data['dec'] * un.deg,
                         distance=1e3/gaia_data['parallax'] * un.pc)

selected_clusters = ['Alessi_13', 'ASCC_18', 'ASCC_20', 'Blanco_1', 'Collinder_65', 'Collinder_70', 'Melotte_20',
                     'Melotte_22', 'NGC_1039', 'NGC_2168', 'Platais_3', 'Platais_5', 'Stock_2']

cluster_fits_out = 'Cluster_members_Gaia_DR1_Kharchenko_2013_init.fits'
os.chdir('Khar_cluster_initial_all')


# ------------------------------------------
# ----------------  STEP 1  ----------------
# ------------------------------------------


if MEMBER_DETECTION:
    out_dir_suffix = '_member_sel'
    # iterate over preselected clusters
    cluster_obj_found_out = Table(names=('source_id', 'cluster'), dtype=('int64', 'S30'))
    for obs_cluster in np.unique(clusters['Cluster']):
    # for obs_cluster in selected_clusters:
        print 'Working on:', obs_cluster

        out_dir = obs_cluster + out_dir_suffix

        clust_data = clusters[clusters['Cluster'] == obs_cluster]
        print ' Basic info -->', 'r1:', clust_data['r1'].data[0], 'r2:', clust_data['r2'].data[0], 'pmra:', clust_data['pmRAc'].data[0], 'pmdec:', clust_data['pmDEc'].data[0]

        clust_center = coord.ICRS(ra=clust_data['RAdeg'] * un.deg,
                                  dec=clust_data['DEdeg'] * un.deg,
                                  distance=clust_data['d'] * un.pc)

        idx_possible_r2 = gaia_ra_dec.separation(clust_center) < clust_data['r2'] * 1.5 * un.deg
        gaia_cluster_sub_r2 = gaia_data[idx_possible_r2]
        idx_distance = np.abs(1e3/gaia_cluster_sub_r2['parallax'] - clust_data['d']) < 250  # for now because of uncertain distances
        gaia_cluster_sub_r2 = gaia_cluster_sub_r2[idx_distance]

        if len(idx_possible_r2) < 50:
            print ' Not enough objects in selection ('+str(len(idx_possible_r2))+')'
            continue

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        os.chdir(out_dir)

        find_members_class = CLUSTER_MEMBERS(gaia_cluster_sub_r2, clust_data)
        find_members_class.plot_on_sky(path='cluster_pos.png', mark_objects=False)
        print ' Multi radius Gaussian mixture'
        for c_rad in np.linspace(0, np.float64(clust_data['r2']), 16)[1:]:
            find_members_class.perform_selection(c_rad, bayesian_mixture=False)
        find_members_class.plot_members(25, path='cluster_pm_multi_sel.png')
        find_members_class.plot_members(25, path='cluster_pm_multi_n.png', show_n_sel=True)
        find_members_class.plot_selected_hist(path='selection_hist.png')
        clust_ok = find_members_class.refine_distances_selection(out_plot=True, path='cluster_parsec.png')
        if clust_ok:
            # elipse fitting to the data and search for additional members inside it, discard distant stars in hull
            find_members_class.include_iniside_hull(distance_limits=True)
            find_members_class.plot_members(path='cluster_pm_multi_sel_final.png', show_final=True)
            members_source_id = find_members_class.get_cluster_members(recompute=False, idx_only=False)
            find_members_class.plot_on_sky(path='cluster_pos_final.png', mark_objects=True)
            # add members to the resulting file
            for m_s_id in members_source_id:
                cluster_obj_found_out.add_row([m_s_id, obs_cluster])

        os.chdir('..')
        print ''  # nicer looking output with blank lines

    # save cluster results
    cluster_obj_found_out.write(cluster_fits_out, format='fits', overwrite=True)


# ------------------------------------------
# ----------------  STEP 2  ----------------
# ------------------------------------------

out_dir_suffix = ''
if RV_USE:
    out_dir_suffix += '_with_rv'
    if RV_ONLY:
        out_dir_suffix += '_only'
if NO_INTERACTIONS:
    out_dir_suffix += '_nointer'
if REVERSE_VEL:
    out_dir_suffix += '_reversevel'
if not GALAXY_POTENTIAL:
    out_dir_suffix += '_nogalpot'

    # print np.sum(idx_possible_r1)
    # # kinematics selection
    # idx_kinem = np.logical_and(np.abs(gaia_cluster_sub['pmra'] - clust_data['pmRAc']) < 3.,
    #                            np.abs(gaia_cluster_sub['pmdec'] - clust_data['pmDEc']) < 3.)
    # idx_dist = np.abs(1e3/gaia_cluster_sub['parallax'] - clust_data['d']) < 20
    # idx_members = np.logical_and(idx_kinem, idx_dist)
    #
    # if RV_USE:
    #     # additional rv refinement and selection of initial members
    #     print np.sum(idx_members)
    #     rv_mean = np.median(gaia_cluster_sub['rv'][idx_members])
    #     rv_std = np.std(gaia_cluster_sub['rv'][idx_members])
    #     idx_rv = np.abs(gaia_cluster_sub['rv'] - rv_mean) < 5  # rv_std*0.1
    #     print 'RV: ', rv_mean, rv_std
    #     idx_members = np.logical_and(idx_members, idx_rv)
    #
    # print np.sum(idx_members)
    # if np.sum(idx_members) < 5:
    #     print ' Low possible members'
    #     os.chdir('..')
    #     continue
    #
    # # determine Galah and cannon members data
    # cluster_members_galah = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], gaia_cluster_sub[idx_members]['source_id'])]['sobject_id']
    # cannon_cluster_data = cannon_data[np.in1d(cannon_data['sobject_id'], cluster_members_galah)]
    #
    # # galpy potential implementation onlys
    # for clust_star in gaia_cluster_sub[idx_members]:
    #
    #     orbit = Orbit(vxvv=[clust_star['ra'] * un.deg,
    #                         clust_star['dec'] * un.deg,
    #                         1e3 / clust_star['parallax'] * un.pc,
    #                         clust_star['pmra'] * un.mas / un.yr,
    #                         clust_star['pmdec'] * un.mas / un.yr,
    #                         clust_star['rv'] * un.km / un.s], radec=True)
    #     orbit.turn_physical_on()
    #
    #     ts = np.linspace(0, -220., 5000) * un.Myr
    #     orbit.integrate(ts, MWPotential2014)
    #     plt.plot(orbit.x(ts), orbit.y(ts), lw=0.5, c='red', alpha=0.3)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(ls='--', alpha=0.5, color='black')
    # plt.savefig('orbits_galpy.png', dpi=400)
    # plt.close()
    #
    # # my implementation of cluster class with only gravitational potential
    # # create and use cluster class
    #
    # pkl_file_test = 'cluster_simulation.pkl'  # TEMP: for faster processing and testing
    # if not os.path.isfile(pkl_file_test):
    #     cluster_class = CLUSTER(meh=0.0, age=10 ** clust_data['logt'], isochrone=iso, id=obs_cluster, reverse=REVERSE_VEL)
    #     cluster_class.init_members(gaia_cluster_sub[idx_members])
    #     cluster_class.init_background(gaia_cluster_sub[~idx_members])
    #     cluster_class.plot_cluster_xyz(path=obs_cluster+'_stanje_zac.png')
    #
    #     gaia_cluster_sub_ra_dec = coord.ICRS(ra=gaia_cluster_sub['ra'] * un.deg,
    #                                          dec=gaia_cluster_sub['dec'] * un.deg,
    #                                          distance=1e3 / gaia_cluster_sub['parallax'] * un.pc)
    #     idx_test = gaia_cluster_sub_ra_dec.separation_3d(clust_center) < 60 * un.pc
    #     idx_test = np.logical_and(idx_test, ~idx_members)
    #     test_stars = gaia_cluster_sub[idx_test]
    #     print 'Number of test stars in cluster vicinity:', len(test_stars)
    #
    #     cluster_class.init_test_particle(test_stars)
    #     cluster_class.integrate_particle(200e6, step_years=1e4, include_galaxy_pot=GALAXY_POTENTIAL,
    #                                      integrate_stars_pos=True, integrate_stars_vel=True, disable_interactions=NO_INTERACTIONS)
    #     cluster_class.determine_orbits_that_cross_cluster()
    #     joblib.dump(cluster_class, pkl_file_test)
    # else:
    #     cluster_class = joblib.load(pkl_file_test)
    # cluster_class.plot_cluster_xyz_movement(path='orbits_integration_all.png')
    #
    # print 'Gaia source_ids:'
    # possible_ejected = cluster_class.get_crossing_objects(min_time=2e6)  # use cluster crossing time for this?
    # print ' possible:', len(possible_ejected)
    #
    # print 'Galah sobject_ids: '
    # possible_ejected_galah = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], possible_ejected)]['sobject_id', 'source_id']
    # print ' possible:', len(possible_ejected_galah)
    #
    # # video and plot outputs
    # for sou_id in possible_ejected:
    #
    #     galah_match_data = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], sou_id)]
    #     if len(galah_match_data) == 1:
    #         sob_id = galah_match_data['sobject_id'].data[0]
    #     else:
    #         sob_id = 0
    #
    #     suffix = str(sob_id)+'_'+str(sou_id)
    #     print 'Output results for:', suffix
    #     cluster_class.plot_cluster_xyz_movement(source_id=sou_id, path=suffix+'_orbit.png')
    #     cluster_class.animate_particle_movement(path=suffix+'_video.mp4', source_id=sou_id, t_step=0.2e6)
    #
    #     cannon_observed_data = cannon_data[np.in1d(cannon_data['sobject_id'], sob_id)]
    #     if len(cannon_observed_data) == 1:
    #         # cluster_class.plot_cluster_xyz_movement(source_id=sou_id)#, path=suffix + '_orbit.png')
    #         # cluster_class.animate_particle_movement(path=suffix + '_video.mp4', source_id=sou_id, t_step=0.2e6)
    #         # it is also possible to create an abundance plot for Galah stars in Cannon dataset
    #         plot_abundances_histograms(cannon_cluster_data, cannon_observed_data,
    #                                    use_flag=True, path=suffix+'_abund.png')
    #
    # os.chdir('..')



