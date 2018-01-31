import os
from sklearn.externals import joblib
from astropy.table import Table
from isochrones_class import *
from cluster_class import *
from galpy.potential import MWPotential2014, LogarithmicHaloPotential, IsochronePotential, MWPotential
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleStaeckel, actionAngleAdiabatic, estimateDeltaStaeckel, actionAngleSpherical


# galahic 8414429 8417585

data_dir = '/home/klemen/data4_mount/'
# read Kharachenko clusters data
clusters = Table.read(data_dir+'clusters/Kharchenko_2013/catalog.csv')
# read Tgas data set
gaia_data = Table.read(data_dir+'TGAS_data_set/TgasSource_all.fits')
galah_data = Table.read(data_dir+'sobject_iraf_52_reduced_20171111.fits')
cannon_data = Table.read(data_dir+'sobject_iraf_iDR2_180108_cannon.fits')
gaia_galah_xmatch = Table.read(data_dir+'galah_tgas_xmatch_20171111.csv')['sobject_id', 'source_id']
gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                         dec=gaia_data['dec'] * un.deg,
                         distance=1e3/gaia_data['parallax'] * un.pc)
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova/isochrones_all.fits')

selected_clusters = ['Alessi_13', 'ASCC_18', 'ASCC_20', 'Blanco_1', 'Collinder_65', 'Collinder_70', 'Melotte_20',
                     'Melotte_22', 'NGC_1039', 'NGC_2168', 'Platais_3', 'Platais_5', 'Stock_2']
# iterate over preselected clusters
for obs_cluster in ['Melotte_22']:  #selected_clusters:
    print 'Working on:', obs_cluster

    if not os.path.isdir(obs_cluster):
        os.mkdir(obs_cluster)
    os.chdir(obs_cluster)

    clust_data = clusters[clusters['Cluster'] == obs_cluster]
    print 'Basic info -->', 'r2:', clust_data['r2'].data[0], 'pmra:', clust_data['pmRAc'].data[0], 'pmdec:', clust_data['pmDEc'].data[0]

    clust_center = coord.ICRS(ra=clust_data['RAdeg'] * un.deg,
                              dec=clust_data['DEdeg'] * un.deg,
                              distance=clust_data['d'] * un.pc)
    # define possible cluster stars
    idx_possible_r1 = gaia_ra_dec.separation(clust_center) < clust_data['r2'] * 1.5 * un.deg
    gaia_cluster_sub = gaia_data[idx_possible_r1]
    print np.sum(idx_possible_r1)
    # kinematics selection
    idx_kinem = np.logical_and(np.abs(gaia_cluster_sub['pmra'] - clust_data['pmRAc']) < 3.,
                               np.abs(gaia_cluster_sub['pmdec'] - clust_data['pmDEc']) < 3.)
    idx_dist = np.abs(1e3/gaia_cluster_sub['parallax'] - clust_data['d']) < 20
    idx_members = np.logical_and(idx_kinem, idx_dist)
    print np.sum(idx_members)
    if np.sum(idx_members) == 0:
        print ' Zero possible members'
        continue


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
    #     ts = np.linspace(0, 250., 2000) * un.Myr
    #     orbit.integrate(ts, MWPotential2014)
    #     plt.plot(orbit.x(ts), orbit.y(ts), lw=0.5, c='red', alpha=0.3)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(ls='--', alpha=0.5, color='black')
    # plt.savefig('orbits_galpy.png', dpi=400)
    # plt.close()

    # my implementation of cluster class with only gravitational potential
    # create and use cluster class

    pkl_file_test = 'cluster_simulation.pkl'  # TEMP: for faster processing and testing
    if not os.path.isfile(pkl_file_test):
        cluster_class = CLUSTER(meh=0.0, age=10 ** clust_data['logt'], isochrone=iso, id=obs_cluster)
        cluster_class.init_members(gaia_cluster_sub[idx_members])
        cluster_class.init_background(gaia_cluster_sub[~idx_members])
        cluster_class.plot_cluster_xyz(path=obs_cluster+'_stanje_zac.png')

        gaia_cluster_sub_ra_dec = coord.ICRS(ra=gaia_cluster_sub['ra'] * un.deg,
                                             dec=gaia_cluster_sub['dec'] * un.deg,
                                             distance=1e3 / gaia_cluster_sub['parallax'] * un.pc)
        idx_test = gaia_cluster_sub_ra_dec.separation_3d(clust_center) < 60 * un.pc
        idx_test = np.logical_and(idx_test, ~idx_members)
        test_stars = gaia_cluster_sub[idx_test]
        test_stars['rv'] = 0
        print 'Number of test stars in cluster vicinity:', len(test_stars)

        cluster_class.init_test_particle(test_stars)
        cluster_class.integrate_particle(-220e6, step_years=-1e4, include_galaxy_pot=True,
                                         integrate_stars_pos=True, integrate_stars_vel=True)
        cluster_class.determine_orbits_that_cross_cluster()
        joblib.dump(cluster_class, pkl_file_test)
    else:
        cluster_class = joblib.load(pkl_file_test)
    cluster_class.plot_cluster_xyz_movement(path='orbits_integration_multi.png')

    # cluster_class.plot_cluster_xyz_movement(source_id=49809491645958528)
    # cluster_class.plot_cluster_xyz_movement(source_id=63774182671931264)
    # cluster_class.animate_particle_movement(source_id=49809491645958528)

    # cluster_class.plot_crossing_orbits(plot_prefix='crossing_orbit')

    possible_ejected = cluster_class.get_crossing_objects()
    print 'Galah stars: '
    possible_ejected_galah = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], possible_ejected)]['sobject_id', 'sobject_id']
    print possible_ejected_galah
    possible_ejected_galah = np.sort(possible_ejected_galah['sobject_id'])
    print ', '.join([str(s) for s in possible_ejected_galah])

    # video and plot and plot outputs
    for s_id in possible_ejected_galah['source_id']:
        cluster_class.plot_cluster_xyz_movement(path='orbit_'+str(s_id)+'.png', source_id=s_id)
        cluster_class.animate_particle_movement(path='video_'+str(s_id)+'.mp4', source_id=s_id)

    # print galah_data[np.in1d(galah_data['sobject_id'], possible_ejected_galah)]['rv_guess']
    # print cannon_data[np.in1d(cannon_data['sobject_id'], possible_ejected_galah)]['Feh_cannon', 'Al_abund_cannon','Ti_abund_cannon']

    os.chdir('..')



