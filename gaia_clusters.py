import os
from sklearn.externals import joblib
from astropy.table import Table
from isochrones_class import *
from cluster_class import *
from abundances_analysis import *
from galpy.potential import MWPotential2014, LogarithmicHaloPotential, IsochronePotential, MWPotential
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleStaeckel, actionAngleAdiabatic, estimateDeltaStaeckel, actionAngleSpherical

RV_USE = False
RV_ONLY = True
NO_INTERACTIONS = True
REVERSE_VEL = True
GALAXY_POTENTIAL = True

data_dir = '/home/klemen/data4_mount/'
# read Kharachenko clusters data
clusters = Table.read(data_dir+'clusters/Gaia_2017/tabled.csv')
# read Tgas data set
if RV_USE:
    gaia_data = Table.read(data_dir + 'TGAS_data_set/TgasSource_all_with_rv.fits')
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
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    os.chdir(out_dir)

    clust_data = clusters[clusters['Cluster'] == obs_cluster]
    clust_center = coord.ICRS(ra=np.mean(clust_data['RAdeg']) * un.deg,
                              dec=np.mean(clust_data['DEdeg']) * un.deg)
    print clust_center
    # define possible cluster stars
    idx_possible_r1 = gaia_ra_dec.separation(clust_center) < 10. * un.deg
    gaia_cluster_sub = gaia_data[idx_possible_r1]
    print np.sum(idx_possible_r1)
    # kinematics selection

    idx_members = np.in1d(gaia_cluster_sub['source_id'], clust_data['Source'])

    # redefine including the distance
    clust_center = coord.ICRS(ra=np.mean(gaia_cluster_sub['ra'][idx_members]) * un.deg,
                              dec=np.mean(gaia_cluster_sub['dec'][idx_members]) * un.deg,
                              distance=np.median(1e3/gaia_cluster_sub['parallax'][idx_members]) * un.pc)

    print np.sum(idx_members)
    if np.sum(idx_members) < 5:
        print ' Too few members'
        os.chdir('..')
        continue

    # print gaia_cluster_sub[idx_members]['rv', 'pmra', 'pmdec']

    # galpy potential implementation onlys
    for clust_star in gaia_cluster_sub[idx_members]:
        orbit = Orbit(vxvv=[clust_star['ra'] * un.deg,
                            clust_star['dec'] * un.deg,
                            1e3 / clust_star['parallax'] * un.pc,
                            clust_star['pmra'] * un.mas / un.yr,
                            clust_star['pmdec'] * un.mas / un.yr,
                            clust_star['rv'] * un.km / un.s], radec=True)
        orbit.turn_physical_on()

        ts = np.linspace(0, -220., 5000) * un.Myr
        orbit.integrate(ts, MWPotential2014)
        plt.plot(orbit.x(ts), orbit.y(ts), lw=0.5, c='red', alpha=0.3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(ls='--', alpha=0.5, color='black')
    plt.savefig('orbits_galpy.png', dpi=400)
    plt.close()

    # determine Galah and cannon members data
    cluster_members_galah = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], gaia_cluster_sub[idx_members]['source_id'])]['sobject_id']
    cannon_cluster_data = cannon_data[np.in1d(cannon_data['sobject_id'], cluster_members_galah)]

    # print cluster_members_galah['sobject_id']

    pkl_file_test = 'cluster_simulation.pkl'  # TEMP: for faster processing and testing
    if not os.path.isfile(pkl_file_test):
        # TODO: age and meh for those clusters
        cluster_class = CLUSTER(meh=0.0, age=10 ** 7, isochrone=iso, id=obs_cluster, reverse=REVERSE_VEL)
        cluster_class.init_members(gaia_cluster_sub[idx_members])
        cluster_class.init_background(gaia_cluster_sub[~idx_members])
        cluster_class.plot_cluster_xyz(path=obs_cluster + '_stanje_zac.png')

        gaia_cluster_sub_ra_dec = coord.ICRS(ra=gaia_cluster_sub['ra'] * un.deg,
                                             dec=gaia_cluster_sub['dec'] * un.deg,
                                             distance=1e3 / gaia_cluster_sub['parallax'] * un.pc)
        idx_test = gaia_cluster_sub_ra_dec.separation_3d(clust_center) < 80 * un.pc
        idx_test = np.logical_and(idx_test, ~idx_members)
        test_stars = gaia_cluster_sub[idx_test]
        print 'Number of test stars in cluster vicinity:', len(test_stars)

        cluster_class.init_test_particle(test_stars)
        cluster_class.galpy_run_all(members=True, particles=True, total_time=-220e6, step_years=1e4)
        # cluster_class.integrate_particle(220e6, step_years=1e4, include_galaxy_pot=GALAXY_POTENTIAL,
        #                                  integrate_stars_pos=True, integrate_stars_vel=True, disable_interactions=NO_INTERACTIONS)
        cluster_class.determine_orbits_that_cross_cluster()
        joblib.dump(cluster_class, pkl_file_test)
    else:
        cluster_class = joblib.load(pkl_file_test)
    cluster_class.plot_cluster_xyz_movement(path='orbits_integration_all.png')

    print 'Gaia source_ids:'
    possible_ejected = cluster_class.get_crossing_objects(min_time=2e6)  # use cluster crossing time for this?
    print ' possible:', len(possible_ejected)

    print 'Galah sobject_ids: '
    possible_ejected_galah = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], possible_ejected)][
        'sobject_id', 'source_id']
    print ' possible:', len(possible_ejected_galah)

    # video and plot outputs
    for sou_id in possible_ejected:

        galah_match_data = gaia_galah_xmatch[np.in1d(gaia_galah_xmatch['source_id'], sou_id)]
        if len(galah_match_data) == 1:
            sob_id = galah_match_data['sobject_id'].data[0]
        else:
            sob_id = 0

        suffix = str(sob_id) + '_' + str(sou_id)
        print 'Output results for:', suffix
        cluster_class.plot_cluster_xyz_movement(path=suffix + '_orbit.png', source_id=sou_id)
        # cluster_class.animate_particle_movement(path=suffix + '_video.mp4', source_id=sou_id, t_step=0.5e6)

        cannon_observed_data = cannon_data[np.in1d(cannon_data['sobject_id'], sob_id)]
        if len(cannon_observed_data) == 1:
            # it is also possible to create an abundance plot for Galah stars in Cannon dataset
            cluster_class.animate_particle_movement(path=suffix + '_video.mp4', source_id=sou_id, t_step=0.5e6)
            plot_abundances_histograms(cannon_cluster_data, cannon_observed_data,
                                       use_flag=True, path=suffix + '_abund.png')

    os.chdir('..')
