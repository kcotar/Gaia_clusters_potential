import os
from astropy.table import Table
from isochrones_class import *
from cluster_class import *
from galpy.potential import MWPotential2014, LogarithmicHaloPotential
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleStaeckel, actionAngleAdiabatic, estimateDeltaStaeckel, actionAngleSpherical



data_dir = '/home/klemen/data4_mount/'
# read Kharachenko clusters data
clusters = Table.read(data_dir+'clusters/Kharchenko_2013/catalog.csv')
# read Tgas data set
gaia_data = Table.read(data_dir+'TGAS_data_set/TgasSource_all.fits')
gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                         dec=gaia_data['dec'] * un.deg)
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova/isochrones_all.fits')

selected_clusters = ['Alessi_13', 'ASCC_18', 'ASCC_20', 'Blanco_1', 'Collinder_65', 'Collinder_70', 'Melotte_20',
                     'Melotte_22', 'NGC_1039', 'NGC_2168', 'Platais_3', 'Platais_5', 'Stock_2']
# iterate over preselected clusters
for obs_cluster in selected_clusters:
    print 'Working on:', obs_cluster

    if not os.path.isdir(obs_cluster):
        os.mkdir(obs_cluster)
    os.chdir(obs_cluster)

    clust_data = clusters[clusters['Cluster'] == obs_cluster]
    clust_center = coord.ICRS(ra=clust_data['RAdeg'] * un.deg,
                              dec=clust_data['DEdeg'] * un.deg)
    # define possible cluster stars
    idx_possible_r1 = gaia_ra_dec.separation(clust_center) < clust_data['r2'] * 1.5 * un.deg
    gaia_cluster_sub = gaia_data[idx_possible_r1]
    print np.sum(idx_possible_r1)
    # kinematics selection
    idx_kinem = np.logical_and(np.abs(gaia_cluster_sub['pmra'] - clust_data['pmRAc']) < 3.,
                               np.abs(gaia_cluster_sub['pmdec'] - clust_data['pmDEc']) < 3.)
    idx_dist = np.abs(1e3/gaia_cluster_sub['parallax'] - clust_data['d']) < 25
    idx_members = np.logical_and(idx_kinem, idx_dist)
    print np.sum(idx_members)
    if np.sum(idx_members) == 0:
        print ' Zero possible members'
        continue


    # galpy potential implementation only
    for clust_star in gaia_cluster_sub[idx_members]:

        orbit = Orbit(vxvv=[clust_star['ra'] * un.deg,
                            clust_star['dec'] * un.deg,
                            1e3 / clust_star['parallax'] * un.pc,
                            clust_star['pmra'] * un.mas / un.yr,
                            clust_star['pmdec'] * un.mas / un.yr,
                            clust_star['rv'] * un.km / un.s], radec=True)
        # orbit.turn_physical_off()

        ts = np.linspace(0, -150., 100) * un.Myr
        orbit.integrate(ts, MWPotential2014)
        plt.plot(orbit.x(ts), orbit.y(ts))
    plt.savefig('galpy_orbits.png', dpi=350)
    plt.close()

    # my implementation of cluster class with only gravitational potential
    # create and use cluster class
    cluster_class = CLUSTER(meh=0.0, age=10 ** clust_data['logt'], isochrone=iso, id=obs_cluster)
    cluster_class.init_members(gaia_cluster_sub[idx_members])
    cluster_class.init_background(gaia_cluster_sub[~idx_members])

    cluster_class.plot_cluster_xyz(path=obs_cluster+'_stanje_zac.png')

    print ' PM:', clust_data['pmRAc'], clust_data['pmDEc']
    test_star = gaia_cluster_sub[~idx_members][7:8]
    test_star['pmra'] = clust_data['pmRAc']
    test_star['pmdec'] = clust_data['pmDEc']
    test_star['rv'] = 0
    cluster_class.init_test_particle(test_star)
    cluster_class.integrate_particle(-150e6, step_years=-10e5,
                                     integrate_stars_pos=True, integrate_stars_vel=True)
    cluster_class.plot_cluster_xyz_movement(path='stanje_p150M.png')

    os.chdir('..')



