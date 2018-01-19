from astropy.table import Table
from isochrones_class import *
from cluster_class import *

data_dir = '/home/klemen/data4_mount/'
# read Kharachenko clusters data
clusters = Table.read(data_dir+'clusters/Kharchenko_2013/catalog.csv')
# read Tgas data set
gaia_data = Table.read(data_dir+'TGAS_data_set/TgasSource_all.fits')
gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                         dec=gaia_data['dec'] * un.deg)
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova/isochrones_all.fits')

# iterate over preselected clusters
for obs_cluster in ['Melotte_22']:
    clust_data = clusters[clusters['Cluster'] == obs_cluster]
    clust_center = coord.ICRS(ra=clust_data['RAdeg'] * un.deg,
                              dec=clust_data['DEdeg'] * un.deg)
    # define possible cluster stars
    idx_possible_r1 = gaia_ra_dec.separation(clust_center) < clust_data['r2'] * un.deg
    gaia_cluster_sub = gaia_data[idx_possible_r1]
    # kinematics selection
    idx_kinem = np.logical_and(np.abs(gaia_cluster_sub['pmra'] - clust_data['pmRAc']) < 4.,
                               np.abs(gaia_cluster_sub['pmdec'] - clust_data['pmDEc']) < 4.)
    idx_dist = np.abs(1e3/gaia_cluster_sub['parallax'] - clust_data['d']) < 25
    idx_members = np.logical_and(idx_kinem, idx_dist)
    # create and use cluster class
    cluster_class = CLUSTER(meh=0.5, age=10 ** clust_data['logt'], isochrone=iso, id=obs_cluster)
    cluster_class.init_members(gaia_cluster_sub[idx_members])
    cluster_class.init_background(gaia_cluster_sub[~idx_members])

    cluster_class.plot_cluster_xyz(path='stanje_zac.png')
    # cluster_class.integrate_stars(years=50e5, step_years=5e5,
    #                               include_background=False, compute_g=False, store_hist=False)
    # cluster_class.plot_cluster_xyz()
    # add test particle to the cluster potential

    test_star = gaia_cluster_sub[~idx_members][7:8]
    test_star['pmra'] = clust_data['pmRAc']
    test_star['pmdec'] = clust_data['pmDEc']
    test_star['RV'] = 0
    cluster_class.init_test_particle(test_star)
    cluster_class.integrate_particle(100e6, step_years=5e5,
                                     integrate_stars_pos=True, integrate_stars_vel=True)
    cluster_class.plot_cluster_xyz_movement(path='stanje_p100M_G-v-kopici.png')


