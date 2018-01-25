from astropy.table import Table
from isochrones_class import *
from cluster_class import *

data_dir = '/home/klemen/data4_mount/'
# read Kharachenko clusters data
clusters = Table.read(data_dir+'clusters/Gaia_2017/tabled.csv')
# read Tgas data set
gaia_data = Table.read(data_dir+'TGAS_data_set/TgasSource_all.fits')
gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                         dec=gaia_data['dec'] * un.deg)
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova/isochrones_all.fits')

# iterate over preselected clusters
for obs_cluster in np.unique(clusters['Cluster']): #['Melotte_22']:
    print 'Working on:', obs_cluster

    clust_data = clusters[clusters['Cluster'] == obs_cluster]
    clust_center = coord.ICRS(ra=np.mean(clust_data['RAdeg']) * un.deg,
                              dec=np.mean(clust_data['DEdeg']) * un.deg)
    print clust_center
    # define possible cluster stars
    idx_possible_r1 = gaia_ra_dec.separation(clust_center) < 5. * un.deg
    gaia_cluster_sub = gaia_data[idx_possible_r1]
    print np.sum(idx_possible_r1)
    # kinematics selection

    idx_members = np.in1d(gaia_cluster_sub['source_id'], clust_data['Source'])
    print np.sum(idx_members)
    if np.sum(idx_members) == 0:
        print ' Zero possible members'
        continue
    # create and use cluster class
    cluster_class = CLUSTER(meh=0.0, age=10**7, isochrone=iso, id=obs_cluster)
    cluster_class.init_members(gaia_cluster_sub[idx_members])
    cluster_class.init_background(gaia_cluster_sub[~idx_members])

    cluster_class.plot_cluster_xyz(path=obs_cluster+'_stanje_zac.png')
