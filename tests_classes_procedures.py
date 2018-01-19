# # general astropy 2.0.2 tests
# import astropy.coordinates as coord
# import astropy.units as un
#
# ra_dec = coord.ICRS(ra=25. * un.deg,
#                     dec=27. * un.deg,
#                     distance=250. * un.pc,
#                     pm_ra_cosdec=4.8 * un.mas/un.yr,
#                     pm_dec=-15.16 * un.mas/un.yr,
#                     radial_velocity=250 * un.km/un.s)
# l_b = ra_dec.transform_to(coord.Galactic)
#
#
# raise SystemExit


# ISOCHRONES class
from isochrones_class import *
iso = ISOCHRONES('/home/klemen/data4_mount/isochrones/padova/isochrones_all.fits')
iso.select_isochrone(meh=0.5, age=10e7)
print iso.detemine_stellar_mass(gmag=5.48)

# CLUSTER CLASS
from cluster_class import *
d = Table.read('/home/klemen/data4_mount/TGAS_data_set/TgasSource_all.fits')
d_clust = d[:50]
d_clust['ra'] = 20.+np.random.randn(len(d_clust))/5.
d_clust['dec'] = 10.+np.random.randn(len(d_clust))/5.
d_clust['pmra'] = 0.
d_clust['pmdec'] = 0.
d_clust['parallax'] = 5.+np.random.randn(len(d_clust))/10.
clus = CLUSTER(meh=0.5, age=10e7, isochrone=iso)
clus.init_members(d_clust)
clus.estimate_masses()
print clus.cluster_center_pos
print clus.cluster_center_vel
# clus.plot_cluster_xyz()
# test particle define
d_particle = d_clust[0:1]
d_particle['ra'] += 2
d_particle['dec'] += 2
print d_particle
clus.init_test_particle(d_particle)
clus.integrate_particle(2e9, step_years=5e5)  # forward integration
# clus.integrate_particle(-2e9, step_years=-5e5)  # backward integration
clus.plot_cluster_xyz_movement()
