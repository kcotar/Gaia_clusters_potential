from get_isochrone import one_iso
from astropy.table import Table
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarWebQuery, BayestarQuery
from os import system, chdir
import astropy.units as un
import numpy as np
import matplotlib.pyplot as plt

magnitudes = {
	'gaia_g': ('tab_mag_odfnew/tab_mag_gaiaDR2maiz.dat', 'G', 0),
	'gaia_bft': ('tab_mag_odfnew/tab_mag_gaiaDR2maiz.dat', 'G_BPft', 1),
	'gaia_bbr': ('tab_mag_odfnew/tab_mag_gaiaDR2maiz.dat', 'G_BPbr', 2),
	'gaia_r': ('tab_mag_odfnew/tab_mag_gaiaDR2maiz.dat', 'G_RP', 3),
	# 'panst_g': ('tab_mag_odfnew/tab_mag_panstarrs1.dat', 'gP1', 4),
	# 'panst_r': ('tab_mag_odfnew/tab_mag_panstarrs1.dat', 'rP1', 5),
	# 'panst_i': ('tab_mag_odfnew/tab_mag_panstarrs1.dat', 'iP1', 6),
	# 'panst_z': ('tab_mag_odfnew/tab_mag_panstarrs1.dat', 'zP1', 7),
	# 'tmass_j': ('tab_mag_odfnew/tab_mag_2mass.dat', 'J', 8),
	# 'tmass_h': ('tab_mag_odfnew/tab_mag_2mass.dat', 'H', 9),
	# 'tmass_k': ('tab_mag_odfnew/tab_mag_2mass.dat', 'Ks', 10),
}

data_dir = '/shared/ebla/cotar/'
clusters_dir = data_dir + 'clusters/Cantat-Gaudin_2018/'

# load auxiliary data
clusters_data = Table.read(clusters_dir + 'table1.fits')
clusters_memb = Table.read(clusters_dir + 'members.fits')
clusters_galah = Table.read(data_dir + 'clusters/' + 'members_open_gaia_r2.fits')
galah_data = Table.read(data_dir + 'GALAH_iDR3_main_alpha_190529.fits')
# remove trailing whitespaces in original cluster names
clusters_data['cluster'] = [str(clusters_data['cluster'][i_l]).lstrip().rstrip() for i_l in range(len(clusters_data))]
clusters_memb['cluster'] = [str(clusters_memb['cluster'][i_l]).lstrip().rstrip() for i_l in range(len(clusters_memb))]
clusters_memb['d'] = 1e3/clusters_memb['parallax']

# initial conditions for clusters
init_clusters = ['NGC_2632']
# init_clusters = np.unique(clusters_galah['cluster'])
bayestar = BayestarWebQuery(version='bayestar2019')

subdir = 'Cluster_iso_fit'
system('mkdir ' + subdir)
chdir(subdir)

for cluster in init_clusters:
	system('mkdir ' + cluster)
	chdir(cluster)
	print cluster

	cluster_stars = clusters_memb[clusters_memb['cluster'] == cluster]
	print ' N stars:', len(cluster_stars)
	if len(cluster_stars) <= 3:
		chdir('..')
		continue
	cluster_info = clusters_data[clusters_data['cluster'] == cluster][0]
	print ' l: {:.3f}, b: {:.3f}, d:{:.1f}'.format(cluster_info['l'], cluster_info['b'], 1e3/cluster_info['par'])
	cl_center = SkyCoord(l=cluster_info['l']*un.deg, b=cluster_info['b']*un.deg,
						 distance=1e3/cluster_info['par']*un.pc, frame="galactic")

	# get initial cluster information
	dist_modulus = 5. * np.log10(1e3 / cluster_info['par'] / 10)
	reddening = bayestar(cl_center, mode='median')
	if ~np.isfinite(reddening): reddening = 0.

	# TODO: query from stilism

	av = 3.1 * reddening
	idx_Gd = np.in1d(galah_data['source_id'], cluster_stars['source_id'])
	idx_Gd = np.logical_and(idx_Gd, galah_data['flag_sp'] == 0)
	# try to determine M/H from Galah iDR# parameters
	fe_h = 0.
	m_h = 0.
	if np.sum(idx_Gd) > 0:
		fe_h = np.nanmedian(galah_data['fe_h'][idx_Gd])
		alpha_fe = np.nanmedian(galah_data['alpha_fe'][idx_Gd])
		m_h = fe_h + np.log10(10**alpha_fe * 0.694 + 0.306)
	print ' Init M/H: {:.3f}, Av: {:.3f}'.format(m_h, av)

	ages = np.arange(50., 1150., 50.)*1e6
	delta_m_hs = [-0.2, -0.1, 0.0, 0.1, 0.2]
	delta_avs = [-0.2, -0.1, 0.0, 0.1, 0.2]
	for d_av in delta_avs:
		if av + d_av < 0:
			continue
		for d_m_h in delta_m_hs:
			for age in ages:
				print '  age: {:3.0f}, delta M/H: {:.2f}, delta Av: {:.2f}'.format(age/1e6, d_m_h, d_av)
				isochrones, zams, teff, logg, mass = one_iso(age, m_h + d_m_h, av + d_av, magnitudes,
															 n_zams=60, version='3.1', z_log=True)

				plt.plot(isochrones[:,magnitudes['gaia_bbr'][2]][isochrones[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.2]-isochrones[:,magnitudes['gaia_r'][2]][isochrones[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.2],
						 isochrones[:,magnitudes['gaia_g'][2]][isochrones[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.2], 'k-', lw=0.6, alpha=0.4, label='fitted\nisochrone')
				plt.plot(isochrones[:,magnitudes['gaia_bft'][2]][isochrones[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.2]-isochrones[:,magnitudes['gaia_r'][2]][isochrones[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.2],
						 isochrones[:,magnitudes['gaia_g'][2]][isochrones[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.2], 'k-', lw=0.6, alpha=0.4)

				plt.plot(isochrones[:,magnitudes['gaia_bbr'][2]][isochrones[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.2]-isochrones[:,magnitudes['gaia_r'][2]][isochrones[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.2],
						 isochrones[:,magnitudes['gaia_g'][2]][isochrones[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.2]-0.752, 'k--', lw=0.6, alpha=0.4, label='binary\nsequence')
				plt.plot(isochrones[:,magnitudes['gaia_bft'][2]][isochrones[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.2]-isochrones[:,magnitudes['gaia_r'][2]][isochrones[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.2],
						 isochrones[:,magnitudes['gaia_g'][2]][isochrones[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.2]-0.752, 'k--', lw=0.6, alpha=0.4)

				plt.plot(zams[:,magnitudes['gaia_bbr'][2]][zams[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.4]-zams[:,magnitudes['gaia_r'][2]][zams[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.4],
						 zams[:,magnitudes['gaia_g'][2]][zams[:,magnitudes['gaia_g'][2]]<10.87-dist_modulus+0.4], 'r-', lw=0.6, alpha=0.3, label='ZAMS')
				plt.plot(zams[:,magnitudes['gaia_bft'][2]][zams[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.4]-zams[:,magnitudes['gaia_r'][2]][zams[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.4],
						 zams[:,magnitudes['gaia_g'][2]][zams[:,magnitudes['gaia_g'][2]]>10.87-dist_modulus-0.4], 'r-', lw=0.6, alpha=0.3)

				plt.scatter(cluster_stars['bp_rp'], cluster_stars['phot_g_mean_mag'] - 5.*np.log10(cluster_stars['d']/10), lw=0, s=2, c='C1')

				plt.gca().invert_yaxis()
				plt.xlabel(u'B$_P$ - R$_P$')
				plt.ylabel(u'M$_G$')
				plt.xlim(-0.5, 3.5)
				plt.ylim(13, -4)
				# plt.show()
				plt.grid(ls='--', c='black', alpha=0.2)
				plt.tight_layout()
				plt.savefig('age_{:04.0f}Myr_dmh_{:.2f}_dav_{:.2f}.png'.format(age/1e6, d_m_h, d_av), dpi=250)
				plt.close()

	chdir('..')

