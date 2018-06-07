import os, imp
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
from gaia_data_queries import *
imp.load_source('hr_class', '../Binaries_clusters/HR_diagram_class.py')
from hr_class import *

# step 1 of the analysis
MEMBER_DETECTION = True  # Step 1
ORBITS_ANALYSIS = False  # Step 2
USE_UPDATED_KHAR = False


# step 2 of the analysis
RV_USE = True
RV_ONLY = False
NO_INTERACTIONS = False
REVERSE_VEL = True
GALAXY_POTENTIAL = True
QUERY_DATA = True

data_dir = '/home/klemen/data4_mount/'
khar_dir = data_dir + 'clusters/Kharchenko_2013/'

csv_out_cols_init = ['source_id', 'ra', 'dec', 'rv', 'pmra', 'pmdec', 'phot_g_mean_mag']
csv_out_cols = ['source_id', 'ra', 'dec', 'rv', 'pmra', 'pmdec', 'phot_g_mean_mag', 'time_in_cluster', 'ang_dist', '3d_dist']

# read Kharachenko clusters data
if USE_UPDATED_KHAR:
    clusters = Table.read(khar_dir + 'catalog_tgas_update.csv')
else:
    clusters = Table.read(khar_dir + 'catalog.csv')
# read Gaia data set
if not QUERY_DATA:
    print 'Reading Gaia data'
    if RV_USE:
        gaia_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV.fits')
        if RV_ONLY:
            gaia_data = gaia_data[np.logical_and(gaia_data['rv'] != 0, gaia_data['rv_error'] != 0)]
    else:
        gaia_data = Table.read(data_dir+'Gaia_DR2/GaiaSource_combined.fits')
    gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                             dec=gaia_data['dec'] * un.deg,
                             distance=1e3 / gaia_data['parallax'] * un.pc)

print 'Reading additional data'
galah_data = Table.read(data_dir+'sobject_iraf_52_reduced_20171111.fits')
cannon_data = Table.read(data_dir+'sobject_iraf_iDR2_180108_cannon.fits')
gaia_galah_xmatch = Table.read(data_dir+'galah_tgas_xmatch_20171111.csv')['sobject_id', 'source_id']
# load isochrones into class
iso = ISOCHRONES(data_dir+'isochrones/padova_Gaia/isochrones_all.fits', photo_system='Gaia')

selected_clusters_Asiago = ['NGC_2264', 'NGC_2281', 'NGC_2301', 'NGC_2548', 'NGC_2632', 'NGC_2682',
                            'Melotte_111', 'Mamajek_2', 'IC_4665', 'Collinder_350', 'Collinder_359',
                            'NGC_6633', 'IC_4756', 'Stephenson_1',
                            'NGC_6738', 'NGC_6793', 'Stock_1',
                            'Turner_9', 'Roslund_1',
                            'NGC_6828', 'Roslund_5', 'NGC_6882',
                            'Roslund_6',
                            'NGC_6940', 'FSR_0251', 'Alessi_12', 'Roslund_7', 'FSR_0261',
                            'NGC_6991A', 'NGC_6997', 'Basel_15', 'NGC_7058',
                            'NGC_7063', 'NGC_7092', 'IC_1396', 'IC_5146', 'NGC_7160',
                            'NGC_7243', 'Teutsch_39']

cluster_fits_out = 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits'

output_dir = 'Khar_cluster_initial_Gaia_DR2_allinr2'
os.system('mkdir '+output_dir)
os.chdir(output_dir)


# ------------------------------------------
# ----------------  STEP 1  ----------------
# ------------------------------------------


if MEMBER_DETECTION:
    out_dir_suffix = '_member_sel'
    cluster_obj_found_out = Table(names=('source_id', 'cluster'), dtype=('int64', 'S30'))

    # iterate over (pre)selected clusters
    # for obs_cluster in np.unique(clusters['Cluster']):
    for obs_cluster in selected_clusters_Asiago:
    # for obs_cluster in ['NGC_2264']:
        print 'Working on:', obs_cluster

        out_dir = obs_cluster + out_dir_suffix

        idx_cluster_pos = np.where(clusters['Cluster'] == obs_cluster)[0]
        clust_data = clusters[idx_cluster_pos]
        print ' Basic info -->', 'r1:', clust_data['r1'].data[0], 'r2:', clust_data['r2'].data[0], 'pmra:', clust_data['pmRAc'].data[0], 'pmdec:', clust_data['pmDEc'].data[0]

        clust_center = coord.ICRS(ra=clust_data['RAdeg'] * un.deg,
                                  dec=clust_data['DEdeg'] * un.deg,
                                  distance=clust_data['d'] * un.pc)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        os.chdir(out_dir)

        if QUERY_DATA:
            uotput_file = 'gaia_query_data.csv'
            if os.path.isfile(uotput_file):
                gaia_data = Table.read(uotput_file)
            else:
                print ' Sending QUERY to download Gaia data'
                gaia_data = get_data_subset(clust_data['RAdeg'].data[0], clust_data['DEdeg'].data[0],
                                            clust_data['r2'].data[0] * 1.25,
                                            clust_data['d'].data[0], dist_span=None)
                if len(gaia_data) == 0:
                    os.chdir('..')
                    continue
                gaia_data.write(uotput_file)
            gaia_ra_dec = coord.ICRS(ra=gaia_data['ra'] * un.deg,
                                     dec=gaia_data['dec'] * un.deg,
                                     distance=1e3 / gaia_data['parallax'] * un.pc)

        idx_possible_r2 = gaia_ra_dec.separation(clust_center) < clust_data['r2'] * 1.5 * un.deg
        gaia_cluster_sub_r2 = gaia_data[idx_possible_r2]
        idx_distance = np.abs(1e3/gaia_cluster_sub_r2['parallax'] - clust_data['d']) < 300  # for now because of uncertain distances
        gaia_cluster_sub_r2 = gaia_cluster_sub_r2[idx_distance]

        n_in_selection = len(gaia_cluster_sub_r2)
        if n_in_selection < 50:
            print ' Not enough objects in selection ('+str(n_in_selection)+')'
            os.chdir('..')
            continue

        find_members_class = CLUSTER_MEMBERS(gaia_cluster_sub_r2, clust_data)
        find_members_class.plot_on_sky(path='cluster_pos.png', mark_objects=False)

        # first determine new cluster center in pm space if needed
        # find_members_class.determine_cluster_center()

        print ' Multi radius Gaussian mixture'
        pm_median_all = [np.nanmedian(gaia_data['pmra']), np.nanmedian(gaia_data['pmdec'])]
        for c_rad in np.linspace(np.float64(clust_data['r1']), np.float64(clust_data['r2']), 3):
            find_members_class.perform_selection_density(c_rad, pm_median_all, suffix='_{:.3f}'.format(c_rad))
            # find_members_class.perform_selection(c_rad, bayesian_mixture=False, covarinace='full', max_com=4)
        # !!!!!!! TEMP !!!!!!!
        os.chdir('..')
        continue

        find_members_class.plot_members(25, path='cluster_pm_multi_sel.png')
        find_members_class.plot_members(25, path='cluster_pm_multi_n.png', show_n_sel=True)
        find_members_class.plot_selected_hist(path='selection_hist.png')
        clust_ok = find_members_class.refine_distances_selection(out_plot=True, path='cluster_parsec.png')
        if clust_ok:
            # elipse fitting to the data and search for additional members inside it, discard distant stars in hull
            find_members_class.include_iniside_hull(distance_limits=True, manual_hull=False)
            find_members_class.plot_members(path='cluster_pm_multi_sel_final.png', show_final=True)
            # possible RV refinements
            clust_ok = find_members_class.refine_distances_selection_RV(out_plot=True, path='cluster_rv.png')
            # get and store member results
            members_source_id = find_members_class.get_cluster_members(recompute=False, idx_only=False)
            find_members_class.plot_on_sky(path='cluster_pos_final.png', mark_objects=True)
            # add members to the resulting file
            if np.isscalar(members_source_id):
                os.chdir('..')
                continue

            for m_s_id in members_source_id:
                cluster_obj_found_out.add_row([m_s_id, obs_cluster])
            # final members selection
            gaia_cluster_members_final = gaia_cluster_sub_r2[find_members_class.get_cluster_members(recompute=False, idx_only=True)]

            # create HR diagram
            output_list_objects(gaia_cluster_members_final, clust_center, csv_out_cols_init, 'memebers_data.csv')
            cluster_hr = HR_DIAGRAM(gaia_cluster_members_final, clust_data[clust_data['Cluster'] == obs_cluster], isochrone=iso, photo_system='Gaia')
            cluster_hr.plot_HR_diagram(include_isocrone=True, include_rv=True, path='hr_diagram.png')

            # update actual Khar data if not using updated file already
            if not USE_UPDATED_KHAR:
                print ' Parameters write out'
                clusters['RAdeg'][idx_cluster_pos] = np.nanmedian(gaia_cluster_members_final['ra'])
                clusters['DEdeg'][idx_cluster_pos] = np.nanmedian(gaia_cluster_members_final['dec'])
                clusters['d'][idx_cluster_pos] = np.nanmedian(1e3 / gaia_cluster_members_final['parallax'])
                clusters['pmRAc'][idx_cluster_pos] = np.nanmedian(gaia_cluster_members_final['pmra'])
                clusters['pmDEc'][idx_cluster_pos] = np.nanmedian(gaia_cluster_members_final['pmdec'])
                clusters.write(khar_dir + 'catalog_tgas_update.csv', overwrite=True, format='ascii.csv')

        os.chdir('..')
        print ''  # nicer looking output with blank lines

    # save cluster results
    cluster_obj_found_out.write(cluster_fits_out, format='fits', overwrite=True)
    # save only ra/dec information for determined objects
    gaia_data[np.in1d(gaia_data['source_id'], cluster_obj_found_out['source_id'])]['source_id', 'ra', 'dec'].write(cluster_fits_out[:-5]+'_pos.fits', format='fits', overwrite=True)


clusters = Table.read(cluster_fits_out)

# ------------------------------------------
# ----------------  STEP 2  ----------------
# ------------------------------------------
if ORBITS_ANALYSIS:
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

    # iterate over (pre)selected clusters
    # for obs_cluster in np.unique(clusters['Cluster']):
    for obs_cluster in selected_clusters_Asiago:
        print 'Working on:', obs_cluster

        out_dir = obs_cluster + out_dir_suffix

        idx_members = np.where(clusters['cluster'] == obs_cluster)[0]
        clust_data = clusters[idx_members]

        if not RV_USE:
            gaia_data['rv'] = 0

        print np.sum(idx_members)
        if np.sum(idx_members) < 5:
            print ' Low possible members'
            os.chdir('..')
            continue

        # determine Galah and cannon members data

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

        # my implementation of cluster class with only gravitational potential
        # create and use cluster class

        pkl_file_test = 'cluster_simulation.pkl'  # TEMP: for faster processing and testing
        if not os.path.isfile(pkl_file_test):
            cluster_class = CLUSTER(meh=0.0, age=10 ** clust_data['logt'], isochrone=iso, id=obs_cluster, reverse=REVERSE_VEL)
            cluster_class.init_members(gaia_cluster_sub[idx_members])
            cluster_class.init_background(gaia_cluster_sub[~idx_members])
            cluster_class.plot_cluster_xyz(path=obs_cluster+'_stanje_zac.png')

            gaia_cluster_sub_ra_dec = coord.ICRS(ra=gaia_cluster_sub['ra'] * un.deg,
                                                 dec=gaia_cluster_sub['dec'] * un.deg,
                                                 distance=1e3 / gaia_cluster_sub['parallax'] * un.pc)
            idx_test = gaia_cluster_sub_ra_dec.separation_3d(clust_center) < 60 * un.pc
            idx_test = np.logical_and(idx_test, ~idx_members)
            test_stars = gaia_cluster_sub[idx_test]
            print 'Number of test stars in cluster vicinity:', len(test_stars)

            cluster_class.init_test_particle(test_stars)
            cluster_class.integrate_particle(200e6, step_years=1e4, include_galaxy_pot=GALAXY_POTENTIAL,
                                             integrate_stars_pos=True, integrate_stars_vel=True, disable_interactions=NO_INTERACTIONS)
            cluster_class.determine_orbits_that_cross_cluster()
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



