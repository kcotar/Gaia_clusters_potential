from astroquery.gaia import Gaia


def get_data_subset(ra_deg, dec_deg, rad_deg, dist, dist_span=None, rv_only=False,
                    login=False, login_path='/shared/ebla/cotar/'):
    if dist_span is not None:
        max_parallax = 1e3/(max(dist-dist_span, 1.))
        min_parallax = 1e3/(dist+dist_span)
    else:
        min_parallax = -1.
        max_parallax = 100.
    # construct complete Gaia data query string
    gaia_query = "SELECT source_id,ra,dec,parallax,parallax_error,pmra,pmra_error,pmdec,pmdec_error,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,radial_velocity,radial_velocity_error " +\
                 "FROM gaiadr2.gaia_source " +\
                 "WHERE parallax >= {:.4f} AND parallax <= {:.4f} ".format(min_parallax, max_parallax) +\
                 "AND CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',{:.5f},{:.5f},{:.5f}))=1 ".format(ra_deg, dec_deg, rad_deg)
    if rv_only:
        gaia_query += 'AND (radial_velocity IS NOT NULL) '
    # print ' QUERY:', gaia_query
    try:
        if login:
            # login enables unlimited asynchronous download of data
            # NOTE: only up to 20 GB in total - needs manual deletition of data in the Gaia portal
            print ' Gaia login initiated'
            Gaia.login(credentials_file=login_path + 'gaia_archive_login.txt')
        # disable dump as results will be saved to a custom location later on in the analysis code
        gaia_job = Gaia.launch_job_async(gaia_query, dump_to_file=False)
        gaia_data = gaia_job.get_results()
        if login:
            Gaia.logout()
    except Exception as ee:
        print ee
        print ' Problem querying data.'
        return list([])
    for g_c in gaia_data.colnames:
        gaia_data[g_c].unit = ''
    gaia_data['radial_velocity'].name = 'rv'
    gaia_data['radial_velocity_error'].name = 'rv_error'
    # print gaia_data
    # print ' QUERY complete'
    print ' Retireved lines:', len(gaia_data)
    return gaia_data