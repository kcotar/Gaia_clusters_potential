from astroquery.gaia import Gaia


def get_data_subset(ra_deg, dec_deg, rad_deg, dist, dist_span=None, rv_only=False):
    if dist_span is not None:
        max_parallax = 1e3/(max(dist-dist_span, 1.))
        min_parallax = 1e3/(dist+dist_span)
    else:
        min_parallax = -1.
        max_parallax = 100.
    # construct complete Gaia data query string
    gaia_query = "SELECT source_id,ra,dec,parallax,parallax_error,pmra,pmra_error,pmdec,pmdec_error,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,radial_velocity,radial_velocity_error,phot_variable_flag,a_g_val " +\
                 "FROM gaiadr2.gaia_source " +\
                 "WHERE parallax >= {:.4f} AND parallax <= {:.4f} ".format(min_parallax, max_parallax) +\
                 "AND CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',{:.7f},{:.7f},{:.7f}))=1 ".format(ra_deg, dec_deg, rad_deg)
    if rv_only:
        gaia_query += 'AND (radial_velocity IS NOT NULL) '
    # print ' QUERY:', gaia_quer
    try:
        gaia_job = Gaia.launch_job_async(gaia_query, dump_to_file=False)
        gaia_data = gaia_job.get_results()
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