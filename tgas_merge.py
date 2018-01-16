from astropy.table import Table, vstack

data_dir = '/home/klemen/data4_mount/TGAS_data_set/'

tgas_all = list()
for i_t in range(0, 16):
    print i_t
    fits = 'TgasSource_000-000-0{:02.0f}.fits'.format(i_t)
    data = Table.read(data_dir + fits)['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'phot_g_mean_mag', 'phot_variable_flag']
    tgas_all.append(data)

tgas_all = vstack(tgas_all)
tgas_all.write(data_dir+'TgasSource_all.fits', overwrite=True)
