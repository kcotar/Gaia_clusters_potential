from astropy.table import Table, vstack, join
import numpy as np

data_dir = '/home/klemen/data4_mount/'
data_tgas = data_dir+'TGAS_data_set/'

tgas_all = list()
for i_t in range(0, 16):
    print i_t
    fits = 'TgasSource_000-000-0{:02.0f}.fits'.format(i_t)
    data = Table.read(data_tgas + fits)['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'phot_g_mean_mag', 'phot_variable_flag']
    tgas_all.append(data)

tgas_all = vstack(tgas_all)

# remove any units description in data
for col in tgas_all.colnames:
    tgas_all[col].unit = None

# add RV velocities from Galah and Rave
data_rave = Table.read(data_dir+'RAVE_DR5.fits')
data_galah = Table.read(data_dir+'sobject_iraf_52_reduced_20171111.fits')
# xmatch lists
xmatch_rave = Table.read(data_dir+'rave_tgas_xmatch_DR5.csv')
xmatch_galah = Table.read(data_dir+'galah_tgas_xmatch_20171111.csv')

# combine datasets
data_rave['HRV'].name = 'rv'
data_rave['eHRV'].name = 'e_rv'
data_galah['rv_guess'].name = 'rv'
data_galah['e_rv_guess'].name = 'e_rv'
data_rave = join(data_rave, xmatch_rave, keys='RAVE_OBS_ID')
data_galah = join(data_galah, xmatch_galah, keys='sobject_id')

# merge repeated observations
rv_data = vstack((data_rave['source_id', 'rv', 'e_rv'], data_galah['source_id', 'rv', 'e_rv']))
source_uniq, source_uniq_n = np.unique(rv_data['source_id'], return_counts=True)
print 'Total rv repeats:', np.sum(source_uniq_n > 1)
for s_id in source_uniq[source_uniq_n > 1]:
    # median join data
    idx_rows = rv_data['source_id'] == s_id
    rv_mean = np.mean(rv_data[idx_rows]['rv'])
    e_rv_mean = np.mean(rv_data[idx_rows]['e_rv'])
    rv_data.remove_rows(np.where(idx_rows)[0])
    rv_data.add_row([s_id, rv_mean, e_rv_mean])
print rv_data

# combine with tgas set
print 'Final join'
tgas_all = join(tgas_all, rv_data, keys='source_id', join_type='outer')

# fill missing data
tgas_all = tgas_all.filled(0)  # temporary for test purposes - Gaia DR1
tgas_all.write(data_tgas+'TgasSource_all_with_rv.fits', overwrite=True)
