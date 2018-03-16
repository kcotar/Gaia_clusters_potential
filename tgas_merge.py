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
data_galah = Table.read(data_dir+'sobject_iraf_53_reduced_20180222.fits')
data_cannon = Table.read(data_dir+'sobject_iraf_iDR2_180108_cannon.fits')
# xmatch lists
xmatch_rave = Table.read(data_dir+'rave_tgas_xmatch_DR5.csv')
xmatch_galah = Table.read(data_dir+'galah_tgas_xmatch_20180222.csv')

# data filtering
data_cannon = data_cannon[data_cannon['flag_cannon'] == 0]

# remove bad an uncertain rv values
data_galah = data_galah[data_galah['e_rv_guess'] <= 1]
data_rave = data_rave[np.logical_and(data_rave['eHRV'] <= 5, data_rave['eHRV'] != 0)]

# rename names of columns to be compatible together between different datasets
data_rave['HRV'].name = 'rv'
data_rave['eHRV'].name = 'e_rv'
data_rave['Met_K'].name = 'feh'
data_rave['eMet_K'].name = 'e_feh'
data_galah['rv_guess'].name = 'rv'
data_galah['e_rv_guess'].name = 'e_rv'
data_cannon['Feh_cannon'].name = 'feh'
data_cannon['e_Feh_cannon'].name = 'e_feh'

final_cols = ['source_id', 'rv', 'e_rv', 'feh', 'e_feh']

# combine datasets
data_rave = join(data_rave, xmatch_rave, keys='RAVE_OBS_ID')
data_galah = join(data_galah, xmatch_galah, keys='sobject_id')
data_galah = join(data_galah, data_cannon, keys='sobject_id', join_type='left')

# merge repeated observations
rv_feh_data = vstack((data_rave[final_cols], data_galah[final_cols]))
source_uniq, source_uniq_n = np.unique(rv_feh_data['source_id'], return_counts=True)
print 'Total repeated objects:', np.sum(source_uniq_n > 1)
for s_id in source_uniq[source_uniq_n > 1]:
    # median join data
    idx_rows = rv_feh_data['source_id'] == s_id
    rv_mean = np.nanmean(rv_feh_data[idx_rows]['rv'])
    e_rv_mean = np.nanmean(rv_feh_data[idx_rows]['e_rv'])
    feh_mean = np.nanmean(rv_feh_data[idx_rows]['feh'])
    e_feh_mean = np.nanmean(rv_feh_data[idx_rows]['e_feh'])
    rv_feh_data.remove_rows(np.where(idx_rows)[0])
    rv_feh_data.add_row([s_id, rv_mean, e_rv_mean, feh_mean, e_feh_mean])
print rv_feh_data

# combine with tgas set
print 'Final join'
tgas_all = join(tgas_all, rv_feh_data, keys='source_id', join_type='left')

tgas_all.write(data_tgas+'TgasSource_all_with_rv_feh.fits', overwrite=True)

# fill missing data
# tgas_all.filled(-9999).write(data_tgas+'TgasSource_all_with_rv_feh_filled.fits', overwrite=True)
# tgas_all = tgas_all.filled(np.nan)  # temporary for test purposes - Gaia DR1
# another filling solution
# for col in final_cols[1:]:
#     if 'rv' in col:
#         idx_bad = np.logical_not(np.logical_or(np.isfinite(tgas_all[col]),
#                                                 tgas_all[col] == 1.))
#     else:
#         idx_bad = np.logical_not(np.isfinite(tgas_all[col]))
#     if np.sum(idx_bad) > 0:
#         tgas_all[col][idx_bad] = np.nan
# tgas_all.write(data_tgas+'TgasSource_all_with_rv_feh_filled2.fits', overwrite=True)
