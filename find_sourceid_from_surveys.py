from astropy.table import Table, vstack, join
import numpy as np

data_dir = '/home/klemen/data4_mount/'

# add RV velocities from Galah and Rave
data_rave = Table.read(data_dir+'RAVE_DR5.fits')
data_galah = Table.read(data_dir+'sobject_iraf_53_reduced_20180222.fits')
data_cannon = Table.read(data_dir+'sobject_iraf_iDR2_180108_cannon.fits')
# xmatch of surveys with gaia
xmatch_rave = Table.read(data_dir+'rave_tgas_xmatch_DR5.csv')
xmatch_galah = Table.read(data_dir+'galah_tgas_xmatch_20180222.csv')

source_ids_find = [2326921471751885440]

for s_id in source_ids_find:
    print s_id
    idx_rave = xmatch_rave['source_id'] == s_id
    if np.sum(idx_rave) >= 1:
        for raveid in np.unique(xmatch_rave[idx_rave]['RAVEID']):
            print data_rave[data_rave['RAVEID'] == raveid]['RAVE_OBS_ID', 'RAdeg', 'DEdeg', 'HRV', 'eHRV']

    idx_galah = xmatch_galah['source_id'] == s_id
    if np.sum(idx_galah) >= 1:
        for sobid in np.unique(xmatch_galah[idx_galah]['sobject_id']):
            print data_galah[data_galah['sobject_id'] == sobid]['sobject_id', 'ra', 'dec', 'rv_guess', 'e_rv_guess']

    print ''
