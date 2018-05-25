import os
from astropy.table import Table, join, vstack, hstack
import numpy as np
from glob import glob

# all data

# data with rv velocities
save_dir = '/home/klemen/data4_mount/Gaia_DR2_RV'
retain_cols = ['source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
               'phot_g_mean_mag',  'phot_bp_mean_mag', 'phot_rp_mean_mag', 'radial_velocity', 'radial_velocity_error',
               'phot_variable_flag']

# 'bp_rp', 'bp_g', 'g_rp',
# 'rv_template_teff', 'rv_template_logg', 'rv_template_fe_h',
# 'phot''teff_val', 'radius_val'

os.chdir(save_dir)

gaia_data_all = list([])
for csv_file in glob('GaiaSource_*.csv'):
    print 'Reading file: '+csv_file
    gaia_data_all.append(Table.read(csv_file, format='ascii.csv')[retain_cols])

# join data together
gaia_data_all = vstack(gaia_data_all)

# rename some cols
gaia_data_all['radial_velocity'].name = 'rv'
gaia_data_all['radial_velocity_error'].name = 'rv_error'

# save everything
print 'Exporting new catalogue file'
gaia_data_all.write('GaiaSource_combined_RV.fits')
