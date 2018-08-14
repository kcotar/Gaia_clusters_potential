from astropy.table import Table, join, hstack, vstack
import os

multi_dir_prefix = 'Khar_cluster_initial_Gaia_DR2__'
n_dirs = 100

fits_file = 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits'

data =list([])

for i_dir in range(n_dirs):
    try:
        data.append(Table.read(multi_dir_prefix+str(i_dir)+'/'+fits_file))
    except:
        pass

data = vstack(data)
data.write(fits_file, overwrite=True)

fits_file = 'Cluster_members_Gaia_DR2_Kharchenko_2013_init_pos.fits'

data =list([])

for i_dir in range(n_dirs):
    try:
        data.append(Table.read(multi_dir_prefix+str(i_dir)+'/'+fits_file))
    except:
        pass

data = vstack(data)
data.write(fits_file, overwrite=True)

