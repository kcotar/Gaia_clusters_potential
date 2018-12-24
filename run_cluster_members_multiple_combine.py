from astropy.table import Table, join, hstack, vstack
import os

data_dir = '/data4/cotar/Gaia_open_clusters_analysis_November-Asiago/'
multi_dir_prefix = data_dir+'Cluster_members_Gaia_DR2_'
n_dirs = 100

fits_file = 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits'

data = list([])

for i_dir in range(n_dirs):
    try:
        # print multi_dir_prefix+'{:02.0f}'.format(i_dir)+'/'+fits_file
        in_data = Table.read(multi_dir_prefix+'{:02.0f}'.format(i_dir)+'/'+fits_file)
        data.append(in_data)
        print len(in_data)
    except:
        pass

data = vstack(data)
data.write(data_dir+fits_file, overwrite=True)

fits_file = 'Cluster_members_Gaia_DR2_Kharchenko_2013_init_pos.fits'

data = list([])

for i_dir in range(n_dirs):
    try:
        in_data = Table.read(multi_dir_prefix+'{:02.0f}'.format(i_dir)+'/'+fits_file)
        data.append(in_data)
        print len(in_data)
    except:
        pass

data = vstack(data)
data.write(data_dir+fits_file, overwrite=True)

