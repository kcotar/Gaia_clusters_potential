from astropy.table import Table, join, hstack, vstack
import os

data_dir = '/home/klemen/data4_mount/Gaia_open_clusters_analysis_rerun/'
multi_dir_prefix = data_dir+'Khar_cluster_initial_Gaia_DR2__'
n_dirs = 100

fits_file = 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits'

data =list([])

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

