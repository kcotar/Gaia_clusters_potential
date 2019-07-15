from astropy.table import Table, join, hstack, vstack
import os

data_dir = '/shared/data-camelot/cotar/GaiaDR2_open_clusters_1907_GALAH/'
multi_dir_prefix = data_dir+'Cluster_members_Gaia_DR2_'
n_dirs = 100

fits_file_in = 'cluster_params.fits'
fits_file_out = 'Cluster_parameters_GaiaDR2_combined.fits'

data = list([])

for i_dir in range(n_dirs):
    try:
        # print multi_dir_prefix+'{:02.0f}'.format(i_dir)+'/'+fits_file
        in_data = Table.read(multi_dir_prefix+'{:02.0f}'.format(i_dir)+'/'+fits_file_in)
        data.append(in_data)
        print len(in_data)
    except:
        pass

data = vstack(data)
data.write(data_dir+fits_file_out, overwrite=True)

fits_file_in = 'table1_modified_parameters.fits'
fits_file_out = 'Cluster_members_analysis_GaiaDR2_combined.fits'

data = list([])

for i_dir in range(n_dirs):
    try:
        in_data = Table.read(multi_dir_prefix+'{:02.0f}'.format(i_dir)+'/'+fits_file_in)
        data.append(in_data)
        print len(in_data)
    except:
        pass

data = vstack(data)
data.write(data_dir+fits_file_out, overwrite=True)

