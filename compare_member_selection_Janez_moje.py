import numpy as np
from astropy.table import Table, join

data_dir = '/data4/cotar/'
data_dir_clusters = data_dir+'Gaia_open_clusters_analysis_October-GALAH-clusters/'

galah_gaia = Table.read(data_dir + 'sobject_iraf_53_gaia.fits')
cluster_sel = Table.read(data_dir_clusters + 'Cluster_members_Gaia_DR2_Kharchenko_2013_init.fits')

cluster_sel = join(cluster_sel, galah_gaia['source_id', 'sobject_id'], keys=['source_id'], join_type='left')

txt_file = open(data_dir_clusters + 'members_open_gaia_r2.txt', 'r')
txt_lines_all = txt_file.readlines()
txt_file.close()

for txt_line in txt_lines_all:
    txt_line.rstrip('\n')
    cluster_name, cluster_sobjectid = txt_line.split('\t')
    cluster_sobjectid = np.int64(cluster_sobjectid.split(','))

    cluster_sel_sub = cluster_sel[cluster_sel['cluster'] == cluster_name]

    in_sel = np.sum(np.in1d(cluster_sel_sub['sobject_id'], cluster_sobjectid))

    print '{:3.0f}  {:3.0f}  - '.format(in_sel, len(cluster_sobjectid)), cluster_name
