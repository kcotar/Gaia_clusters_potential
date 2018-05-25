from os import chdir
from glob import glob
import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt

# DATASET 1
data_dir = '/home/klemen/data4_mount/isochrones/padova_Gaia_DR2/'
orig_colnames = ['Zini','Age','Mini','Mass','logL','logTe','logg','label','McoreTP','C_O','period0','period1','pmode','Mloss','tau1m','X','Y','Xc','Xn','Xo','Cexcess','Z','mbolmag','Gmag','G_BPmag','G_RPmag']
get_cols = ['Zini','Age','Mini','Mass','logL','logTe','logg','Z','mbolmag','Gmag','G_BPmag','G_RPmag']

# DATASET 2
# data_dir = '/home/klemen/data4_mount/isochrones/padova_UBVRIJHK/'
# orig_colnames = ['Zini','Age','Mini','Mass','logL','logTe','logg','label','McoreTP','C_O','period0','period1','pmode','Mloss','tau1m','X','Y','Xc','Xn','Xo','Cexcess','Z','mbolmag','Umag','Bmag','Vmag','Rmag','Imag','Jmag','Hmag','Kmag']
# get_cols = ['Zini','Age','Mini','Mass','logL','logTe','logg','Z','mbolmag','Umag','Bmag','Vmag','Rmag','Imag','Jmag','Hmag','Kmag']

chdir(data_dir)

Z_0 = 0.0152
dat_files = glob('output*.dat')

dat_data_all = list()
for dat in dat_files:
    print dat
    data = Table(np.genfromtxt(dat))
    data_colnames = data.colnames
    for i_col in range(len(data_colnames)):
        data[data_colnames[i_col]].name = orig_colnames[i_col]
    dat_data_all.append(data[get_cols])
    # idx = data['Age'] == 10000000
    # plt.scatter(data['logTe'][idx], data['logg'][idx], lw=0, s=2)
    # plt.show()
    # plt.close()

dat_data_all = vstack(dat_data_all)
dat_data_all['MH'] = np.log10(dat_data_all['Z']/Z_0)  # convert to dex metalicity
dat_data_all['MHini'] = np.log10(dat_data_all['Zini']/Z_0)  # convert to dex metalicity
dat_data_all['teff'] = 10**(dat_data_all['logTe'])  # convert to effective temperatures in K
dat_data_all.remove_columns(['logTe', 'Z', 'Zini'])

dat_data_all.write('isochrones_all.fits', overwrite=True)
