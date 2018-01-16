from os import chdir
from glob import glob
import numpy as np
from astropy.table import Table, vstack

data_dir = '/home/klemen/data4_mount/isochrones/padova/'
chdir(data_dir)

orig_colnames = ['Zini','Age','Mini','Mass','logL','logTe','logg','label','McoreTP','C_O','period0','period1','pmode','Mloss','tau1m','X','Y','Xc','Xn','Xo','Cexcess','Z','mbolmag','Gmag','G_BPmag','G_RPmag']
dat_files = glob('output*.dat')

dat_data_all = list()
for dat in dat_files:
    data = Table(np.genfromtxt(dat))
    data_colnames = data.colnames
    for i_col in range(len(data_colnames)):
        data[data_colnames[i_col]].name = orig_colnames[i_col]
    print data
