import subprocess
import os
import numpy as np

n_cpu = 10
selected_clusters = ["NGC_189","NGC_225","ASCC_4","ASCC_5","NGC_381","NGC_752","NGC_743","NGC_886","NGC_1039","NGC_1027","ASCC_10","NGC_1333","NGC_1342","ASCC_11","IC_348","NGC_1444","NGC_1496","NGC_1502","NGC_1528","NGC_1545","NGC_1579","NGC_1582","ASCC_12","NGC_1664","NGC_1708","ASCC_13","IC_4665","NGC_6633","IC_4756","NGC_6709","NGC_6728","NGC_6738","ASCC_100","ASCC_101","NGC_6793","NGC_6800","ASCC_103","NGC_6811","ASCC_104","ASCC_105","ASCC_106","ASCC_107","NGC_6828","ASCC_108","ASCC_109","NGC_6856","ASCC_110","NGC_6882","ASCC_112","NGC_6910","NGC_6913","NGC_6940","NGC_6991A","NGC_6997","NGC_7031","NGC_7036","ASCC_113","NGC_7058","NGC_7063","NGC_7082","NGC_7092","NGC_7084","IC_1396","ASCC_114","NGC_7129","IC_5146","NGC_7160","ASCC_115","NGC_7209","ASCC_118","NGC_7243","ASCC_119","ASCC_122","ASCC_123","ASCC_124","NGC_7429","ASCC_125","NGC_7438","ASCC_126","ASCC_127","ASCC_128","NGC_7772","NGC_7762"]


# manual selection of the best cluster candidates
# n_cpu = 5
# selected_clusters = ['NGC_129','NGC_189','NGC_225','NGC_366','NGC_188','NGC_381','NGC_663','NGC_6694','NGC_6704','NGC_6709','NGC_6735','NGC_6866','NGC_7142','NGC_7209','NGC_7243','NGC_752','NGC_6940']

n_per_cpu = np.ceil(1. * len(selected_clusters) / n_cpu)
# n_per_cpu=4

print 'Total number of stars is '+str(len(selected_clusters))+ ', '+str(n_per_cpu)+' per process'

# generate strings to be run
for i_cpu in range(n_cpu):
    run_on = selected_clusters[int(n_per_cpu*i_cpu): int(n_per_cpu*(i_cpu+1))]
    run_string = 'nohup python tgas_clusters.py --clusters='+','.join(run_on)+' --suffix=_'+str(i_cpu+1)+' > cluster_members_run_'+str(i_cpu+1)+'.txt &'
    # run_string = 'nohup python gaia_clusters_sim_dr2.py --clusters='+','.join(run_on)+' --suffix=_'+str(i_cpu+1)+' > cluster_orbits_run_'+str(i_cpu+1)+'.txt &'
    print run_string
    pid = subprocess.Popen(run_string, shell=True)
    print 'PID run:', pid
