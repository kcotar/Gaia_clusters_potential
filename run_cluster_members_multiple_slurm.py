import subprocess
import os
import numpy as np
import time
from astropy.table import Table
from sys import argv
from getopt import getopt

# WHAT TO RUN
run_membership = False
galah_clusters = True
suffix = ''

if len(argv) > 1:
    # parse input options
    opts, args = getopt(argv[1:], '', ['galah_only=', 'suffix=', 'members='])
    # set parameters, depending on user inputs
    print opts
    for o, a in opts:
        if o == '--galah_only':
            galah_clusters = int(a) > 0
        if o == '--suffix':
            suffix = str(a)
        if o == '--members':
            run_membership = int(a) > 0

# n_cpu = 30
# selected_clusters = [
# "NGC_225","NGC_752","NGC_886","NGC_1039","NGC_1027","Melotte_20","ASCC_10","NGC_1333","NGC_1342","IC_348","Melotte_22","NGC_1528","NGC_1545","NGC_1579","NGC_1582","NGC_1647","NGC_1662","NGC_1708","NGC_1746","ASCC_23","NGC_2281","IC_4665","NGC_6633","IC_4756","NGC_6738","ASCC_100","ASCC_101","NGC_6793","ASCC_103","ASCC_104","ASCC_105","ASCC_106","ASCC_107","NGC_6828","ASCC_109","ASCC_110","NGC_6882","ASCC_112","NGC_6940","NGC_6991A","NGC_6997","ASCC_113","NGC_7058","NGC_7063","NGC_7092","IC_1396","ASCC_114","IC_5146","NGC_7160","ASCC_115","NGC_7243","ASCC_122","ASCC_123","ASCC_124","ASCC_125","NGC_7438","ASCC_126","ASCC_127","ASCC_128","NGC_7762"
# ]

# all clusters from 'NGC', 'IC', 'ASCC', 'Melotte', 'Ruprecht', 'Platais', 'Blanco'
# n_cpu = 20
# selected_clusters = ["NGC_7801","Blanco_1","ASCC_1","ASCC_2","NGC_104","NGC_103","NGC_110","NGC_129","ASCC_3","NGC_133","NGC_136","NGC_146","NGC_189","NGC_225","NGC_188","NGC_288","IC_1590","ASCC_4","ASCC_5","NGC_362","NGC_366","NGC_381","Platais_2","NGC_433","NGC_436","NGC_457","NGC_559","NGC_581","NGC_609","NGC_637","NGC_657","NGC_654","NGC_659","NGC_663","ASCC_6","IC_166","NGC_752","NGC_744","NGC_743","ASCC_7","NGC_869","ASCC_8","NGC_884","NGC_886","NGC_956","IC_1805","NGC_957","NGC_1039","NGC_1027","ASCC_9","IC_1848","NGC_1193","NGC_1252","NGC_1220","NGC_1261","NGC_1245","Melotte_20","ASCC_10","NGC_1333","NGC_1342","ASCC_11","NGC_1348","IC_348","Melotte_22","NGC_1444","NGC_1520","NGC_1496","NGC_1502","NGC_1513","NGC_1557","NGC_1528","IC_361","NGC_1545","NGC_1548","NGC_1579","NGC_1582","NGC_1605","Platais_3","NGC_1641","NGC_1624","NGC_1647","Ruprecht_148","NGC_1662","NGC_1663","ASCC_12","NGC_1664","NGC_1708","NGC_1724","NGC_1746","Platais_4","NGC_1778","NGC_1802","NGC_1807","NGC_1798","NGC_1817","ASCC_13","NGC_1851","NGC_1901","NGC_1857","ASCC_14","ASCC_15","NGC_1893","NGC_1904","ASCC_16","ASCC_17","NGC_1896","NGC_1883","ASCC_18","ASCC_19","NGC_1907","ASCC_20","NGC_1912","ASCC_21","NGC_1931","NGC_1963","NGC_1981","NGC_1976","NGC_1980","NGC_1977","NGC_1960","NGC_1996","NGC_2026","NGC_2039","NGC_2068","NGC_2071","NGC_2099","NGC_2112","NGC_2132","NGC_2129","NGC_2126","NGC_2141","NGC_2143","IC_2157","NGC_2158","NGC_2169","NGC_2168","Platais_5","NGC_2175","NGC_2180","NGC_2183","NGC_2184","NGC_2186","NGC_2189","NGC_2194","ASCC_22","NGC_2192","Platais_6","NGC_2204","NGC_2202","ASCC_23","NGC_2220","NGC_2215","NGC_2219","NGC_2225","NGC_2232","ASCC_24","NGC_2234","NGC_2243","NGC_2236","IC_446","NGC_2244","NGC_2240","NGC_2250","NGC_2252","NGC_2248","NGC_2251","NGC_2254","Ruprecht_1","NGC_2259","NGC_2262","NGC_2264","Ruprecht_2","NGC_2265","Ruprecht_3","NGC_2269","NGC_2266","NGC_2270","ASCC_25","NGC_2287","NGC_2286","NGC_2281","Ruprecht_4","NGC_2298","ASCC_26","NGC_2301","NGC_2302","Ruprecht_149","ASCC_27","ASCC_28","ASCC_29","NGC_2306","NGC_2304","Ruprecht_5","Ruprecht_6","NGC_2309","ASCC_30","NGC_2311","NGC_2312","NGC_2318","NGC_2319","ASCC_31","Ruprecht_8","ASCC_32","Ruprecht_9","NGC_2323","NGC_2348","ASCC_33","NGC_2324","Ruprecht_150","Ruprecht_10","NGC_2335","NGC_2331","Ruprecht_12","Ruprecht_11","NGC_2338","Ruprecht_13","NGC_2343","NGC_2345","ASCC_34","NGC_2349","ASCC_35","NGC_2352","NGC_2351","NGC_2354","ASCC_36","NGC_2353","Ruprecht_14","NGC_2358","NGC_2355","NGC_2360","ASCC_37","NGC_2362","Ruprecht_15","NGC_2367","NGC_2368","Ruprecht_16","Ruprecht_17","NGC_2374","Ruprecht_18","NGC_2383","NGC_2384","Ruprecht_19","Melotte_66","Ruprecht_20","Ruprecht_21","ASCC_38","NGC_2395","NGC_2396","Ruprecht_22","NGC_2401","Ruprecht_23","Ruprecht_24","ASCC_39","NGC_2413","NGC_2414","Ruprecht_40","ASCC_40","NGC_2421","NGC_2422","Ruprecht_25","NGC_2423","Ruprecht_26","Ruprecht_27","Melotte_71","NGC_2419","NGC_2425","NGC_2420","Melotte_72","NGC_2428","NGC_2430","Ruprecht_28","NGC_2439","NGC_2451A","Ruprecht_151","NGC_2432","Ruprecht_29","NGC_2437","Ruprecht_30","Ruprecht_31","NGC_2451B","NGC_2448","NGC_2447","Ruprecht_32","Ruprecht_34","Ruprecht_33","Ruprecht_35","ASCC_41","NGC_2453","Ruprecht_36","NGC_2455","Ruprecht_37","Ruprecht_38","NGC_2459","NGC_2477","NGC_2467","Ruprecht_39","NGC_2467-east","ASCC_42","ASCC_43","Ruprecht_41","Ruprecht_152","NGC_2479","NGC_2482","NGC_2483","NGC_2489","Ruprecht_42","NGC_2516","Ruprecht_44","Ruprecht_43","Ruprecht_45","NGC_2506","Ruprecht_153","NGC_2509","ASCC_44","Ruprecht_154","Ruprecht_46","Ruprecht_47","Ruprecht_48","Ruprecht_49","Ruprecht_50","Ruprecht_51","NGC_2527","Ruprecht_155","Ruprecht_52","NGC_2533","NGC_2547","NGC_2539","Ruprecht_53","Ruprecht_54","NGC_2546","Ruprecht_55","Ruprecht_56","NGC_2548","Ruprecht_58","Ruprecht_57","ASCC_45","ASCC_46","NGC_2567","NGC_2571","Ruprecht_59","Ruprecht_156","NGC_2579","NGC_2580","NGC_2588","NGC_2587","Ruprecht_60","Ruprecht_61","NGC_2609","Ruprecht_157","ASCC_47","Ruprecht_62","Ruprecht_63","ASCC_48","NGC_2627","Ruprecht_64","NGC_2635","Ruprecht_65","NGC_2632","IC_2391","Ruprecht_66","Ruprecht_67","IC_2395","NGC_2659","NGC_2660","NGC_2658","Ruprecht_68","Ruprecht_69","NGC_2670","NGC_2671","NGC_2669","NGC_2664","ASCC_49","Ruprecht_71","ASCC_50","NGC_2682","Ruprecht_72","Ruprecht_158","Ruprecht_73","Platais_8","NGC_2808","Platais_9","NGC_2818","ASCC_51","NGC_2849","Ruprecht_159","Ruprecht_74","Ruprecht_75","NGC_2866","Ruprecht_76","Ruprecht_77","IC_2488","ASCC_52","Ruprecht_78","NGC_2910","NGC_2925","NGC_2932","ASCC_53","NGC_2972","Ruprecht_79","Ruprecht_80","ASCC_54","Ruprecht_81","Ruprecht_82","NGC_3033","Ruprecht_84","Ruprecht_83","NGC_3036","ASCC_55","Ruprecht_160","NGC_3105","Ruprecht_85","Ruprecht_86","NGC_3114","ASCC_56","Ruprecht_161","ASCC_57","ASCC_58","Ruprecht_87","NGC_3201","Ruprecht_88","ASCC_59","NGC_3228","NGC_3255","IC_2581","Ruprecht_89","Ruprecht_90","ASCC_60","NGC_3293","NGC_3324","NGC_3330","Melotte_101","IC_2602","ASCC_61","Ruprecht_91","ASCC_62","NGC_3446","Ruprecht_162","Ruprecht_92","ASCC_63","NGC_3496","ASCC_64","Ruprecht_93","Ruprecht_163","NGC_3532","NGC_3572","ASCC_65","NGC_3590","ASCC_66","NGC_3603","IC_2714","Melotte_105","NGC_3680","Ruprecht_94","Ruprecht_164","IC_2944","NGC_3766","IC_2948","ASCC_67","Ruprecht_95","NGC_3909","NGC_3960","Ruprecht_96","Ruprecht_97","Ruprecht_98","NGC_4052","ASCC_68","Ruprecht_99","Ruprecht_100","ASCC_69","NGC_4103","Ruprecht_101","NGC_4147","Ruprecht_102","ASCC_70","Ruprecht_103","NGC_4230","ASCC_71","Melotte_111","NGC_4337","NGC_4349","Ruprecht_104","NGC_4372","NGC_4439","Ruprecht_165","NGC_4463","ASCC_72","Ruprecht_105","ASCC_73","Ruprecht_106","NGC_4590","NGC_4609","NGC_4755","NGC_4815","NGC_4833","NGC_4852","NGC_5024","NGC_5045","NGC_5043","NGC_5053","Ruprecht_107","NGC_5139","NGC_5138","NGC_5155","NGC_5168","Ruprecht_108","ASCC_74","Platais_10","NGC_5272","NGC_5269","NGC_5286","NGC_5281","NGC_5284","ASCC_75","NGC_5288","NGC_5299","ASCC_76","Platais_12","NGC_5316","NGC_5359","NGC_5381","NGC_5466","Ruprecht_110","NGC_5460","ASCC_77","Ruprecht_167","NGC_5593","NGC_5606","NGC_5634","NGC_5617","IC_1023","NGC_5662","Ruprecht_111","NGC_5694","NGC_5715","NGC_5749","NGC_5764","Ruprecht_112","IC_4499","NGC_5800","NGC_5824","NGC_5822","ASCC_78","NGC_5823","NGC_5897","NGC_5904","ASCC_79","ASCC_80","NGC_5925","NGC_5927","NGC_5946","NGC_5986","ASCC_81","ASCC_82","NGC_5998","ASCC_83","NGC_5999","ASCC_84","NGC_6005","Ruprecht_113","NGC_6025","Ruprecht_114","NGC_6031","Ruprecht_115","NGC_6067","Ruprecht_176","NGC_6093","NGC_6087","Ruprecht_116","Ruprecht_117","NGC_6121","Ruprecht_118","NGC_6124","NGC_6101","NGC_6144","NGC_6139","NGC_6134","Ruprecht_119","NGC_6171","NGC_6152","NGC_6169","NGC_6167","Ruprecht_120","NGC_6178","NGC_6192","NGC_6193","NGC_6205","Ruprecht_121","NGC_6200","NGC_6204","NGC_6229","NGC_6218","ASCC_85","NGC_6216","NGC_6208","NGC_6222","NGC_6235","NGC_6231","NGC_6242","NGC_6254","NGC_6249","NGC_6250","NGC_6253","NGC_6256","NGC_6259","NGC_6266","ASCC_86","NGC_6268","NGC_6273","ASCC_87","NGC_6284","NGC_6281","NGC_6287","ASCC_88","NGC_6293","NGC_6304","NGC_6318","NGC_6316","NGC_6341","NGC_6325","NGC_6322","NGC_6333","NGC_6334","NGC_6342","ASCC_89","Ruprecht_123","NGC_6356","NGC_6355","NGC_6357","IC_4651","NGC_6360","NGC_6352","IC_1257","NGC_6366","Ruprecht_124","Ruprecht_125","NGC_6362","NGC_6380","NGC_6383","Ruprecht_126","NGC_6388","NGC_6402","NGC_6396","Ruprecht_127","NGC_6401","ASCC_90","NGC_6404","NGC_6400","NGC_6405","NGC_6397","Ruprecht_128","NGC_6416","NGC_6426","IC_4665","NGC_6425","Ruprecht_129","Ruprecht_130","NGC_6440","ASCC_91","Ruprecht_131","NGC_6444","NGC_6441","NGC_6451","NGC_6453","ASCC_92","NGC_6455","Ruprecht_133","Ruprecht_134","Ruprecht_168","NGC_6481","NGC_6469","NGC_6475","NGC_6480","NGC_6494","Ruprecht_135","NGC_6496","Ruprecht_136","NGC_6507","Ruprecht_138","Ruprecht_137","Ruprecht_139","NGC_6517","NGC_6525","NGC_6514","NGC_6520","NGC_6522","NGC_6535","NGC_6531","NGC_6530","NGC_6528","NGC_6539","NGC_6540","NGC_6544","NGC_6546","NGC_6541","ASCC_93","NGC_6554","NGC_6553","NGC_6558","NGC_6561","IC_1276","NGC_6568","NGC_6569","NGC_6573","ASCC_94","NGC_6583","ASCC_95","NGC_6596","NGC_6604","NGC_6605","NGC_6603","NGC_6584","NGC_6611","NGC_6613","ASCC_96","NGC_6588","NGC_6618","NGC_6625","NGC_6624","NGC_6626","Ruprecht_170","NGC_6631","NGC_6633","NGC_6638","NGC_6639","Ruprecht_141","NGC_6637","NGC_6647","IC_4725","NGC_6642","Ruprecht_142","Ruprecht_171","Ruprecht_143","NGC_6645","NGC_6649","Ruprecht_144","NGC_6659","NGC_6652","NGC_6656","NGC_6664","ASCC_97","IC_4756","NGC_6682","NGC_6683","ASCC_98","NGC_6681","NGC_6694","NGC_6698","ASCC_99","Ruprecht_145","NGC_6704","NGC_6705","NGC_6709","Ruprecht_146","NGC_6712","NGC_6716","NGC_6715","NGC_6717","NGC_6724","NGC_6728","NGC_6723","NGC_6735","NGC_6738","NGC_6743","ASCC_100","NGC_6737","NGC_6749","NGC_6755","NGC_6756","NGC_6752","NGC_6760","ASCC_101","NGC_6779","Ruprecht_147","NGC_6775","NGC_6791","NGC_6793","ASCC_102","NGC_6800","NGC_6802","ASCC_103","NGC_6811","ASCC_104","NGC_6809","NGC_6819","ASCC_105","NGC_6823","ASCC_106","NGC_6832","ASCC_107","NGC_6827","NGC_6828","NGC_6830","NGC_6834","NGC_6837","NGC_6838","ASCC_108","ASCC_109","NGC_6839","NGC_6840","NGC_6843","NGC_6846","NGC_6856","NGC_6858","ASCC_110","NGC_6866","NGC_6871","NGC_6864","NGC_6873","IC_1311","ASCC_111","NGC_6883","Ruprecht_172","NGC_6882","NGC_6885","ASCC_112","NGC_6895","IC_4996","Melotte_227","NGC_6904","NGC_6910","NGC_6913","NGC_6939","NGC_6934","NGC_6940","NGC_6938","NGC_6950","Ruprecht_173","Ruprecht_174","Ruprecht_175","NGC_6981","NGC_6989","NGC_6991A","NGC_6997","NGC_7006","NGC_7024","NGC_7031","NGC_7036","NGC_7039","NGC_7037","ASCC_113","IC_1369","NGC_7044","NGC_7050","NGC_7055","NGC_7058","NGC_7062","NGC_7063","NGC_7067","NGC_7082","NGC_7078","Platais_1","NGC_7086","NGC_7092","NGC_7084","NGC_7089","NGC_7093","IC_1396","ASCC_114","NGC_7099","NGC_7129","NGC_7127","NGC_7128","NGC_7142","IC_5146","NGC_7160","ASCC_115","ASCC_116","NGC_7175","ASCC_117","NGC_7209","ASCC_118","NGC_7226","IC_1434","NGC_7235","NGC_7243","NGC_7245","IC_1442","ASCC_119","NGC_7261","NGC_7281","NGC_7296","ASCC_120","ASCC_121","ASCC_122","ASCC_123","NGC_7380","ASCC_124","NGC_7419","NGC_7423","NGC_7429","ASCC_125","NGC_7438","ASCC_126","ASCC_127","NGC_7492","NGC_7510","NGC_7538","ASCC_128","NGC_7654","ASCC_129","NGC_7686","NGC_7708","NGC_7772","ASCC_130","NGC_7788","NGC_7789","NGC_7790","NGC_7795","NGC_7762"]

# manual selection of the best cluster candidates
# n_cpu = 5
# selected_clusters = ['NGC_129','NGC_189','NGC_225','NGC_366','NGC_188','NGC_381','NGC_663','NGC_6694','NGC_6704','NGC_6709','NGC_6735','NGC_6866','NGC_7142','NGC_7209','NGC_7243','NGC_752','NGC_6940']

# manual selection of the best GALAH cluster candidates
# n_cpu = 5
# selected_clusters = ['NGC_225','ASCC_4','ASCC_5','Platais_2','NGC_752','NGC_886','NGC_1039','NGC_1027','Melotte_20','ASCC_10','NGC_1333','NGC_1342','ASCC_11','IC_348','Melotte_22','NGC_1528','NGC_1545','NGC_1579','NGC_1582','Platais_3','NGC_1647','NGC_1662','ASCC_12','NGC_1708','NGC_1746','Platais_4','ASCC_13','ASCC_16','ASCC_18','ASCC_19','ASCC_20','ASCC_21','NGC_1981','NGC_1976','NGC_1980','NGC_1977','NGC_2068','NGC_2071','NGC_2112','NGC_2168','NGC_2184','ASCC_22','Platais_6','ASCC_23','NGC_2232','ASCC_24','NGC_2252','NGC_2264','NGC_2281','ASCC_26','NGC_2301','ASCC_28','ASCC_29','ASCC_30','NGC_2319','ASCC_31','NGC_2323','ASCC_34','ASCC_35','ASCC_38','ASCC_41','NGC_2548','NGC_2632','NGC_2682','Melotte_111','ASCC_100','ASCC_101','NGC_6793','ASCC_103','ASCC_104','ASCC_105','ASCC_107','NGC_6828','ASCC_109','ASCC_110','NGC_6882','ASCC_112','NGC_6940','NGC_6991A','NGC_6997','ASCC_113','NGC_7058','NGC_7063','NGC_7092','IC_1396','ASCC_114','IC_5146','NGC_7160','ASCC_115','NGC_7243','ASCC_122','ASCC_123','ASCC_124','ASCC_125','NGC_7438','ASCC_126','ASCC_127','ASCC_128','NGC_7762']
# suffix = ''

if not galah_clusters:
    # All C-G 2018 clusters
    if run_membership:
        n_cpu = 10
    else:
        n_cpu = 40
    data_dir = '/shared/ebla/cotar/'
    khar_dir = data_dir + 'clusters/Cantat-Gaudin_2018/'
    # read Cantat-Gaudin_(2018) clusters data
    clusters = Table.read(khar_dir + 'table1.fits')
    # remove trailing whitespaces in original cluster names
    selected_clusters = [str(clusters['cluster'][i_l]).lstrip().rstrip() for i_l in range(len(clusters))]
    root_suffix = '_ALL'
else:
    # All GALAH clusters
    if run_membership:
        n_cpu = 4
    else:
        n_cpu = 20
    cluster_dir = '/shared/ebla/cotar/clusters/'
    clusters = Table.read(cluster_dir + 'members_open_gaia_r2.fits')
    # remove trailing whitespaces in original cluster names
    selected_clusters = [str(cc).lstrip().rstrip() for cc in np.unique(clusters['cluster'])]
    root_suffix = '_GALAH_CGmebers'

n_per_cpu = np.ceil(1. * len(selected_clusters) / n_cpu)
# n_per_cpu=4

print 'Total number of clusters is '+str(len(selected_clusters))+', '+str(n_per_cpu)+' per process'

# generate strings to be run
for i_cpu in range(n_cpu):
    sh_file = 'run_gaia_clusters' + root_suffix + '.sh'
    txt_file = open(sh_file, 'w')

    run_on = selected_clusters[int(n_per_cpu*i_cpu): int(n_per_cpu*(i_cpu+1))]
    if run_membership:
        run_string = 'python tgas_clusters.py --r=1 --c='+','.join(run_on)+' --s='+suffix+'_{:02.0f}'.format(i_cpu+1)+' --d='+root_suffix
        run_log = 'cluster_members_run_{:02.0f}'.format(i_cpu+1) + root_suffix
    else:
        run_string = 'python gaia_clusters_sim_dr2.py --r=1 --c='+','.join(run_on)+' --s='+suffix+'{:02.0f}'.format(i_cpu+1)+' --d='+root_suffix
        run_log = 'cluster_orbits_run_{:02.0f}'.format(i_cpu + 1) + root_suffix
    print run_string

    txt_file.write('#!/bin/bash \n')
    txt_file.write('#\n')
    # txt_file.write('#SBATCH --partition=rude \n')
    # txt_file.write('#SBATCH --qos=rude \n')
    txt_file.write('#SBATCH --partition=astro \n')
    txt_file.write('#SBATCH --qos=astro \n')
    txt_file.write('#SBATCH --nodes=1 \n')
    if run_membership:
        txt_file.write('#SBATCH --tasks-per-node=4 \n')
    else:
        txt_file.write('#SBATCH --tasks-per-node=2 \n')
    txt_file.write('#SBATCH --mem=15G \n')
    txt_file.write('#SBATCH --time=2-00:00 \n')
    txt_file.write('#SBATCH -o logs/'+run_log+'.out \n')
    txt_file.write('#SBATCH -e logs/'+run_log+'.err \n')
    txt_file.write('#SBATCH --nodelist=astro01 \n')
    txt_file.write('\n')
    txt_file.write('export PYTHONHTTPSVERIFY=0 \n')
    # txt_file.write('export OMP_NUM_THREADS=1 \n')
    txt_file.write('\n')
    txt_file.write(run_string+' \n')
    txt_file.close()

    # run script in slurm
    os.system('sbatch ' + sh_file)

    # wait few second
    if i_cpu < n_cpu-1:
        time.sleep(60)
