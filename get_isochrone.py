import numpy as np
import os
import requests
import re
from copy import deepcopy


def one_iso(age, feh, av, magnitudes, n_zams=15, **kwargs):
    isochrones = []
    zams = []
    order = []
    teff = []
    logg = []
    mass = []
    for i in magnitudes.values():
        isochrone = _get_one_iso(i[0], i[1], age, feh, av, **kwargs)
        isochrones.append(isochrone[2])
        teff.append(isochrone[4])
        logg.append(isochrone[5])
        mass.append(isochrone[0])
        zams_tmp = []
        for age_tmp in np.logspace(4.5, 10.0, num=n_zams):
            isochrone = _get_one_iso(i[0], i[1], age_tmp, feh, av, **kwargs)
            if len(np.where(isochrone[1] > 0)[0]) == 0:
                zams_index = -1
            else:
                zams_index = np.where(isochrone[1] > 0)[0][0]
            zams_tmp.append(isochrone[2][zams_index])
        order.append(i[2])
        zams.append(zams_tmp)

    # reorder isochrones to the same order as magnitudes are in the data file
    isochrones = np.array(isochrones)
    order = np.array(order)
    zams = np.array(zams)
    teff = np.array(teff)
    logg = np.array(logg)
    mass = np.array(mass)
    order_inds = order.argsort()
    isochrones = isochrones[order_inds]
    isochrones = isochrones.T
    zams = zams[order_inds]
    zams = zams.T
    teff = teff[order_inds]
    teff = teff.T
    logg = logg[order_inds]
    logg = logg.T
    mass = mass[order_inds]
    mass = mass.T

    return isochrones, zams, teff, logg, mass


def _get_one_iso(photsys_file, mag, age, z, a_v, z_log=True, version='3.1'):
    # if metallicity is given as a log ([M/H]) convert it here:
    z_tmp = deepcopy(z)
    if z_log:
        if float(version) < 3.3:
            z_tmp = 0.0152 * np.power(10, z)
        else:
            # Ages / metallicities Choose your metallicity values using the approximation
            # [M / H] = log(Z / X) - log(Z / X)o, with (Z / X)o=0.0207 and Y=0.2485+1.78Z for PARSEC tracks.
            return []

    mass = []
    label = []
    mags = []
    imf = []
    teff = []
    logg = []

    # parameters other than default
    d = {
        'cmd_version': version,
        'isoc_kind': 'parsec_CAF09_v1.2S',
        'track_parsec': 'parsec_CAF09_v1.2S',
        'track_colibri': 'parsec_CAF09_v1.2S_S35',
        'track_postagb': 'no',
        'n_inTPC': 10,
        'eta_reimers': 0.2,
        'kind_interp': 1,
        'kind_postagb': -1,
        'kind_tpagb': 0,
        'kind_pulsecycle': 0,
        'photsys_file': photsys_file,
        'photsys_version': 'yang',
        'kind_cspecmag': 'aringer09',
        'dust_sourceM': 'dpmod60alox40',
        'dust_sourceC': 'AMCSIC15',
        'kind_mag': 2,
        'kind_dust': 0,
        'extinction_av': a_v,
        'extinction_coeff': 'constant',
        'extinction_curve': 'cardelli',
        'imf_file': 'tab_imf/imf_kroupa_orig.dat',
        'isoc_val': 0,  # single isochron
        'isoc_isagelog': 0,
        'isoc_age': age,
        'isoc_agelow': age,
        'isoc_zeta': z_tmp,
        'isoc_zlow': z_tmp,
        'output_kind': 0,
        'output_evstage': 1,
        'submit_form': 'Submit',
        # M/H for 3.3
        'isoc_ismetlog': 0,  # 0 -> log Z, 1 -> M/H values
        'isoc_metlow': -2.,
        'isoc_metupp': 0.2,
        'isoc_dmet': 0,  # 0 -> single value
        # TODO: more fields from version 3.3
        # 'isoc_lage0': 6.6,
        # 'isoc_lagelow': 6.6,
        # 'isoc_lage1': 10.13,
        # 'isoc_lageupp': 10.13,
    }

    if float(version) > 3.1:
        d['photsys_version'] = 'YBC'

    # Check if we already downloaded this isochrone.
    # Isochrones are saved as txt files and the filename is the hash of the dictionary values.
    d_hash = hash(''.join([str(i) for i in d.values()]))
    use_dir = '/shared/data-camelot/cotar/downloaded_isochrones/'
    if os.path.isfile(use_dir + '%s' % d_hash):
        in_file = open(use_dir + '%s' % d_hash, 'r')
        r = in_file.read()
        in_file.close()
    else:
        webserver = 'http://stev.oapd.inaf.it'
        c = requests.get(webserver + '/cgi-bin/cmd_'+version, params=d).text
        aa = re.compile('output\d+')
        fname = aa.findall(c)
        if len(fname) > 0:
            url = '{0}/tmp/{1}.dat'.format(webserver, fname[0])
            # print url
            r = requests.get(url).text
            out_file = open(use_dir + '%s' % d_hash, 'w')
            out_file.write(r)
            out_file.close()

    # parse the output
    r = r.split('\n')
    for line in r:
        # get column names
        if len(line) > 50 and line[0] == '#' and line.split()[1] == 'Z' and line.split()[2] == 'log(age/yr)':
            mag_idx = line.split().index(mag) - 1
            mass_idx = 3
            label_idx = -1
            imf_idx = -2
            teff_idx = 5
            logg_idx = 6
        elif len(line) > 10 and line[0] != '#':
            data = line.split()
            if float(data[mass_idx]) in mass:
                continue  # sometimes one mass appears twice in a row.
            mass.append(float(data[mass_idx]))
            label.append(int(data[label_idx]))
            mags.append(float(data[mag_idx]))
            imf.append(float(data[imf_idx]))
            teff.append(float(data[teff_idx]))
            logg.append(float(data[logg_idx]))

    mass_new = np.linspace(mass[0], mass[-1], len(mass) * 20)
    label = np.array(np.interp(mass_new, mass, label), dtype=int)
    mags = np.interp(mass_new, mass, mags)
    imf = np.interp(mass_new, mass, imf)
    teff = np.interp(mass_new, mass, teff)
    logg = np.interp(mass_new, mass, logg)

    return np.array([mass_new, label, mags, imf, teff, logg])
