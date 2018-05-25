import numpy as np
from astropy.table import Table
from copy import deepcopy


class ISOCHRONES():
    """

    """

    def __init__(self, file_path, photo_system='UBVRIJHK'):
        """

        :param file_path:
        :param photo_system: Can be UBVRIJHK or Gaia
        """
        self.data_all = Table.read(file_path)
        self.isochrone_meh = None
        self.isochrone_age = None
        self.isochrone_data = None
        self.system = photo_system

    def _is_isochrone_selected(self):
        """

        :return:
        """
        return self.isochrone_data is not None

    def select_isochrone(self, meh, age):
        """
        Determine isochrone that fits best to the selected input parameters

        :param meh:
        :param age:
        :return:
        """
        meh_uniq = np.unique(self.data_all['MHini'])
        self.isochrone_meh = meh_uniq[np.argmin(np.abs(meh_uniq - meh))]

        age_uniq = np.unique(self.data_all['Age'])
        self.isochrone_age = age_uniq[np.argmin(np.abs(age_uniq - age))]

        self.isochrone_data = self.data_all[np.logical_and(self.data_all['MHini'] == self.isochrone_meh,
                                                           self.data_all['Age'] == self.isochrone_age)]
        self.isochrone_data['Mloss'] = self.isochrone_data['Mini'] - self.isochrone_data['Mass']

    def detemine_stellar_mass(self, parsec_dist, teff=None, logg=None,
                              gmag=None, gbpmag=None, grpmag=None):
        """
        Determine mass off observed star based on its input parameters - photometric and spectroscopic physical.

        :param teff:
        :param logg:
        :param gmag:
        :param gbpmag:
        :param grpmag:
        :return:
        """

        if not self._is_isochrone_selected():
            raise ValueError('Isochrone not selected')

        # selection based on Gmag only - experimental first try for dr51
        if gmag is None:
            raise ValueError('Gmag not given')
        gmag_abs = gmag - 2.5*np.log10((parsec_dist/10.)**2)
        idx_iso = np.argsort(np.abs(self.isochrone_data['Gmag'] - gmag_abs))[:2]
        # get point point between the nearest ones
        d_frac = (self.isochrone_data['Gmag'][idx_iso[0]] - gmag_abs) / (self.isochrone_data['Gmag'][idx_iso[0]] - self.isochrone_data['Gmag'][idx_iso[1]])
        # print d_frac
        # print self.isochrone_data['Gmag'][idx_iso]
        # print self.isochrone_data['Mass'][idx_iso]
        mass = self.isochrone_data['Mass'][idx_iso[0]] - d_frac * (self.isochrone_data['Mass'][idx_iso[0]] - self.isochrone_data['Mass'][idx_iso[1]])
        return mass

    def get_hr_magnitudes_data(self, max_Mini=None, max_Mloss=None, cluster_dist=100):
        """

        :param max_Mini: in solar mass
        :param max_Mloss: in solar mass
        :param cluster_dist: in parsecs
        :return:
        """

        isochrone_data_sub = deepcopy(self.isochrone_data)

        # select the maximum initial stellar mas
        if max_Mini is not None:
            isochrone_data_sub = isochrone_data_sub[isochrone_data_sub['Mini'] < max_Mini]

        # select the maximum stellar mas loss
        if max_Mloss is not None:
            isochrone_data_sub = isochrone_data_sub[isochrone_data_sub['Mloss'] < max_Mloss]

        # correct magnitudes for cluster distance as the are give in absolute mag (@10pc) in isochrone
        if self.system == 'UBVRIJHK':
            b_mag = isochrone_data_sub['Bmag'] + 2.5*np.log10((cluster_dist/10.)**2)
            v_mag = isochrone_data_sub['Vmag'] + 2.5*np.log10((cluster_dist/10.)**2)
            x_data = b_mag - v_mag
            y_data = v_mag
        elif self.system == 'Gaia':
            x_data = isochrone_data_sub['G_BPmag'] - isochrone_data_sub['G_RPmag']
            y_data = isochrone_data_sub['Gmag'] + 2.5 * np.log10((cluster_dist / 10.) ** 2)

        return x_data, y_data

