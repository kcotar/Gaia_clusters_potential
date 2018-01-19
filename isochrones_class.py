import numpy as np
from astropy.table import Table


class ISOCHRONES():
    """

    """

    def __init__(self, file_path):
        """

        :param file_path: path to fits file with Padova isochrones data
        """
        self.data_all = Table.read(file_path)
        self.isochrone_meh = None
        self.isochrone_age = None
        self.isochrone_data = None

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
