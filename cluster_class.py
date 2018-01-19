import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import astropy.constants as const
import matplotlib.pyplot as plt
from copy import deepcopy
# import gala.coordinates as gal_coord


def _add_galactic_cartesian_data(input_data):
    """

    :param input_data:
    :return:
    """
    output_data = deepcopy(input_data)
    # input ra/dec coordinates of object(s)
    ra_dec = coord.ICRS(ra=input_data['ra'] * un.deg,
                        dec=input_data['dec'] * un.deg,
                        distance=1e3 / input_data['parallax'] * un.pc,
                        pm_ra_cosdec=input_data['pmra'] * un.mas / un.yr,
                        pm_dec=input_data['pmdec'] * un.mas / un.yr,
                        radial_velocity=input_data['rv'] * un.km / un.s)
    # convert to galactic cartesian values
    cartesian_coord = ra_dec.transform_to(coord.Galactic).cartesian
    # store computed positional values back to the
    output_data['x'] = cartesian_coord.x.value
    output_data['y'] = cartesian_coord.y.value
    output_data['z'] = cartesian_coord.z.value
    # get differential (velocities) cartesian values
    cartesian_vel = cartesian_coord.differentials
    cartesian_vel = cartesian_vel[cartesian_vel.keys()[0]]
    # convert them to pc/yr for easier computation
    output_data['d_x'] = cartesian_vel.d_x.to(un.pc / un.yr).value
    output_data['d_y'] = cartesian_vel.d_y.to(un.pc / un.yr).value
    output_data['d_z'] = cartesian_vel.d_z.to(un.pc / un.yr).value
    return output_data


def _cluster_parameters_cartesian(input_data, vel=False, pos=False):
    if not vel and not pos:
        raise KeyError('Determine what to return')

    if vel:
        return [np.nanmedian(input_data['d_x']), np.nanmedian(input_data['d_y']), np.nanmedian(input_data['d_z'])]
    if pos:
        return [np.nanmedian(input_data['x']), np.nanmedian(input_data['y']), np.nanmedian(input_data['z'])]


def _data_stack(input_data, cols):
    """

    :param input_data:
    :param cols:
    :return:
    """
    return input_data[list(cols)].to_pandas().values


def _data_stack_write_back(input_table, cols, input_array):
    """

    :param input_data:
    :param cols:
    :return:
    """
    for i_c in range(len(cols)):
        input_table[cols[i_c]] = input_array[:, i_c]


def _size_vect(vect):
    vect_len = np.sqrt(np.sum(vect ** 2, axis=1))
    return vect_len.reshape(len(vect_len), 1)


def _add_xyz_points(ax, in_data, c='black', s=2, compute_limits=False):
    ax[0, 0].scatter(in_data['x'], in_data['y'], lw=0, s=s, c=c)
    ax[0, 0].set(xlabel='X', ylabel='Y')
    ax[0, 1].scatter(in_data['z'], in_data['y'], lw=0, s=s, c=c)
    ax[0, 1].set(xlabel='Z', ylabel='Y')
    ax[1, 0].scatter(in_data['x'], in_data['z'], lw=0, s=s, c=c)
    ax[1, 0].set(xlabel='X', ylabel='Z')
    if compute_limits:
        _dx = np.max(in_data['x']) - np.min(in_data['x'])
        x_lim = (np.min(in_data['x']) - 0.5 * _dx, np.max(in_data['x']) + 0.5 * _dx)
        _dy = np.max(in_data['y']) - np.min(in_data['y'])
        y_lim = (np.min(in_data['y']) - 0.5 * _dy, np.max(in_data['y']) + 0.5 * _dy)
        _dz = np.max(in_data['z']) - np.min(in_data['z'])
        z_lim = (np.min(in_data['z']) - 0.5 * _dz, np.max(in_data['z']) + 0.5 * _dz)
        ax[0, 0].set(xlim=x_lim, ylim=y_lim)
        ax[0, 1].set(xlim=z_lim, ylim=y_lim)
        ax[1, 0].set(xlim=x_lim, ylim=z_lim)
    return ax


class CLUSTER:
    """

    """

    def __init__(self, meh, age, isochrone, id=None):
        """

        :param meh: in dex
        :param age:
        """
        # input cluster values
        self.cluster_meh = meh
        self.cluster_age = age
        self.cluster_iso = isochrone
        self.id = str(id)
        if not self.cluster_iso._is_isochrone_selected():
            self.cluster_iso.select_isochrone(self.cluster_meh, self.cluster_age)
        # stellar members sets
        self.members = None
        self.members_background = None
        # computed values
        self.cluster_center_pos = None
        self.cluster_center_vel = None
        # other - mass particle variables
        self.particle = None
        self.particle_pos = None

    def init_members(self, members_data):
        """

        :param members_data:
        :return:
        """
        # compute their xyz position and velocities
        self.members = _add_galactic_cartesian_data(members_data)
        # determine mean parameters of the cluster
        self.cluster_center_vel = _cluster_parameters_cartesian(self.members, vel=True)
        self.cluster_center_pos = _cluster_parameters_cartesian(self.members, pos=True)

    def init_background(self, members_data):
        """

        :param members_data:
        :return:
        """
        # compute their xyz position and velocities
        self.members_background = _add_galactic_cartesian_data(members_data)

    def estimate_masses(self):
        """

        :param isochrone_class:
        :return:
        """
        if 'Mass' not in self.members.colnames:
            self.members['Mass'] = np.nan
        # compute mass for every object in the set
        print 'Estimating masses of cluster stars'
        # TODO: correction for absolute magnitudes
        for i_obj in range(len(self.members)):
            iso_mass = self.cluster_iso.detemine_stellar_mass(1e3/self.members[i_obj]['parallax'],
                                                              gmag=self.members[i_obj]['phot_g_mean_mag'])
            self.members[i_obj]['Mass'] = iso_mass
        # print self.members['Mass']

    def _potential_at_coordinates(self, loc_vec,
                                  include_background=False, include_galaxy=False):
        """

        :param loc_vec:
        :param include_background:
        :param include_galaxy:
        :return:
        """
        # TODO: include implementation for potential of background stellar members and galaxy
        if 'Mass' not in self.members.colnames:
            self.estimate_masses()
        mass_memb = _data_stack(self.members, ['Mass'])
        # constant scaled to correct units used elsewhere in the code
        G_const = (const.G.to(un.pc ** 3 / (un.kg * un.yr ** 2)) * const.M_sun).value
        # determine vectors from members to the observed location
        if loc_vec.shape[0] == 1:
            vect_memb = loc_vec - _data_stack(self.members, ['x', 'y', 'z'])
            # compute gravitational potential
            g_vect = -1. * np.nansum(G_const * mass_memb * vect_memb / _size_vect(vect_memb)**3, axis=0)
        else:
            loc_members = _data_stack(self.members, ['x', 'y', 'z'])
            # TODO: faster implementation using array operations instead for loop
            g_vect_list = list([])
            for loc_vec_one in loc_vec:
                vect_memb = loc_vec_one - loc_members
                g_vect_list.append(-1. * np.nansum(G_const * mass_memb * vect_memb / _size_vect(vect_memb) ** 3, axis=0))
            g_vect = np.vstack(g_vect_list)
        return g_vect

    def plot_cluster_xyz(self, path=None):
        """

        :return:
        """
        fig, ax = plt.subplots(2, 2)
        if self.members_background is not None:
            ax = _add_xyz_points(ax, self.members_background, c='black', s=2)
        ax = _add_xyz_points(ax, self.members, c='blue', s=3, compute_limits=True)
        plt.suptitle('Cluster: ' + self.id)
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=350)
        plt.close()

    def plot_cluster_xyz_movement(self, path=None):
        """

        :return:
        """
        fig, ax = plt.subplots(2, 2)
        if self.members_background is not None:
            ax = _add_xyz_points(ax, self.members_background, c='black', s=2)
        ax = _add_xyz_points(ax, self.members, c='blue', s=3, compute_limits=True)
        ax[0, 0].plot(self.particle_pos[:, 0], self.particle_pos[:, 1], lw=1, c='red')
        ax[0, 1].plot(self.particle_pos[:, 2], self.particle_pos[:, 1], lw=1, c='red')
        ax[1, 0].plot(self.particle_pos[:, 0], self.particle_pos[:, 2], lw=1, c='red')
        plt.suptitle('Cluster: '+self.id)
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=350)
        plt.close()

    def init_test_particle(self, input_data):
        self.particle = _add_galactic_cartesian_data(input_data)

    def integrate_stars(self, years=1, step_years=1.,
                        include_background=False, compute_g=False, store_hist=False):
        """

        :param years:
        :param step_years:
        :param include_background:
        :param compute_g:
        :return:
        """
        # TODO: implement complete functionality for keywords include_background, store_hist and compute_g
        pos_init = _data_stack(self.members, ['x', 'y', 'z'])
        vel_init = _data_stack(self.members, ['d_x', 'd_y', 'd_z'])
        obs_year = np.arange(0, years, step_years)
        for i_y in range(len(obs_year)):
            pos_new = pos_init + vel_init * step_years
            pos_init = np.array(pos_new)
            if compute_g:
                g_clust = self._potential_at_coordinates(pos_init)
                vel_new = vel_init + g_clust * step_years
                vel_init = np.array(vel_new)
        # write results back
        _data_stack_write_back(self.members, ['x', 'y', 'z'], pos_init)
        if compute_g:
            _data_stack_write_back(self.members, ['d_x', 'd_y', 'd_z'], vel_init)

    def integrate_particle(self, years=1, step_years=1., store_hist=True,
                           integrate_stars_pos=False, integrate_stars_vel=False):
        """

        :param years:
        :param step_years:
        :param store_hist:
        :param integrate_other_stars:
        :return:
        """
        pos_init = _data_stack(self.particle, ['x', 'y', 'z'])
        vel_init = _data_stack(self.particle, ['d_x', 'd_y', 'd_z'])
        obs_year = np.arange(0, years, step_years)
        if store_hist:
            self.particle_pos = np.zeros((len(obs_year), 3))
        print 'Integrating orbit'
        for i_y in range(len(obs_year)):
            if i_y % 100 == 0:
                print ' Time step: '+str(i_y+1)
            g_clust = self._potential_at_coordinates(pos_init)
            pos_new = pos_init + vel_init * step_years
            vel_new = vel_init + g_clust * step_years
            # set computed values as new initial values
            pos_init = np.array(pos_new)
            vel_init = np.array(vel_new)
            if store_hist:
                self.particle_pos[i_y, :] = pos_init
            # correct positions of stars in cluster and in background
            if integrate_stars_pos:
                # integrate for one time step
                self.integrate_stars(years=step_years, step_years=step_years,
                                     include_background=False, compute_g=integrate_stars_vel, store_hist=False)
