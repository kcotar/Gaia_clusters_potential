import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import astropy.constants as const
import matplotlib.pyplot as plt
from copy import deepcopy
from galpy.potential import MWPotential2014, evaluatePotentials


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
    # cartesian_coord = ra_dec.transform_to(coord.Galactic).cartesian
    cartesian_coord = ra_dec.transform_to(coord.Galactocentric).cartesian
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
    if len(vect.shape) == 1:
        vect_len = np.sqrt(np.sum(vect ** 2))
        return vect_len
    else:
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


def _add_xyz_plots(ax, in_data, c='black', lw=2, alpha=1., compute_limits=False):
    # TODO
    pass


def _galactic_potential(r_gal, z_gal):
    m_sun = 2e30  # kg
    G_val = const.G.to(un.km ** 3 / (un.kg * un.s ** 2)).value  # km3 kg-1 s-2
    kpc_to_km = (un.kpc).to(un.km)

    def _spherical_halo(r):
        v_h = 220.  # km/s
        r_0 = 5.5  # kpc
        return 1./2. * v_h**2 * np.log(r**2 + r_0**2)  # r conversion pc -> kpc

    def _central_bulge(r):
        m_1 = 3e9 * m_sun  # kg
        r_1 = 2.7  # kpc
        m_2 = 1.6e10 * m_sun  # kg
        r_2 = 0.42  # kpc
        return -1.*G_val*m_1/(np.sqrt(r**2 + r_1**2)*kpc_to_km) - 1.*G_val*m_2/(np.sqrt(r**2 + r_2**2)*kpc_to_km)

    def _disk(r, z, a, b, M):
        return -1. * G_val * M * m_sun / (np.sqrt(r**2 + (a + np.sqrt(z**2 + b**2)))*kpc_to_km)

    gal_pot = _spherical_halo(r_gal) + _central_bulge(r_gal)  # km2/s2
    gal_pot += _disk(r_gal, z_gal, 2.18, 0.2, 2.5e10) + _disk(r_gal, z_gal, 3.05, 0.65, 0.65e10) +\
               _disk(r_gal, z_gal, 9.5, 0.18, 1.18e10) + _disk(r_gal, z_gal, 1.52, 0.1, 0.23e10)  # km2/s2
    kms_to_pcyr = (un.km ** 2 / un.s ** 2).to(un.pc ** 2 / un.yr ** 2)
    return gal_pot * kms_to_pcyr


def _get_gravity_accel(x, y, z, galpy_pot=False):
    r_gal = np.sqrt(x ** 2 + y ** 2)/1e3  # in pc -> kpc
    z_gal = z/1e3  # in pc -> kpc as required by the function
    dr_gal = 2.5e-4  # in kpc
    if galpy_pot:
        dpot = (evaluatePotentials(MWPotential2014, (r_gal + dr_gal)*un.kpc, z_gal*un.kpc, phi=0, t=0, dR=0) - evaluatePotentials(MWPotential2014, (r_gal - dr_gal)*un.kpc, z_gal*un.kpc, phi=0, t=0, dR=0))
        kms_to_pcyr = (un.km ** 2 / un.s ** 2).to(un.pc ** 2 / un.yr ** 2)
        dpot *= kms_to_pcyr
    else:
        dpot = (_galactic_potential(r_gal + dr_gal, z_gal) - _galactic_potential(r_gal - dr_gal, z_gal))
    dpot_dr = dpot / (2.*dr_gal*1e3)  # dr_gal converted to pc as dpot is returned in pc2/yr2
    return -1.*dpot_dr  # pc/yr2

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
        self.cluster_memb_pos = None
        self.background_memb_pos = None

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
        def _nonzero_sep_only(in_2d_a):
            return (in_2d_a != 0).all(axis=1)

        # TODO: include implementation for potential of background stellar members and galaxy
        if 'Mass' not in self.members.colnames:
            self.estimate_masses()
        mass_memb = _data_stack(self.members, ['Mass'])
        # potenconversionsrion const
        kms_to_pcyr = (un.km**2/un.s**2).to(un.pc**2/un.yr**2)
        # constant scaled to correct units used elsewhere in the code
        G_const = (const.G.to(un.pc ** 3 / (un.kg * un.yr ** 2)) * const.M_sun).value
        # determine vectors from members to the observed location
        loc_members = _data_stack(self.members, ['x', 'y', 'z'])
        if loc_vec.shape[0] == 1:
            vect_memb = loc_vec - loc_members
            idx_star_use = _nonzero_sep_only(vect_memb)
            # compute gravitational potential
            g_vect = -1. * np.nansum((G_const * mass_memb * vect_memb / _size_vect(vect_memb)**3)[idx_star_use], axis=0)
            if include_galaxy:
                g_pot = _get_gravity_accel(loc_vec[:, 0], loc_vec[:, 1], loc_vec[:, 2])

                # print g_pot
                # g_pot = evaluatePotentials(MWPotential2014, np.sqrt(loc_vec[:, 0]**2 + loc_vec[:, 1]**2)*un.pc,
                #                            loc_vec[:, 2]*un.pc) * kms_to_pcyr  # coordinates in parsecs
                g_vect += (g_pot * loc_vec / _size_vect(loc_vec))[0]  # vector from center of Galaxy
        else:
            # TODO: faster implementation using array operations instead for loop
            g_vect_list = list([])
            for loc_vec_one in loc_vec:
                vect_memb = loc_vec_one - loc_members
                idx_star_use = _nonzero_sep_only(vect_memb)
                g_vect_list.append(-1. * np.nansum((G_const * mass_memb * vect_memb / _size_vect(vect_memb) ** 3)[idx_star_use], axis=0))
            g_vect = np.vstack(g_vect_list)
            if include_galaxy:
                g_pot = _get_gravity_accel(loc_vec[:, 0], loc_vec[:, 1], loc_vec[:, 2])
                # g_pot = evaluatePotentials(MWPotential2014, np.sqrt(loc_vec[:, 0] ** 2 + loc_vec[:, 1] ** 2)*un.pc,
                #                            loc_vec[:, 2]*un.pc)  # coordinates in parsecs
                g_pot = g_pot.repeat(3).reshape(loc_vec.shape[0], 3)
                g_vect += (g_pot * loc_vec / _size_vect(loc_vec))  # vector from center of Galaxy
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
        plt.grid(ls='--', alpha=0.5, color='black')
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
        # background members
        if self.members_background is not None:
            ax = _add_xyz_points(ax, self.members_background, c='black', s=2)
        # cluster members
        ax = _add_xyz_points(ax, self.members, c='blue', s=3, compute_limits=False)
        if self.cluster_memb_pos is not None:
            for i_m in range(len(self.members)):
                ax[0, 0].plot(self.cluster_memb_pos[:, i_m, 0], self.cluster_memb_pos[:, i_m, 1], lw=0.5, c='red', alpha=0.3)
                ax[0, 1].plot(self.cluster_memb_pos[:, i_m, 2], self.cluster_memb_pos[:, i_m, 1], lw=0.5, c='red', alpha=0.3)
                ax[1, 0].plot(self.cluster_memb_pos[:, i_m, 0], self.cluster_memb_pos[:, i_m, 2], lw=0.5, c='red', alpha=0.3)
        # investigated particle movement
        ax[0, 0].plot(self.particle_pos[:, 0], self.particle_pos[:, 1], lw=1, c='green')
        ax[0, 1].plot(self.particle_pos[:, 2], self.particle_pos[:, 1], lw=1, c='green')
        ax[1, 0].plot(self.particle_pos[:, 0], self.particle_pos[:, 2], lw=1, c='green')
        # TEMPORARY: axis limits selection
        ax[0, 0].set(xlim=(-10e3, 10e3), ylim=(-10e3, 10e3))
        ax[0, 1].set(xlim=(-10e3, 10e3), ylim=(-10e3, 10e3))
        ax[1, 0].set(xlim=(-10e3, 10e3), ylim=(-10e3, 10e3))
        plt.suptitle('Cluster: '+self.id)
        plt.tight_layout()
        plt.grid(ls='--', alpha=0.5, color='black')
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=400)
        plt.close()

    def init_test_particle(self, input_data):
        self.particle = _add_galactic_cartesian_data(input_data)

    def _integrate_stars(self, years=1, step_years=1., include_galaxy=False,
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
                g_clust = self._potential_at_coordinates(pos_init, include_galaxy=include_galaxy)
                vel_new = vel_init + g_clust * step_years
                vel_init = np.array(vel_new)
            if store_hist:
                self.cluster_memb_pos.append(pos_init)
        # write results back
        _data_stack_write_back(self.members, ['x', 'y', 'z'], pos_init)
        if compute_g:
            _data_stack_write_back(self.members, ['d_x', 'd_y', 'd_z'], vel_init)

    def integrate_particle(self, years=1, step_years=1., store_hist=True,
                           integrate_stars_pos=False, integrate_stars_vel=False, include_galaxy_pot=False):
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
            self.cluster_memb_pos = list([])
        print 'Integrating orbit'
        for i_y in range(len(obs_year)):
            if i_y % 100 == 0:
                print ' Time step: '+str(i_y+1), 'out of '+str(len(obs_year))
            g_clust = self._potential_at_coordinates(pos_init, include_galaxy=include_galaxy_pot)
            # print pos_init, 'pc'
            # print vel_init, 'pc/yr'
            # print g_clust
            # print step_years
            # print g_clust * step_years, 'pc/yr'
            # print np.sqrt(np.sum(vel_init**2))
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
                self._integrate_stars(years=step_years, step_years=step_years, include_galaxy=include_galaxy_pot,
                                     include_background=False, compute_g=integrate_stars_vel, store_hist=store_hist)
        if store_hist and integrate_stars_pos:
            # stack computed coordinates along new axis
            self.cluster_memb_pos = np.stack(self.cluster_memb_pos)
        else:
            self.cluster_memb_pos = None

