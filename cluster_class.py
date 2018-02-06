import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import astropy.constants as const
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from galpy.potential import MWPotential2014, evaluatePotentials
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Point, MultiPoint
from time import time
from multiprocessing import Pool
from functools import partial
np.seterr(invalid='ignore')


def _add_galactic_cartesian_data(input_data, reverse=False, gal_coor=True):
    """

    :param input_data:
    :param reverse:
    :param gal_coor:
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
    if not gal_coor:
        cartesian_coord = ra_dec.transform_to(coord.Galactic).cartesian
    else:
        cartesian_coord = ra_dec.transform_to(coord.Galactocentric).cartesian
    # store computed positional values back to the
    output_data['x'] = cartesian_coord.x.value
    output_data['y'] = cartesian_coord.y.value
    output_data['z'] = cartesian_coord.z.value
    # get differential (velocities) cartesian values
    cartesian_vel = cartesian_coord.differentials
    cartesian_vel = cartesian_vel[cartesian_vel.keys()[0]]
    # convert them to pc/yr for easier computation
    if reverse:
        vel_multi = -1.
    else:
        vel_multi = 1.
    output_data['d_x'] = vel_multi * cartesian_vel.d_x.to(un.pc / un.yr).value
    output_data['d_y'] = vel_multi * cartesian_vel.d_y.to(un.pc / un.yr).value
    output_data['d_z'] = vel_multi * cartesian_vel.d_z.to(un.pc / un.yr).value

    return output_data


def _cluster_parameters_cartesian(input_data, vel=False, pos=False):
    """

    :param input_data:
    :param vel:
    :param pos:
    :return:
    """
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
    """

    :param vect:
    :return:
    """
    if len(vect.shape) == 1:
        vect_len = np.sqrt(np.sum(vect ** 2))
        return vect_len
    else:
        vect_len = np.sqrt(np.sum(vect ** 2, axis=1))
        return vect_len.reshape(len(vect_len), 1)


def _add_xyz_points(ax, in_data, c='black', s=2, compute_limits=False):
    """

    :param ax:
    :param in_data:
    :param c:
    :param s:
    :param compute_limits:
    :return:
    """
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
    """

    :param ax:
    :param in_data:
    :param c:
    :param lw:
    :param alpha:
    :param compute_limits:
    :return:
    """
    # TODO
    pass


def _galactic_potential(r_gal, z_gal):
    """

    :param r_gal:
    :param z_gal:
    :return:
    """
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
    """

    :param x:
    :param y:
    :param z:
    :param galpy_pot:
    :return:
    """
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


def _integrate_pos_vel(pos, vel, g, t, method='newt'):
    """

    :param pos:
    :param vel:
    :param g:
    :param t:
    :param method:
    :return:
    """
    # TODO: add more sophisticated and faster methods to do this
    if 'newt' in method:
        pos_new = pos + vel * t
        vel_new = vel + g * t
    elif 'rk4' in method:
        pass
        # TODO: harder to implement in current setup of functions
        # k1 = acceleration(R_init)
        # k2 = acceleration(R_init + 0.5 * k1)
        # k3 = acceleration(R_init + 0.5 * k2)
        # k4 = acceleration(R_init + k3)
        # # print k1, k2, k3, k4
        # a_star = 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        # V_new = V_init + a_star * t_step_s
        # R_new = R_init + V_init * t_step_s
        # R_init = np.array(R_new)
        # V_init = np.array(V_new)
        # r_RK4_2[i_t] = size_vect(R_new) / AU
    return pos_new, vel_new


def _nonzero_sep_only(in_2d_a):
    return (in_2d_a != 0).all(axis=1)


def _func_g_vect_one(loc_vec_one, loc_members, mass_memb, G_const):
    vect_memb = loc_vec_one - loc_members
    idx_star_use = _nonzero_sep_only(vect_memb)
    return -1. * np.nansum((G_const * mass_memb * vect_memb / _size_vect(vect_memb) ** 3)[idx_star_use], axis=0)


class CLUSTER:
    """

    """

    def __init__(self, meh, age, isochrone, id=None, reverse=False, galactocentric=True, verbose=False):
        """

        :param meh:
        :param age:
        :param isochrone:
        :param id:
        :param reverse:
        :param galactocentric:
        :param verbose:
        """
        self.verbose = verbose
        # input cluster values
        self.cluster_meh = meh
        self.cluster_age = age
        self.cluster_iso = isochrone
        self.reverse_orbit = reverse
        self.galactic_coord = galactocentric
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
        # orbital positions of objects
        self.cluster_memb_pos = None
        self.background_memb_pos = None
        # orbital crosses
        self.final_inside_hull = None
        # integration settings
        self.step_years = None

    def init_members(self, members_data):
        """

        :param members_data:
        :return:
        """
        # compute their xyz position and velocities
        self.members = _add_galactic_cartesian_data(members_data, reverse=self.reverse_orbit, gal_coor=self.galactic_coord)
        # determine mean parameters of the cluster
        self.cluster_center_vel = _cluster_parameters_cartesian(self.members, vel=True)
        self.cluster_center_pos = _cluster_parameters_cartesian(self.members, pos=True)

    def init_background(self, members_data):
        """

        :param members_data:
        :return:
        """
        # compute their xyz position and velocities
        self.members_background = _add_galactic_cartesian_data(members_data, reverse=self.reverse_orbit, gal_coor=self.galactic_coord)

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
                                  include_background=False, include_galaxy=False, no_interactions=False):
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
        loc_members = _data_stack(self.members, ['x', 'y', 'z'])
        if loc_vec.shape[0] == 1:
            vect_memb = loc_vec - loc_members
            idx_star_use = _nonzero_sep_only(vect_memb)
            # compute gravitational potential
            if no_interactions:
                g_vect = 0.
            else:
                g_vect = -1. * np.nansum((G_const * mass_memb * vect_memb / _size_vect(vect_memb)**3)[idx_star_use], axis=0)
            if include_galaxy:
                g_pot = _get_gravity_accel(loc_vec[:, 0], loc_vec[:, 1], loc_vec[:, 2], galpy_pot=False)
                g_vect += (g_pot * loc_vec / _size_vect(loc_vec))[0]  # vector from center of Galaxy
        else:
            # TODO: faster implementation using array operations instead for loop
            if no_interactions:
                g_vect = 0.
            else:
                # TEMP: palatalization attempts
                # g_vect_list = Parallel(n_jobs=2, mmap_mode='r', backend='threading')(delayed(_func_g_vect_one)(loc_vec_one, loc_members, mass_memb, G_const) for loc_vec_one in loc_vec)
                pool = Pool(processes=2)  # greatest speedup with 2 processes
                _func_g_vect_one_partial = partial(_func_g_vect_one, loc_members=loc_members, mass_memb=mass_memb, G_const=G_const)
                g_vect_list = pool.map(_func_g_vect_one_partial, loc_vec)
                pool.close()

                # g_vect_list = list([])
                # for loc_vec_one in loc_vec:
                #     vect_memb = loc_vec_one - loc_members
                #     idx_star_use = _nonzero_sep_only(vect_memb)
                #     g_vect_list.append(-1. * np.nansum((G_const * mass_memb * vect_memb / _size_vect(vect_memb) ** 3)[idx_star_use], axis=0))
                g_vect = np.vstack(g_vect_list)

            if include_galaxy:
                g_pot = _get_gravity_accel(loc_vec[:, 0], loc_vec[:, 1], loc_vec[:, 2], galpy_pot=False)
                g_pot = g_pot.repeat(3).reshape(loc_vec.shape[0], 3)
                g_vect += (g_pot * loc_vec / _size_vect(loc_vec))  # vector from center of Galaxy
        return g_vect

    def plot_cluster_xyz(self, path=None):
        """

        :return:
        """
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        if self.members_background is not None:
            ax = _add_xyz_points(ax, self.members_background, c='black', s=2)
        ax = _add_xyz_points(ax, self.members, c='blue', s=3, compute_limits=True)
        plt.suptitle('Cluster: ' + self.id)
        plt.tight_layout()
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.98, top=0.96, wspace=0.15, hspace=0.1)
        plt.grid(ls='--', alpha=0.5, color='black')
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=350)
        plt.close()

    def plot_cluster_xyz_movement(self, idx_obj_only=None, path=None, source_id=None, sobject_id=None):
        """

        :param idx_obj_only:
        :param path:
        :param source_id:
        :param sobject_id:
        :return:
        """
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
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
        if self.particle_pos is not None:
            if idx_obj_only is not None:
                i_m_to_plot = [idx_obj_only]
            elif source_id is not None:
                idx_obj_only = np.where(self.particle['source_id'] == source_id)[0]
                i_m_to_plot = [idx_obj_only]
            else:
                i_m_to_plot = range(len(self.particle))
            for i_m in i_m_to_plot:
                ax[0, 0].plot(self.particle_pos[:, i_m, 0], self.particle_pos[:, i_m, 1], lw=0.5, c='green', alpha=0.3)
                ax[0, 1].plot(self.particle_pos[:, i_m, 2], self.particle_pos[:, i_m, 1], lw=0.5, c='green', alpha=0.3)
                ax[1, 0].plot(self.particle_pos[:, i_m, 0], self.particle_pos[:, i_m, 2], lw=0.5, c='green', alpha=0.3)
        # TEMPORARY: axis limits selection
        ax[0, 0].set(xlim=(-10e3, 10e3), ylim=(-10e3, 10e3))
        ax[0, 1].set(xlim=(-10e3, 10e3), ylim=(-10e3, 10e3))
        ax[1, 0].set(xlim=(-10e3, 10e3), ylim=(-10e3, 10e3))
        plt.suptitle('Cluster: '+self.id)
        plt.tight_layout()
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.98, top=0.96, wspace=0.15, hspace=0.1)
        plt.grid(ls='--', alpha=0.5, color='black')
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=350)
        plt.close()

    def init_test_particle(self, input_data):
        self.particle = _add_galactic_cartesian_data(input_data, reverse=self.reverse_orbit, gal_coor=self.galactic_coord)

    def _integrate_stars(self, years=1, step_years=1., include_galaxy=False,
                        include_background=False, compute_g=False, store_hist=False, no_interactions=False):
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
            if compute_g:
                g_clust = self._potential_at_coordinates(pos_init, include_galaxy=include_galaxy, no_interactions=no_interactions)
            else:
                g_clust = 0.
            # integrate particle for given time step in years
            pos_init, vel_init = _integrate_pos_vel(pos_init, vel_init, g_clust, step_years, method='newt')
            if store_hist:
                self.cluster_memb_pos.append(pos_init)
        # write results back
        _data_stack_write_back(self.members, ['x', 'y', 'z'], pos_init)
        if compute_g:
            _data_stack_write_back(self.members, ['d_x', 'd_y', 'd_z'], vel_init)

    def integrate_particle(self, years=1, step_years=1., store_hist=True,
                           integrate_stars_pos=False, integrate_stars_vel=False, include_galaxy_pot=False,
                           disable_interactions=False):
        """

        :param years:
        :param step_years:
        :param store_hist:
        :param integrate_stars_pos:
        :param integrate_stars_vel:
        :param include_galaxy_pot:
        :param disable_interactions:
        :return:
        """
        self.step_years = step_years
        #
        pos_init = _data_stack(self.particle, ['x', 'y', 'z'])
        vel_init = _data_stack(self.particle, ['d_x', 'd_y', 'd_z'])
        obs_year = np.arange(0, years, step_years)
        if store_hist:
            self.particle_pos = list([])
            self.cluster_memb_pos = list([])
        print 'Integrating orbit'
        for i_y in range(len(obs_year)):
            if i_y % 100 == 0:
                print ' Time step: '+str(i_y+1), 'out of '+str(len(obs_year))
            time_s = time()
            g_clust = self._potential_at_coordinates(pos_init, include_galaxy=include_galaxy_pot, no_interactions=disable_interactions)
            # integrate particle for given time step in years
            pos_init, vel_init = _integrate_pos_vel(pos_init, vel_init, g_clust, step_years, method='newt')
            if store_hist:
                self.particle_pos.append(pos_init)
            # correct positions of stars in cluster and in background
            if integrate_stars_pos:
                # integrate for one time step
                self._integrate_stars(years=step_years, step_years=step_years, include_galaxy=include_galaxy_pot,
                                      include_background=False, compute_g=integrate_stars_vel, store_hist=store_hist, no_interactions=disable_interactions)
            if self.verbose:
                print 'Step integ s:', time()-time_s

        if store_hist and integrate_stars_pos:
            # stack computed coordinates along new axis
            self.particle_pos = np.stack(self.particle_pos)
            self.cluster_memb_pos = np.stack(self.cluster_memb_pos)
        else:
            self.particle_pos = None
            self.cluster_memb_pos = None

    def determine_orbits_that_cross_cluster(self, method='scipy'):
        """

        :return:
        """
        # number of time steps that the simulation was run for
        print 'Determining crossing orbits'
        n_steps = self.particle_pos.shape[0]
        n_parti = self.particle_pos.shape[1]
        # analyse every step
        self.final_inside_hull = np.full((n_steps, n_parti), False)
        for i_s in range(n_steps):
            if method is 'shapely':
                # VERSION1: implementation using shapley - works only in 2D
                # create an convex hull out of cluster members positions
                points_obj = MultiPoint(self.cluster_memb_pos[i_s, :, :].tolist())  # conversion to preserve z coordinate
                hull_obj = points_obj.convex_hull
                # investigate which points are in the hull
                inside_hull = [hull_obj.contains(Point(particle_coord.tolist())) for particle_coord in self.particle_pos[i_s, :, :]]
                self.final_inside_hull[i_s, :] = np.array(inside_hull)
            else:
                # VERSION2: using scipy ConvexHull and Delaunay teseltation
                # determine convex hull vertices from members points
                idx_hull_vert = ConvexHull(self.cluster_memb_pos[i_s, :, :]).vertices
                # create a Delaunay grid based on given points
                delanuay_surf = Delaunay(self.cluster_memb_pos[i_s, :, :][idx_hull_vert])
                inside_hull = delanuay_surf.find_simplex(self.particle_pos[i_s, :, :]) >= 0
                self.final_inside_hull[i_s, :] = np.array(inside_hull)

    def get_crossing_objects(self, min_time=None):
        """

        :param min_time:
        :return:
        """
        if self.final_inside_hull is None:
            # first run the analysis of crossing orbits
            self.determine_orbits_that_cross_cluster()
        time_in_cluster = np.sum(self.final_inside_hull, axis=0) * np.abs(self.step_years)
        if min_time is not None:
            idx_ret = time_in_cluster > min_time
        else:
            idx_ret = time_in_cluster > 0
        return self.particle[idx_ret]['source_id']

    def plot_crossing_orbits(self, plot_prefix=None):
        """

        :param plot_prefix:
        :return:
        """
        print 'Plotting crossing orbis'
        if self.final_inside_hull is None:
            # first run the analysis of crossing orbits
            self.determine_orbits_that_cross_cluster()
        # determine what is to be plotted
        idx_to_plot = np.where(np.sum(self.final_inside_hull, axis=0) > 0)[0]
        # perform plots
        for idx_obj in idx_to_plot:
            if plot_prefix is not None:
                plot_path = plot_prefix + '_{:04.0f}.png'.format(idx_obj)
            else:
                plot_path = None
            self.plot_cluster_xyz_movement(idx_obj_only=idx_obj, path=plot_path)

    def animate_particle_movement(self, path='video.mp4', sobject_id=None, source_id=None, t_step=None):
        """

        :param path:
        :param sobject_id:
        :param source_id:
        :param t_step:
        :return:
        """
        if sobject_id is None and source_id is None:
            print 'Object to animate not defined'
            return

        print 'Creating animation'

        def _update_graph(i_s):
            # TODO: is there any speedup if array subset at time i_s is read only once?
            graph_memb._offsets3d = (self.cluster_memb_pos[i_s, :, 0],
                                     self.cluster_memb_pos[i_s, :, 1],
                                     self.cluster_memb_pos[i_s, :, 2])
            graph_part._offsets3d = (self.particle_pos[i_s, idx_particle, 0],
                                     self.particle_pos[i_s, idx_particle, 1],
                                     self.particle_pos[i_s, idx_particle, 2])
            # TODO: better/faster definition of limits
            # x_lim = (np.min(self.cluster_memb_pos[i_s, :, 0]), np.max(self.cluster_memb_pos[i_s, :, 0]))
            # y_lim = (np.min(self.cluster_memb_pos[i_s, :, 1]), np.max(self.cluster_memb_pos[i_s, :, 1]))
            # z_lim = (np.min(self.cluster_memb_pos[i_s, :, 2]), np.max(self.cluster_memb_pos[i_s, :, 2]))
            # d_x = (x_lim[1] - x_lim[0]) / 2. * 0.3
            # d_y = (y_lim[1] - y_lim[0]) / 2. * 0.3
            # d_z = (z_lim[1] - z_lim[0]) / 2. * 0.3
            # ax.set(xlim=(x_lim[0]-d_x, x_lim[1]+d_x),
            #        ylim=(y_lim[0]-d_y, y_lim[1]+d_y),
            #        zlim=(z_lim[0]-d_z, z_lim[1]+d_z))
            # view_angle = np.rad2deg(np.arctan2((y_lim[1] + y_lim[0]) / 2., (x_lim[1] + x_lim[0]) / 2.)) - 180.
            d_axis = 80  # pc
            ax.set(xlim=(self.particle_pos[i_s, idx_particle, 0] - d_axis, self.particle_pos[i_s, idx_particle, 0] + d_axis),
                   ylim=(self.particle_pos[i_s, idx_particle, 1] - d_axis, self.particle_pos[i_s, idx_particle, 1] + d_axis),
                   zlim=(self.particle_pos[i_s, idx_particle, 2] - d_axis, self.particle_pos[i_s, idx_particle, 2] + d_axis))
            view_angle = np.rad2deg(np.arctan2(self.particle_pos[i_s, idx_particle, 1], self.particle_pos[i_s, idx_particle, 0]))[0] - 180.
            ax.view_init(elev=10, azim=view_angle)
            title.set_text('Time = {:.1f} Myr'.format(i_s*self.step_years/1e6, view_angle))
            # set colour of the title according to the position of particle inside cluster
            if self.final_inside_hull[i_s, idx_particle] >= 1:
                plt.setp(title, color='red')
            else:
                plt.setp(title, color='black')

        # get idx of particle
        if source_id is not None:
            idx_particle = np.where(self.particle['source_id'] == source_id)[0]
            if len(idx_particle) == 0:
                print ' Skipping animation. Given source_id ('+str(source_id)+') not found between integrated particles'
                return

        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=20)
        fig = plt.figure()
        ax = Axes3D(fig)
        title = ax.set_title('')
        graph_memb = ax.scatter([], [], [], lw=0, s=10, c='black')
        graph_part = ax.scatter([], [], [], lw=0, s=20, c='red', marker='*')

        n_time_integ = self.particle_pos.shape[0]
        # determine time interval between video frames
        if t_step is None:
            t_step = self.step_years
        # determine number odf frames based on their interval
        n_steps = np.int64(n_time_integ*self.step_years / t_step)  # for the whole range of orbit simulation
        movement_anim = animation.FuncAnimation(fig, _update_graph,
                                                frames=np.int64(np.linspace(0, n_time_integ-1, n_steps)))
        movement_anim.save(path, writer, dpi=150)
        plt.close()


    # --------------------------------------------------
    # ---------- ONLY TESTS BELLOW THIS POINT ----------
    # --------------------------------------------------

    def _test_convex_hull(self, idx_time, idx_particle):
        memb_data = self.cluster_memb_pos[idx_time, :, :]
        part_data = self.particle_pos[idx_time, idx_particle, :][0]
        mp = MultiPoint(memb_data.tolist())
        p = Point(part_data.tolist())
        h = mp.convex_hull
        print 'Is inside: ', h.contains(p)
        # lets see what strange is happening here
        # not even needed any more as the library manual states:
        # A third z coordinate value may be used when constructing instances, but has no effect
        # on geometric analysis. All operations are performed in the x-y plane.



