import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 5})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from scipy.spatial import ConvexHull, Delaunay
import astropy.coordinates as coord
import astropy.units as un


def _prepare_hist_data(d, bins, range, norm=True):
    """

    :param d:
    :param bins:
    :param range:
    :param norm:
    :return:
    """
    heights, edges = np.histogram(d, bins=bins, range=range)
    width = np.abs(edges[0] - edges[1])
    if norm:
        heights = 1.*heights / np.max(heights)
    return edges[:-1], heights, width


def plot_abundances_histograms(cluster_data, other_data=None,
                               use_flag=True, path=None):
    """

    :param cluster_data:
    :param other_data:
    :param use_flag:
    :param path:
    :return:
    """

    abund_cols = cluster_data.colnames
    abund_cols = [col for col in abund_cols if '_abund_cannon' in col and 'e_' not in col and 'flag_' not in col and len(col.split('_'))<4]
    abund_names = [col.split('_')[0] for col in abund_cols]
    # generate distribution plots independently for every chemical abundance
    n_abund = len(abund_cols)
    fig, ax = plt.subplots(5, 6)  # correction for more abundances in cannon3 data release
    fig.set_size_inches(7, 7)
    # plot range
    x_range = (-1.5, 0.8)
    y_range = (0, 1)
    for i_a in range(n_abund):
        subplot_x = i_a % 6
        subplot_y = int(i_a / 6)
        # hist background
        if other_data is not None:
            plot_data = other_data[abund_cols[i_a]]
            if use_flag:
                plot_data[other_data['flag_' + abund_cols[i_a]] > 0] = np.nan
            h_edg, h_hei, h_wid = _prepare_hist_data(plot_data, 50, x_range)
            ax[subplot_y, subplot_x].bar(h_edg, h_hei, width=h_wid, color='black', alpha=0.25)
        plot_data = cluster_data[abund_cols[i_a]]
        # hist interesting data
        if use_flag:
            plot_data[cluster_data['flag_'+abund_cols[i_a]] > 0] = np.nan
        n_non_flag = np.sum(np.isfinite(plot_data))
        h_edg, h_hei, h_wid = _prepare_hist_data(plot_data, 50, x_range)
        ax[subplot_y, subplot_x].bar(h_edg, h_hei, width=h_wid, color='red', alpha=0.35)
        # make it nice
        ax[subplot_y, subplot_x].set(title=abund_names[i_a]+' ['+str(n_non_flag)+']', xlim=x_range, ylim=y_range)
        ax[subplot_y, subplot_x].grid(True, color='black', linestyle='dashed', linewidth=0.5, alpha=0.15)

    if other_data is not None:
        n_non_flag_other = np.sum(np.isfinite(other_data[abund_cols].to_pandas().values))
        plt.suptitle('Cluster GALAH members: {:.0f},   target unflagged abund: {:.0f}'.format(len(cluster_data), n_non_flag_other))
    else:
        plt.suptitle('Cluster GALAH members: {:.0f}'.format(len(cluster_data)))

    if path is None:
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.2, left=0.05, bottom=0.05, right=0.95, top=0.92)
        plt.show()
    else:
        plt.subplots_adjust(hspace=0.6, wspace=0.3, left=0.05, bottom=0.05, right=0.95, top=0.88)
        plt.savefig(path, dpi=400)
    plt.close()


def plot_orbits(input_obj_data, other_data=None, path=None):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ts = np.linspace(0, -200., 2000) * un.Myr

    def get_orbit(in_obj):
        if np.isfinite(in_obj['rv']):
            # print source_id
            orbit = Orbit(vxvv=[in_obj['ra'] * un.deg,
                                in_obj['dec'] * un.deg,
                                1e3 / in_obj['parallax'] * un.pc,
                                in_obj['pmra'] * un.mas / un.yr,
                                in_obj['pmdec'] * un.mas / un.yr,
                                in_obj['rv'] * un.km / un.s], radec=True)
            orbit.turn_physical_on()
            orbit.integrate(ts, MWPotential2014)
            return orbit
        else:
            return None

    def add_orbit_plot(orb, c, a):
        ax[0, 0].plot(orb.x(ts), orb.y(ts), lw=0.5, c=c, alpha=a)
        ax[0, 1].plot(orb.x(ts), orb.z(ts), lw=0.5, c=c, alpha=a)
        ax[1, 0].plot(orb.z(ts), orb.y(ts), lw=0.5, c=c, alpha=a)

    if other_data is not None:
        for in_obj in other_data[:100]:  # maximum of first 100 orbits in vicinity - otherwise can be to slow
            orbit = get_orbit(in_obj)
            if orbit is not None:
                add_orbit_plot(orbit, 'black', a=0.2)

    for in_obj in input_obj_data:
        orbit = get_orbit(in_obj)
        if orbit is not None:
            add_orbit_plot(orbit, 'red', a=0.3)

    ax[0, 0].set(xlim=(-10, 10), ylim=(-10, 10))
    ax[0, 1].set(xlim=(-10, 10), ylim=(-1, 1))
    ax[1, 0].set(xlim=(-1, 1), ylim=(-10, 10))
    plt.tight_layout()
    if path is None:
        plt.show(fig)
    else:
        plt.savefig(path, dpi=300)
    plt.close(fig)


def get_cartesian_coords(input_data):
    ra_dec = coord.ICRS(ra=input_data['ra'] * un.deg,
                        dec=input_data['dec'] * un.deg,
                        distance=1e3 / input_data['parallax'] * un.pc)
    # convert to galactic cartesian values
    cartesian_coord = ra_dec.transform_to(coord.Galactocentric).cartesian
    # store computed positional values back to the
    return np.transpose(np.vstack((cartesian_coord.x.value, cartesian_coord.y.value, cartesian_coord.z.value)))


def get_objects_inside_selection(sel_data, all_data):
    # determine convex hull vertices from members points
    idx_hull_vert = ConvexHull(sel_data).vertices
    # create a Delaunay grid based on given points
    delanuay_surf = Delaunay(sel_data[idx_hull_vert])
    inside_hull = delanuay_surf.find_simplex(all_data) >= 0
    return np.where(inside_hull)[0]


def plot_3D_pos(memb_pos, back_pos=None, path=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(memb_pos[:, 0], memb_pos[:, 1], memb_pos[:, 2], lw=0, s=15, c='red')
    if back_pos is not None:
        ax.scatter(back_pos[:, 0], back_pos[:, 1], back_pos[:, 2], lw=0, s=10,  c='black')
    # save or show plot
    plt.tight_layout()
    if path is None:
        plt.show(fig)
    else:
        plt.savefig(path, dpi=250)
    plt.close(fig)


def plot_tsne_sel(tsne_data, sel=None, ins=None, path=None):
    plt.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0., s=0.5, c='#bebebe', alpha=0.5)
    if sel is not None:
        idx_s = np.in1d(tsne_data['source_id'], sel)
        plt.scatter(tsne_data['tsne_axis_1'][idx_s], tsne_data['tsne_axis_2'][idx_s], lw=0., s=1.2, c='red', alpha=1.)
    if ins is not None:
        idx_s = np.in1d(tsne_data['source_id'], ins)
        plt.scatter(tsne_data['tsne_axis_1'][idx_s], tsne_data['tsne_axis_2'][idx_s], lw=0., s=1.2, c='blue', alpha=1.)
    # save or show plot
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=250)
    plt.close()


def plt_vel_distr(data, other_data=None, path=None):
    fig, ax = plt.subplots(1, 3)

    if other_data is not None:
        h_edg, h_hei, h_wid = _prepare_hist_data(other_data['rv'], 50, (-100, 100))
        ax[0].bar(h_edg, h_hei, width=h_wid, color='black', alpha=0.2)
        h_edg, h_hei, h_wid = _prepare_hist_data(other_data['pmra'], 50, (-45., 45.))
        ax[1].bar(h_edg, h_hei, width=h_wid, color='black', alpha=0.2)
        h_edg, h_hei, h_wid = _prepare_hist_data(other_data['pmdec'], 50, (-45., 45))
        ax[2].bar(h_edg, h_hei, width=h_wid, color='black', alpha=0.2)

    h_edg, h_hei, h_wid = _prepare_hist_data(data['rv'], 50, (-100, 100))
    ax[0].bar(h_edg, h_hei, width=h_wid, color='red', alpha=0.4)
    h_edg, h_hei, h_wid = _prepare_hist_data(data['pmra'], 50, (-45., 45.))
    ax[1].bar(h_edg, h_hei, width=h_wid, color='red', alpha=0.4)
    h_edg, h_hei, h_wid = _prepare_hist_data(data['pmdec'], 50, (-45., 45.))
    ax[2].bar(h_edg, h_hei, width=h_wid, color='red', alpha=0.4)

    ax[1].set(title='PmRA')
    ax[0].set(title='RV')
    ax[2].set(title='PmDEC')

    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)
    plt.close()