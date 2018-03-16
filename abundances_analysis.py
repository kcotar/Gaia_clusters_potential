import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 5})
import matplotlib.pyplot as plt

from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
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
        plot_data = cluster_data[abund_cols[i_a]]
        if use_flag:
            plot_data[cluster_data['flag_'+abund_cols[i_a]] > 0] = np.nan
        n_non_flag = np.sum(np.isfinite(plot_data))
        h_edg, h_hei, h_wid = _prepare_hist_data(plot_data, 50, x_range)
        ax[subplot_y, subplot_x].bar(h_edg, h_hei, width=h_wid, color='C1', alpha=0.8)
        # hist interesting data
        if other_data is not None:
            plot_data = other_data[abund_cols[i_a]]
            if use_flag:
                plot_data[other_data['flag_' + abund_cols[i_a]] > 0] = np.nan
            h_edg, h_hei, h_wid = _prepare_hist_data(plot_data, 50, x_range)
            ax[subplot_y, subplot_x].bar(h_edg, h_hei, width=h_wid, color='red', alpha=0.6)
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


def plot_orbits(input_obj_data, path=None):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for in_obj in input_obj_data:
        # print source_id
        orbit = Orbit(vxvv=[in_obj['ra'] * un.deg,
                            in_obj['dec'] * un.deg,
                            1e3 / in_obj['parallax'] * un.pc,
                            in_obj['pmra'] * un.mas / un.yr,
                            in_obj['pmdec'] * un.mas / un.yr,
                            in_obj['rv'] * un.km / un.s], radec=True)
        orbit.turn_physical_on()

        ts = np.linspace(0, -200., 2000) * un.Myr
        orbit.integrate(ts, MWPotential2014)

        ax[0, 0].plot(orbit.x(ts), orbit.y(ts), lw=0.5, c='red', alpha=0.3)
        ax[0, 1].plot(orbit.x(ts), orbit.z(ts), lw=0.5, c='red', alpha=0.3)
        ax[1, 0].plot(orbit.z(ts), orbit.y(ts), lw=0.5, c='red', alpha=0.3)

    if path is None:
        plt.show(fig)
        plt.close(fig)
    else:
        plt.savefig(path, dpi=250)
        plt.close(fig)
