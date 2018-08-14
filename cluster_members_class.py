import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import astropy.constants as const
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from copy import deepcopy
from sklearn import mixture
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull, Delaunay
from sklearn.neighbors import KernelDensity
from skimage.feature import peak_local_max
# from astropy.modeling import models, fitting
from gaussian2d_lmfit import *
from scipy.stats import norm
from scipy.stats import multivariate_normal


# -------------------------------
# other useful functions
# -------------------------------


# -------------------------------
# class functions
# -------------------------------
def get_gauss_prob(vals, v_mean, v_std):
    return 0.5 + norm.cdf(-np.abs(np.array(vals) - v_mean), 0., v_std)


# -------------------------------
# class implementation
# -------------------------------
class CLUSTER_MEMBERS:
    """
    Class intended for the selection of probable open cluster members based on Gaia data and
    ancillary spectroscopic data sets
    """
    def __init__(self, in_obs, clust_data):

        # inputs
        self.data = deepcopy(in_obs)
        self.ref = deepcopy(clust_data)

        # some useful computations that are need multiple times
        self.pm_center = np.array([np.float64(self.ref['pmRAc']), np.float64(self.ref['pmDEc'])])
        self.data['parsec'] = 1e3/self.data['parallax']
        self.data['center_sep'] = coord.ICRS(ra=self.data['ra'] * un.deg,
                                             dec=self.data['dec'] * un.deg).separation(coord.ICRS(ra=self.ref['RAdeg'] * un.deg,
                                                                                                  dec=self.ref['DEdeg'] * un.deg))

        # plot parameters
        self.pl_xlim = (np.percentile(self.data['pmra'], 3), np.percentile(self.data['pmra'], 97))
        self.pl_ylim = (np.percentile(self.data['pmdec'], 3), np.percentile(self.data['pmdec'], 97))
        self.parsec_lim = (np.percentile(self.data['parsec'], 1), np.percentile(self.data['parsec'], 99))

        # results holders
        self.cluster_g2d_params = None
        self.cluster_dist_params = None
        self.cluster_pos_params = None

        self.pm_probability = None
        self.dist_probability = None
        self.pos_probability = None

        self.selected_final = None
        self.selected = np.zeros(len(self.data))
        self.n_runs = 0

    def perform_selection_density(self, rad, suffix='', n_runs=1):
        """

        :param rad:
        :param suffix:
        :param n_runs:
        :return:
        """
        print 'Density analysis of pm space'
        idx_sel = self.data['center_sep'] <= rad
        data_cur = self.data[idx_sel]

        n_data_sel = len(data_cur)
        if n_data_sel < 5:
            print ' Not enough data points in selected radius'
            return

        pm_plane_orig = np.vstack((data_cur['pmra'].data, data_cur['pmdec'].data)).T
        pm_plane_errors = np.vstack((data_cur['pmra_error'].data, data_cur['pmra_error'].data)).T
        # determine ranges to remove outlying data points
        x_range = np.percentile(pm_plane_orig[:, 0], [2., 98.])
        y_range = np.percentile(pm_plane_orig[:, 1], [2., 98.])
        d_xy = 0.05
        if (x_range[1]-x_range[0])/d_xy > 2e3 or (y_range[1]-y_range[0])/d_xy > 2e3:
            # to reduce processing time in the case of large x or y axis range
            d_xy = 0.1
        # print '  Ranges:', x_range, y_range, '  (d_pm - {:.2f})'.format(d_xy)

        final_list_g2d_params = list([])
        for i_run in np.arange(n_runs)+1:
            print ' Creating new random pm plane based on observations'
            pm_plane = list([])
            for i_pm in range(pm_plane_orig.shape[0]):
                pm_plane.append([np.random.normal(pm_plane_orig[i_pm, 0], pm_plane_errors[i_pm, 0]),
                                 np.random.normal(pm_plane_orig[i_pm, 1], pm_plane_errors[i_pm, 1])])
            pm_plane = np.array(pm_plane)
            pm_plane_median = np.median(pm_plane, axis=0)
            pm_plane_median_px = np.int32((pm_plane_median - np.array([x_range[0], y_range[0]])) / d_xy)[::-1]  # also swap x and y

            grid_pos_x = np.arange(x_range[0], x_range[1], d_xy)
            grid_pos_y = np.arange(y_range[0], y_range[1], d_xy)
            # print '  Grid points:', len(grid_pos_x), len(grid_pos_y)
            _x, _y = np.meshgrid(grid_pos_x, grid_pos_y)

            print ' Computing density field'
            # TODO: maybe check wider bandwidths
            stars_density = KernelDensity(bandwidth=1., kernel='epanechnikov').fit(pm_plane)
            density_field = stars_density.score_samples(np.vstack((_x.ravel(), _y.ravel())).T)
            # density_field += np.log(pm_plane.shape[0])
            density_field = np.exp(density_field).reshape(_x.shape) * 1e3  # scale the field for easier use

            # find and evaluate peaks if needed
            peak_coord_init = peak_local_max(density_field, min_distance=int(1. / d_xy), num_peaks=5)
            # add aditional peak in the middle of the image if only one
            if peak_coord_init.shape[0] == 1:
                peak_coord_init = np.vstack((peak_coord_init, [int((y_range[1]-y_range[0])/d_xy/2.),
                                                               int((x_range[1]-x_range[0])/d_xy/2.)]))
            # add a peak in te middle of the distribution if it is not present there or in its vicinity
            if np.min(np.sqrt(np.sum((peak_coord_init - pm_plane_median_px)**2, axis=1))) > 1./d_xy:
                peak_coord_init = np.vstack((peak_coord_init, pm_plane_median_px))
            # image scaling from py to pixel space
            ceter_pm_img_x = (self.pm_center[0] - x_range[0]) / d_xy
            ceter_pm_img_y = (self.pm_center[1] - y_range[0]) / d_xy

            # prepare peak coordinates and peak values
            lm_peak = list([])
            lm_peak_val = list([])
            for i_p in range(peak_coord_init.shape[0]):
                lm_peak.append([peak_coord_init[i_p, 1] * d_xy + x_range[0], peak_coord_init[i_p, 0] * d_xy + y_range[0]])  # convert to pm values
                lm_peak_val.append(density_field[peak_coord_init[i_p, 0], peak_coord_init[i_p, 1]])
            density_field_background, multi_gauss2d_res = fit_multi_gaussian2d(density_field, _x, _y,
                                                                               np.array(lm_peak), np.array(lm_peak_val),
                                                                               vary_position=True)

            # remove the background
            density_field_new = density_field - density_field_background

            # extract new peak centers end evaluate results
            final_peaks_pm = list([])
            for i_p in range(peak_coord_init.shape[0]):
                final_peaks_pm.append([multi_gauss2d_res['x'+str(i_p)].value, multi_gauss2d_res['y'+str(i_p)].value,
                                       multi_gauss2d_res['xs'+str(i_p)].value, multi_gauss2d_res['ys'+str(i_p)].value,
                                       multi_gauss2d_res['amp' + str(i_p)].value])
            final_peaks_pm = np.array(final_peaks_pm)
            final_peaks_pm_dist = np.sqrt((final_peaks_pm[:,0]-self.pm_center[0])**2 + (final_peaks_pm[:,1]-self.pm_center[1])**2)
            # evaluate the closest peak(s)
            idx_peak_close = np.where(final_peaks_pm_dist < 10.)[0]
            n_peak_close = len(idx_peak_close)
            idx_peak_closest = np.argmin(final_peaks_pm_dist)
            idx_peak_sort = np.argsort(final_peaks_pm_dist)
            print '  Number of close peaks detected: ', n_peak_close
            if n_peak_close > 1:
                # V2
                # determine ok and reorder by distance
                max_sigma = 1.3
                min_amp = np.percentile(density_field, 80.)
                idx_ok = np.logical_and(np.logical_and(final_peaks_pm[:, 2] < max_sigma, final_peaks_pm[:, 3] < max_sigma),
                                        final_peaks_pm[:, 4] > min_amp)[idx_peak_sort]
                idx_ok = np.where(idx_ok)[0]
                # select first arg that is ok
                if len(idx_ok) > 0:
                    idx_peak_sel = idx_peak_sort[idx_ok[0]]
                else:
                    idx_peak_sel = None
            else:
                idx_peak_sel = idx_peak_closest

            if idx_peak_sel is None:
                final_list_g2d_params.append(np.full(6, np.nan))
            else:
                final_list_g2d_params.append([multi_gauss2d_res[p_str+str(idx_peak_sel)].value for p_str in ['amp', 'x', 'y', 'xs', 'ys', 'th']])

            # plot and save resulting plots
            fig, ax = plt.subplots(2, 2)
            im = ax[0,0].imshow(density_field, interpolation=None, cmap='viridis', origin='lower', vmin=0., alpha=1.)
            ax[0,0].set(xlim=(x_range-x_range[0])/d_xy, ylim=(y_range-y_range[0])/d_xy, ylabel='Original pm space and density')
            ax[0,0].scatter((pm_plane[:,0]-x_range[0])/d_xy, (pm_plane[:,1]-y_range[0])/d_xy, lw=0, s=1, c='black', alpha=0.1)
            ax[0,0].scatter(peak_coord_init[:, 1], peak_coord_init[:, 0], lw=0, s=5, c='C3')
            ax[0,0].scatter(ceter_pm_img_x, ceter_pm_img_y, lw=0, s=10, c='C0')
            fig.colorbar(im, ax=ax[0,0])

            ax[0,1].imshow(density_field_background, interpolation=None, cmap='viridis', origin='lower', alpha=1.)
            ax[0,1].set(xlim=(x_range - x_range[0]) / d_xy, ylim=(y_range - y_range[0]) / d_xy, ylabel='Fitted pm distribution')
            if idx_peak_sel is not None:
                ax[0, 1].scatter(peak_coord_init[idx_peak_sel, 1], peak_coord_init[idx_peak_sel, 0], lw=0, s=6, c='C3')

            im = ax[1,0].imshow(density_field_new, interpolation=None, cmap='viridis', origin='lower', alpha=1.)
            ax[1,0].set(xlim=(x_range - x_range[0]) / d_xy, ylim=(y_range - y_range[0]) / d_xy, ylabel='Residuals after fit, new cluster pm center')
            fig.colorbar(im, ax=ax[1, 0])

            if i_run < n_runs:
                # skip plotting at this point for the last pm incarnation run
                plt.tight_layout()
                plt.savefig('pm_gaussian_fit' + suffix + '_{:02.0f}.png'.format(i_run), dpi=300)
                # plt.show()
                plt.close()

        # combine results from multiple runs and add results tho the final fit plot
        final_g2d_params = np.nanmedian(final_list_g2d_params, axis=0)

        ax[1,1].scatter(pm_plane_orig[:,0], pm_plane_orig[:,1], lw=0, s=1, c='black', alpha=0.1)

        ax[1,1].scatter(final_g2d_params[1], final_g2d_params[2], lw=0, s=10, c='C0')
        ax[1,1].set(xlim=x_range, ylim=y_range, ylabel='Objects inside cluster pm selection, new center')

        # skip plotting at this point for the last pm incarnation run
        plt.tight_layout()
        plt.savefig('pm_gaussian_fit' + suffix + '_{:02.0f}.png'.format(i_run), dpi=300)
        plt.close()

        if np.sum(np.isfinite(final_g2d_params)) != 0:
            # plot histograms of individual parameter in final_g2d_params
            # 'amp', 'x', 'y', 'xs', 'ys', 'th'
            final_list_g2d_params = np.array(final_list_g2d_params)
            fig, ax = plt.subplots(2, 3)
            ax[0, 0].hist(final_list_g2d_params[:, 0], range=(np.nanmin(final_list_g2d_params[:, 0]), np.nanmax(final_list_g2d_params[:, 0])), bins=25)
            ax[0, 0].axvline(final_g2d_params[0], c='black')
            ax[0, 0].set(title='Amp ({:.3f})'.format(np.nanstd(final_list_g2d_params[:, 0])))
            ax[0, 1].hist(final_list_g2d_params[:, 1], range=(np.nanmin(final_list_g2d_params[:, 1]), np.nanmax(final_list_g2d_params[:, 1])), bins=25)
            ax[0, 1].axvline(final_g2d_params[1], c='black')
            ax[0, 1].set(title='x ({:.3f})'.format(np.nanstd(final_list_g2d_params[:, 1])))
            ax[0, 2].hist(final_list_g2d_params[:, 2], range=(np.nanmin(final_list_g2d_params[:, 2]), np.nanmax(final_list_g2d_params[:, 2])), bins=25)
            ax[0, 2].axvline(final_g2d_params[2], c='black')
            ax[0, 2].set(title='y ({:.3f})'.format(np.nanstd(final_list_g2d_params[:, 2])))
            ax[1, 0].hist(final_list_g2d_params[:, 3], range=(np.nanmin(final_list_g2d_params[:, 3]), np.nanmax(final_list_g2d_params[:, 3])), bins=25)
            ax[1, 0].axvline(final_g2d_params[3], c='black')
            ax[1, 0].set(title='xs ({:.3f})'.format(np.nanstd(final_list_g2d_params[:, 3])))
            ax[1, 1].hist(final_list_g2d_params[:, 4], range=(np.nanmin(final_list_g2d_params[:, 4]), np.nanmax(final_list_g2d_params[:, 4])), bins=25)
            ax[1, 1].axvline(final_g2d_params[4], c='black')
            ax[1, 1].set(title='ys ({:.3f})'.format(np.nanstd(final_list_g2d_params[:, 4])))
            ax[1, 2].hist(final_list_g2d_params[:, 5], range=(np.nanmin(final_list_g2d_params[:, 5]), np.nanmax(final_list_g2d_params[:, 5])), bins=25)
            ax[1, 2].axvline(final_g2d_params[5], c='black')
            ax[1, 2].set(title='th ({:.3f})'.format(np.nanstd(final_list_g2d_params[:, 5])))
            plt.tight_layout()
            plt.savefig('pm_gaussian_fit' + suffix + '_hist.png', dpi=250)
            plt.close()

        # store params in the class and return them
        self.cluster_g2d_params = final_g2d_params
        return final_g2d_params

    def perform_selection(self, rad, bayesian_mixture=False, model_from_selection=False, max_com=10,
                          covarinace='full', determine_cluster_center=False):
        """

        :param rad:
        :param bayesian_mixture:
        :param model_from_selection:
        :param max_com:
        :param covarinace:
        :param determine_cluster_center:
        :return:
        """
        idx_sel = self.data['center_sep'] <= rad
        data_cur = self.data[idx_sel]

        n_data_sel = len(data_cur)
        if n_data_sel < 5:
            print ' Not enough data points in selected radius'
            return

        if model_from_selection:
            max_com = 2
            idx_p = self.get_cluster_members()
            X_train = np.vstack((self.data['pmra'][idx_p].data, self.data['pmdec'][idx_p].data)).T
        else:
            max_com = int(min(max_com, round(n_data_sel / 2.)))
            X_train = np.vstack((data_cur['pmra'].data, data_cur['pmdec'].data)).T
        X_test = np.vstack((data_cur['pmra'].data, data_cur['pmdec'].data)).T

        # train step
        if bayesian_mixture:
            clf = mixture.BayesianGaussianMixture(n_components=max_com, covariance_type=covarinace, n_init=10, max_iter=200,
                                                  init_params='random', weight_concentration_prior_type='dirichlet_process')
            clf.fit(X_train)
        else:
            bic_res = list([])
            # determine number of components to be used in the final fit
            if covarinace == 'select':
                bic_res1 = list([])
                bic_res2 = list([])
                for n_c in range(2, max_com):
                    clf1 = mixture.GaussianMixture(n_components=n_c, covariance_type='diag', n_init=10, init_params='random', max_iter=250)
                    clf1.fit(X_train)
                    clf2 = mixture.GaussianMixture(n_components=n_c, covariance_type='full', n_init=10, init_params='random', max_iter=250)
                    clf2.fit(X_train)
                    bic_res1.append(clf1.bic(X_train))
                    bic_res2.append(clf2.bic(X_train))
                # first select the covariance, then number of components
                if np.min(bic_res1) <= np.min(bic_res2):
                    mixtuer_comp_use = np.argmin(bic_res1) + 2
                    covarinace_use = 'diag'
                else:
                    mixtuer_comp_use = np.argmin(bic_res2) + 2
                    covarinace_use = 'full'
                # print ' Covarinace to use:', covarinace_use
            else:
                for n_c in range(2, max_com):
                    clf = mixture.GaussianMixture(n_components=n_c, covariance_type=covarinace,
                                                  n_init=10, init_params='random', max_iter=250)
                    clf.fit(X_train)
                    bic_res.append(clf.bic(X_train))
                mixtuer_comp_use = np.argmin(bic_res) + 2
                covarinace_use = covarinace

            clf = mixture.GaussianMixture(n_components=mixtuer_comp_use, covariance_type=covarinace_use,
                                          n_init=10, init_params='random', max_iter=250)
            clf.fit(X_train)

        # apply the model to test data and produce labels
        gm_means = clf.means_
        gm_labels = clf.predict(X_test)

        # select the label and center of gaussian that represent observed cluster

        # use label of the nearest clump
        clust_pm_cent_new = gm_means[np.argmin(np.sqrt(np.sum((gm_means - self.pm_center) ** 2, axis=1)))]
        if determine_cluster_center:
            return clust_pm_cent_new  # pmra / pmdec of the new center
        clust_label = clf.predict(clust_pm_cent_new.reshape(1, -1))
        # label is the same as predicted for the previous center
        # clust_label = clf.predict(self.pm_center.reshape(1, -1))

        idx_clust = gm_labels == clust_label

        if not model_from_selection:
            idx_clust = np.where(idx_sel)[0][idx_clust]
            self.selected[idx_clust] += 1
            self.n_runs += 1
        else:
            self.selected_final = idx_clust

        print 'plot GM result'
        plt.scatter(X_train[:,0], X_train[:,1], c=gm_labels, s=2, lw=0)
        plt.colorbar()
        plt.scatter(self.pm_center[0], self.pm_center[1], c='red', lw=0, s=5)
        plt.show()
        plt.close()

    def determine_cluster_center(self):
        print ' Proper motion cluster center'
        print '   old center:', self.pm_center
        new_centers = list([])
        for c_rad in np.linspace(0, np.float64(self.ref['r1']), 8)[1:]:
            c_center = self.perform_selection(c_rad, bayesian_mixture=False, covarinace='diag',
                                                 determine_cluster_center=True, max_com=8)
            new_centers.append(c_center)
        new_center = np.median(np.array(new_centers), axis=0)
        print '   new center:', new_center
        self.pm_center = new_center

    def _selected_deriv(self):
        c, s = np.histogram(self.selected, range=(1, self.n_runs + 1), bins=self.n_runs)
        s = s[:-1]
        idx = s > 0
        # resample the signal to much finer grid
        func = interp1d(s[idx], c[idx], kind='linear')
        x = np.linspace(np.min(s[idx]), np.max(s[idx])-3, 200)
        return x, -10.*np.gradient(func(x))

    def _member_prob_cut(self):
        x, y = self._selected_deriv()
        # find the fist major jump, going from the right side of the plot
        for i in range(0, len(y))[::-1]:
            if y[i] > 5.:
                return x[i]
        return 1.

    def get_cluster_members(self, n_min=25., idx_only=True, recompute=False):
        if recompute is True or self.selected_final is None:
            n_min_grad = 100. * self._member_prob_cut() / self.n_runs
            n_min_final = max(n_min, n_min_grad)
            idx_memebers = 1. * self.selected / self.n_runs >= n_min_final / 100.
            if idx_only:
                self.selected_final = deepcopy(idx_memebers)
                return idx_memebers
        else:
            if idx_only:
                return self.selected_final
        if not idx_only:
            return np.int64(self.data[self.selected_final]['source_id'].data)

    def plot_members(self, n_min=30, show_n_sel=False, show_final=False, path='plot.png'):
        if show_n_sel:
            plt.scatter(self.data['pmra'], self.data['pmdec'], lw=0, s=6, c=100.*self.selected/self.n_runs,
                        cmap='viridis', vmin=0, vmax=100)
            plt.colorbar()
        else:
            if show_final:
                if self.selected_final is None:
                    # final selection was not yet performed
                    print 'Running final selection from selected objects'
                    self.perform_selection(np.max(self.data['center_sep']), bayesian_mixture=True, max_com=5,
                                           model_from_selection=True)
                idx_p = self.selected_final
            else:
                idx_p = self.get_cluster_members(n_min=n_min)
            plt.scatter(self.data['pmra'][~idx_p], self.data['pmdec'][~idx_p], lw=0, s=3, c='black', alpha=0.2)
            plt.scatter(self.data['pmra'][idx_p], self.data['pmdec'][idx_p], lw=0, s=3, c='blue')
        plt.scatter(self.pm_center[0], self.pm_center[1], lw=0, s=8, marker='*', c='red')
        plt.xlim(self.pl_xlim)
        plt.ylim(self.pl_ylim)
        plt.savefig(path, dpi=250)
        plt.close()

    def plot_selected_hist(self, path='plot.png'):
        c, s = np.histogram(self.selected, range=(1, self.n_runs+1), bins=self.n_runs)
        s = s[:-1]
        idx = s > 0
        x_deriv, y_deriv = self._selected_deriv()
        plt.plot(s[idx], c[idx], c='black')
        plt.plot(x_deriv, y_deriv, c='red')
        plt.axvline(x=self._member_prob_cut(), c='blue', lw=1)
        plt.grid(alpha=0.5, color='black', ls='--')
        plt.tight_layout()
        plt.savefig(path, dpi=250)
        plt.close()

    def refine_distances_selection(self, out_plot=False, path='plot.png', n_iter=1):
        idx_c = self.get_cluster_members()
        data_c = self.data[idx_c]

        dist_med = list([])
        dist_std = list([])
        members_par = data_c['parallax'].data
        members_par_e = data_c['parallax_error'].data
        for i_plx in np.arange(n_iter)+1:
            print ' Creating new incarnation of stellar distances'
            plx_new_list = list([])
            for i_mp in range(len(members_par)):
                plx_new_list.append(np.random.normal(members_par[i_mp], members_par_e[i_mp]))
            dist_list = 1e3/np.array(plx_new_list)  # pc
            dist_list[np.logical_or(dist_list < np.percentile(dist_list,2), dist_list > np.percentile(dist_list,98))] = np.nan
            dist_med.append(np.nanmedian(dist_list))
            dist_std.append(np.nanstd(dist_list))
        cluster_pc_medi = np.nanmedian(dist_med)
        cluster_pc_std = np.nanmedian(dist_std)
        print 'Medians:', dist_med
        print 'Stds:   ', dist_std
        # save results
        self.cluster_dist_params = [cluster_pc_medi, cluster_pc_std]
        self.dist_probability = (1./(cluster_pc_std*np.sqrt(2.*np.pi))) * np.exp(-0.5*(self.data['parsec']-cluster_pc_medi)**2/cluster_pc_std**2)

        if out_plot:
            plt.hist(self.data['parsec'], range=self.parsec_lim, bins=100, color='black', alpha=0.3)
            plt.hist(data_c['parsec'], range=self.parsec_lim, bins=100, color='red', alpha=0.3)
            plt.axvline(x=cluster_pc_medi, color='red', ls='--')
            plt.axvline(x=cluster_pc_medi + cluster_pc_std, color='red', ls='--', alpha=0.4)
            plt.axvline(x=cluster_pc_medi - cluster_pc_std, color='red', ls='--', alpha=0.4)
            plt.axvline(x=cluster_pc_medi + 2.0*cluster_pc_std, color='red', ls='--', alpha=0.2)
            plt.axvline(x=cluster_pc_medi - 2.0*cluster_pc_std, color='red', ls='--', alpha=0.2)
            plt.axvline(x=self.ref['d'], color='black', ls='--')
            plt.tight_layout()
            plt.savefig(path, dpi=250)
            plt.close()

        # TODO: exclude those checks for now
        # if np.abs(self.ref['d'] - cluster_pc_medi) > 150:
        #     print ' Distance error is too large for this object'
        #     return False



        return True

    def refine_distances_selection_RV(self, out_plot=False, path='plot.png'):
        idx_c = self.get_cluster_members()
        data_c = self.data[idx_c]

        # determine number of valid observations of radial velocity
        n_val_rv = np.sum(np.isfinite(data_c['rv']))
        if n_val_rv >= 2:
            rv_eval = True
        else:
            rv_eval = False

        if rv_eval:
            cluster_rv_medi = np.nanmedian(data_c['rv'])
            if len(data_c) >= 3:
                cluster_rv_std = np.nanstd(data_c['rv'])
            else:
                cluster_rv_std = 10.

        if out_plot:
            plt.hist(self.data['rv'], range=(-100,100), bins=100, color='black', alpha=0.3)
            plt.hist(data_c['rv'], range=(-100,100), bins=100, color='red', alpha=0.3)
            if rv_eval:
                plt.axvline(x=cluster_rv_medi, color='red', ls='--')
                # plt.axvline(x=cluster_rv_medi + 0.5*cluster_rv_std, color='red', ls='--', alpha=0.6)
                # plt.axvline(x=cluster_rv_medi - 0.5*cluster_rv_std, color='red', ls='--', alpha=0.6)
                plt.axvline(x=cluster_rv_medi + cluster_rv_std, color='red', ls='--', alpha=0.4)
                plt.axvline(x=cluster_rv_medi - cluster_rv_std, color='red', ls='--', alpha=0.4)
                plt.axvline(x=cluster_rv_medi + 2.0*cluster_rv_std, color='red', ls='--', alpha=0.2)
                plt.axvline(x=cluster_rv_medi - 2.0*cluster_rv_std, color='red', ls='--', alpha=0.2)
            plt.tight_layout()
            plt.savefig(path, dpi=250)
            plt.close()

        if not rv_eval:
            print ' Not enough RV values in gaia data to evaluate it'
            return False

        # else do the selection based on distance distribution
        std_multi_thr = 2.
        idx_bad_dist = np.abs(data_c['rv'] - cluster_rv_medi) > std_multi_thr * cluster_rv_std
        n_nbad = np.sum(idx_bad_dist)

        if n_nbad > 0:
            mark_bad = np.where(idx_c)[0][idx_bad_dist]
            n_bef = np.sum(self.selected_final)
            self.selected_final[mark_bad] = False
            n_aft = np.sum(self.selected_final)
            print ' Removed by outlying RV velocity:', n_bef-n_aft

        return True

    def include_iniside_hull(self, distance_limits=True, manual_hull=False):
        idx_c = self.get_cluster_members()
        data_c = self.data[idx_c]

        if len(data_c) < 3:
            print ' Not enough points to construct a hull'
            return False

        min_dist = np.min(data_c['parsec'])
        max_dist = np.max(data_c['parsec'])

        pm_data_a = self.data['pmra', 'pmdec'].to_pandas().values
        pm_data_c = data_c['pmra', 'pmdec'].to_pandas().values
        if manual_hull:
            class PointSelector:
                def __init__(self, axis, x_data, y_data):
                    self.x = x_data
                    self.y = y_data
                    self.lasso = LassoSelector(axis, self.determine_points)

                def determine_points(self, vertices):
                    if len(vertices) > 0:
                        vert_path = Path(vertices)
                        # determine objects in region
                        self.idx_sel = list([])
                        for i_p in range(len(self.x)):
                            if vert_path.contains_point((self.x[i_p], self.y[i_p])):
                                self.idx_sel.append(True)
                            else:
                                self.idx_sel.append(False)

                def get_selected(self):
                    return np.array(self.idx_sel)

            fig, ax = plt.subplots(1, 1)
            ax.scatter(pm_data_a[:,0], pm_data_a[:,1], lw=0, s=5, c='black')
            ax.scatter(pm_data_c[:,0], pm_data_c[:,1], lw=0, s=5, c='red')
            selector = PointSelector(ax, pm_data_a[:,0], pm_data_a[:,1])
            plt.show()
            plt.close()
            idx_inside_hull = selector.get_selected()
        else:
            idx_hull_vert = ConvexHull(pm_data_c).vertices
            # create a Delaunay grid based on given points
            delanuay_surf = Delaunay(pm_data_c[idx_hull_vert])
            idx_inside_hull = delanuay_surf.find_simplex(pm_data_a) >= 0

        # apply distance limit if requested
        if distance_limits:
            idx_dist_ok = np.logical_and(self.data['parsec'] >= min_dist, self.data['parsec'] <= max_dist)
            idx_inside_hull = np.logical_and(idx_inside_hull, idx_dist_ok)

        n_bf = len(data_c)
        n_af = np.sum(idx_inside_hull)
        # update results
        print ' New objects inside hull:', n_af-n_bf
        self.selected_final = deepcopy(idx_inside_hull)
        return True

    def plot_on_sky(self, path='plot.png', mark_objects=False):
        plt.scatter(self.data['ra'], self.data['dec'], lw=0, s=2, c='black', alpha=1.)
        if mark_objects:
            idx_c = self.get_cluster_members()
            plt.scatter(self.data['ra'][idx_c], self.data['dec'][idx_c], lw=0, s=3, c='red', alpha=1.)
        plt.scatter(self.ref['RAdeg'], self.ref['DEdeg'], lw=0, s=10, marker='*', c='green')
        plt.savefig(path, dpi=250)
        plt.close()

    # def _get_pro_complete(self, include_dist=False, include_radec=False):
    #     # pmra_prob = get_gauss_prob(self.data['pmra'], self.cluster_g2d_params[1], self.cluster_g2d_params[3])
    #     # pmdec_prob = get_gauss_prob(self.data['pmdec'], self.cluster_g2d_params[2], self.cluster_g2d_params[4])
    #     # joint_probability = pmdec_prob * pmra_prob
    #     #
    #     # if include_dist:
    #     #     joint_probability *= get_gauss_prob(self.data['parsec'], self.cluster_dist_params[0],
    #     #                                         self.cluster_dist_params[1])
    #     #
    #     # if include_radec:
    #     #     joint_probability *= get_gauss_prob(self.data['ra'], self.cluster_pos_params[0], self.cluster_pos_params[2])
    #     #     joint_probability *= get_gauss_prob(self.data['dec'], self.cluster_pos_params[1],
    #     #                                         self.cluster_pos_params[3])
    #     #
    #     # return joint_probability
    #
    #     # VERSION 2
    #     def gauss2d(v1, v2, m1, m2, s1, s2):
    #         return 1. * np.exp(-1 * ((v1 - m1) / (2. * s1)) ** 2 - 1 * ((v2 - m2) / (2. * s2)) ** 2)
    #
    #     def gauss1d(v1, m1, s1):
    #         return 1. * np.exp(-1 * ((v1 - m1) / (2. * s1)) ** 2)
    #
    #     g_val = gauss2d(self.data['pmra'], self.data['pmdec'],
    #                     self.cluster_g2d_params[1], self.cluster_g2d_params[2],
    #                     self.cluster_g2d_params[3], self.cluster_g2d_params[4])
    #     if include_dist:
    #         g_val *= gauss1d(self.data['parsec'], self.cluster_dist_params[0], self.cluster_dist_params[1])
    #
    #     if include_radec:
    #         g_val *= gauss2d(self.data['ra'], self.data['dec'],
    #                          self.cluster_pos_params[0], self.cluster_pos_params[1],
    #                          self.cluster_pos_params[2], self.cluster_pos_params[3])
    #
    #     return g_val
    #
    # def _get_pro_complete_selection(self, include_dist=False, include_radec=False, prob_thr=25):
    #     def gauss2d(v1, v2, m1, m2, s1, s2):
    #         return 1. * np.exp(-1 * ((v1 - m1) / (2. * s1)) ** 2 - 1 * ((v2 - m2) / (2. * s2)) ** 2)
    #
    #     def gauss1d(v1, m1, s1):
    #         return 1. * np.exp(-1 * ((v1 - m1) / (2. * s1)) ** 2)
    #
    #     g_val = gauss2d(self.data['pmra'], self.data['pmdec'],
    #                     self.cluster_g2d_params[1], self.cluster_g2d_params[2],
    #                     self.cluster_g2d_params[3], self.cluster_g2d_params[4])
    #     idx_sel = g_val >= prob_thr/100.
    #
    #     if include_dist:
    #         g_val = gauss1d(self.data['parsec'], self.cluster_dist_params[0], self.cluster_dist_params[1])
    #         idx_sel = np.logical_and(idx_sel, g_val >= prob_thr/100.)
    #
    #     if include_radec:
    #         g_val = gauss2d(self.data['ra'], self.data['dec'],
    #                         self.cluster_pos_params[0], self.cluster_pos_params[1],
    #                         self.cluster_pos_params[2], self.cluster_pos_params[3])
    #         idx_sel = np.logical_and(idx_sel, g_val >= prob_thr/100.)
    #
    #     self.selected_final = deepcopy(idx_sel)
    #     return idx_sel

    def _get_pro_complete(self, prob_thr=None, prob_sigma=None, return_sigma_vals=False):
        # pmra, pmdec, parsec, ra, dec
        dist_mean_vals = np.array([self.cluster_g2d_params[1], self.cluster_g2d_params[2],
                                   self.cluster_dist_params[0],
                                   self.cluster_pos_params[0], self.cluster_pos_params[1]])
        dist_std_vals = np.array([self.cluster_g2d_params[3], self.cluster_g2d_params[4],
                                  self.cluster_dist_params[1],
                                  self.cluster_pos_params[2], self.cluster_pos_params[3]])
        data_vals = self.data['pmra', 'pmdec', 'parsec', 'ra', 'dec'].to_pandas().values
        n_params = len(dist_std_vals)

        # cov_matrix = np.identity(len(dist_std_vals)) * dist_std_vals**2
        cov_matrix = np.cov(data_vals[self.selected_final].T)

        multi_prob = multivariate_normal.pdf(data_vals, mean=dist_mean_vals, cov=cov_matrix)
        # multi_prob = multivariate_normal.cdf(-1.*np.abs(data_vals - dist_mean_vals), mean=None, cov=cov_matrix)

        max_sigma = 2.
        eval_param_values = dist_mean_vals + dist_std_vals*np.repeat([np.arange(0, max_sigma+1, 0.5)], n_params, axis=0).T
        multi_prob_sigmas = multivariate_normal.pdf(eval_param_values, mean=dist_mean_vals, cov=cov_matrix)
        # eval_param_values = dist_std_vals * np.repeat([np.arange(0, max_sigma + 1, 0.5)], n_params, axis=0).T
        # multi_prob_sigmas = multivariate_normal.cdf(-1.*np.abs(eval_param_values), mean=None, cov=cov_matrix)
        # print 'Sigma prob', multi_prob_sigmas

        if prob_thr is None and prob_sigma is None:
            if return_sigma_vals:
                return multi_prob, multi_prob_sigmas
            else:
                return multi_prob
        elif prob_sigma > 0:
            prob_thr_sigma = multivariate_normal.pdf(dist_mean_vals + prob_sigma*dist_std_vals, mean=dist_mean_vals, cov=cov_matrix)
            # prob_thr_sigma = multivariate_normal.cdf(-1.*np.abs(prob_sigma*dist_std_vals), mean=None, cov=cov_matrix)

            idx_sel = multi_prob >= prob_thr_sigma
            self.selected_final = deepcopy(idx_sel)
        else:
            idx_sel = multi_prob >= prob_thr / 100.
            self.selected_final = deepcopy(idx_sel)
            return idx_sel

    def plot_selection_density_pde(self, path='plot.png', plot=True, include_dist=False, include_radec=False):
        # pm_probability = single_gaussian2D(self.data['pmra'], self.data['pmdec'], self.cluster_g2d_params, pde=True)

        joint_probability = self._get_pro_complete(prob_thr=None)

        if plot:
            max_prob = np.max(joint_probability)
            plt.scatter(self.data['ra'], self.data['dec'], lw=0, s=2, c=joint_probability, alpha=1., vmin=0., vmax=max_prob)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(path, dpi=250)
            plt.close()
            # pde histogram
            plt.hist(joint_probability, range=[0, max_prob], bins=25)
            plt.xlim([0, 1])
            plt.gca().set_yscale('log', nonposy='clip')
            plt.tight_layout()
            plt.savefig(path[:-4]+'_hist.png', dpi=250)
            plt.close()
        return joint_probability

    def get_cluster_members_pmprob(self, min_prob=10.):

        joint_probability = self._get_pro_complete(prob_thr=None)
        idx_members = joint_probability >= min_prob/100.
        self.selected_final = deepcopy(idx_members)
        return idx_members

    def plot_cluster_members_pmprob(self, path='plot.png', min_prob=None, max_sigma=1.):

        g_val = single_gaussian2D(self.data['pmra'], self.data['pmdec'], self.cluster_g2d_params, pde=False)
        g_val_sigma = single_gaussian2D(self.cluster_g2d_params[1] + max_sigma*self.cluster_g2d_params[3],
                                        self.cluster_g2d_params[2] + max_sigma*self.cluster_g2d_params[4],
                                        self.cluster_g2d_params, pde=False)

        self.selected_final = g_val >= g_val_sigma
        idx_p = self.selected_final
        plt.scatter(self.data['pmra'][~idx_p], self.data['pmdec'][~idx_p], lw=0, s=3, c='black', alpha=0.2)
        plt.scatter(self.data['pmra'][idx_p], self.data['pmdec'][idx_p], lw=0, s=3, c='red', alpha=0.2)
        # plt.scatter(self.data['pmra'], self.data['pmdec'], lw=0, s=2, c='black', alpha=0.2)
        # for s in [5., 4., 3., 2., 1.]:
        #     g_val_sigma = single_gaussian2D(self.cluster_g2d_params[1] + s*self.cluster_g2d_params[3],
        #                                     self.cluster_g2d_params[2] + s*self.cluster_g2d_params[4],
        #                                     self.cluster_g2d_params, pde=False)
        #     idx_p = g_val >= g_val_sigma
        #     plt.scatter(self.data['pmra'][idx_p], self.data['pmdec'][idx_p], lw=0, s=2)
        plt.scatter(self.pm_center[0], self.pm_center[1], lw=0, s=8, marker='*', c='blue')
        plt.xlim(self.pl_xlim)
        plt.ylim(self.pl_ylim)
        plt.savefig(path, dpi=250)
        plt.close()

    def initial_distance_cut(self, path='plot.png', max_sigma=2.):
        idx_p = self.selected_final
        h_y, h_e = np.histogram(self.data['parsec'][idx_p], bins=50, range=self.parsec_lim)
        h_x = h_e[:-1] + (h_e[1]-h_e[0])/2.
        # fit model to that
        m_p = np.nanmedian(self.data['parsec'][idx_p])

        fit_model_1 = models.LinearModel()
        pars = fit_model_1.guess(h_y, x=h_x)
        fit_model_2 = models.GaussianModel()
        pars.update(fit_model_2.make_params())
        fit_model = fit_model_1 + fit_model_2
        pars['center'].set(m_p, min=m_p-100., max=m_p+100.)
        pars['sigma'].set(75., min=10., max=120.)
        pars['amplitude'].set(np.nanmax(h_y), min=5.)
        pars['slope'].set(0.)
        pars['intercept'].set(2.)

        fit_out = fit_model.fit(h_y, pars, x=h_x)
        parsec_std = fit_out.params['sigma'].value
        parsec_mean = fit_out.params['center'].value

        n_init = np.sum(self.selected_final)
        idx_parsec = np.logical_and(self.data['parsec'] >= parsec_mean - max_sigma*parsec_std,
                                    self.data['parsec'] <= parsec_mean + max_sigma*parsec_std)
        self.selected_final = np.logical_and(self.selected_final, idx_parsec)
        print '  Removed by distance:', n_init-np.sum(self.selected_final)

        plt.plot(h_x, h_y, c='black')
        # plt.plot(h_x, fit_init, 'b--')
        plt.plot(h_x, fit_out.best_fit, 'g--')
        plt.axvline(parsec_mean + max_sigma*parsec_std, color='b', ls='--', alpha=1., lw=0.5)
        plt.axvline(parsec_mean - max_sigma*parsec_std, color='b', ls='--', alpha=1., lw=0.5)
        for s in range(1, 6):
            plt.axvline(parsec_mean + s*parsec_std, color='red', ls='--', alpha=1.-0.15*s)
            plt.axvline(parsec_mean - s*parsec_std, color='red', ls='--', alpha=1.-0.15*s)
        plt.title('Mean: {:.0f}   Sigma: {:.2f}'.format(parsec_mean, parsec_std))
        plt.tight_layout()
        plt.savefig(path, dpi=250)
        plt.close()

        return parsec_mean, parsec_std

    def define_ra_dec_distribution(self):
        idx_p = self.selected_final
        self.cluster_pos_params = [np.nanmedian(self.data['ra'][idx_p]), np.nanmedian(self.data['dec'][idx_p]),
                                   np.nanstd(self.data['ra'][idx_p]), np.nanstd(self.data['dec'][idx_p])]

    def update_selection_parameters(self, sel_limit=None, prob_sigma=None):
        idx_p = self.selected_final

        self.cluster_g2d_params = [1., np.nanmedian(self.data['pmra'][idx_p]), np.nanmedian(self.data['pmdec'][idx_p]),
                                   np.nanstd(self.data['pmra'][idx_p]), np.nanstd(self.data['pmdec'][idx_p]), 0.]
        self.cluster_dist_params = [np.nanmedian(self.data['parsec'][idx_p]), np.nanstd(self.data['parsec'][idx_p])]
        self.cluster_pos_params = [np.nanmedian(self.data['ra'][idx_p]), np.nanmedian(self.data['dec'][idx_p]),
                                   np.nanstd(self.data['ra'][idx_p]), np.nanstd(self.data['dec'][idx_p])]

        if sel_limit is not None or prob_sigma is not None:
            # self._get_pro_complete_selection(prob_thr=sel_limit, include_dist=True, include_radec=True)
            self._get_pro_complete(prob_thr=sel_limit, prob_sigma=prob_sigma)

    def plot_selected_complete(self, path='plot.png'):
        fig, ax = plt.subplots(2, 3, figsize=(13, 9))
        idx_p = self.selected_final

        fig.suptitle('Number stars: '+str(np.sum(idx_p)))

        ax[0, 0].scatter(self.data['ra'][~idx_p], self.data['dec'][~idx_p], lw=0, s=3, c='black', alpha=1)
        ax[0, 0].scatter(self.data['ra'][idx_p], self.data['dec'][idx_p], lw=0, s=3, c='red', alpha=1)

        ax[0, 1].scatter(self.data['pmra'][~idx_p], self.data['pmdec'][~idx_p], lw=0, s=3, c='black', alpha=0.3)
        ax[0, 1].scatter(self.data['pmra'][idx_p], self.data['pmdec'][idx_p], lw=0, s=3, c='red', alpha=0.3)
        ax[0, 1].set(xlim=self.pl_xlim, ylim=self.pl_ylim)

        ax[1, 0].hist(self.data['parsec'][~idx_p], range=self.parsec_lim, bins=100, color='black', alpha=0.3)
        ax[1, 0].hist(self.data['parsec'][idx_p], range=self.parsec_lim, bins=100, color='red', alpha=0.3)
        ax[1, 0].set(xlim=self.parsec_lim)

        joint_probability, sigma_probabilities = self._get_pro_complete(prob_thr=None, return_sigma_vals=True)
        min_max_prob = [np.nanmin(joint_probability), np.nanmax(joint_probability)]
        ax[0, 2].scatter(self.data['ra'], self.data['dec'], lw=0, s=3, c=joint_probability,
                         vmin=min_max_prob[0], vmax=min_max_prob[1])

        ax[1, 2].hist(joint_probability, range=min_max_prob, bins=40)
        for s_p in sigma_probabilities:
            ax[1, 2].axvline(s_p, color='black', ls='--', alpha=0.75)
        # ax[1, 2].set(xlim=(0., 1.))
        ax[1, 2].set(xlim=min_max_prob)
        ax[1, 2].set_yscale('log', nonposy='clip')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def plot_selected_complete_dist(self, path='plot.png'):
        fig, ax = plt.subplots(2, 3, figsize=(13, 9))
        idx_p = self.selected_final

        fig.suptitle('Number stars: '+str(np.sum(idx_p)))

        # ax[0, 0].hist(self.data['ra'][~idx_p], bins=100, color='black', alpha=0.3)
        ax[0, 0].hist(self.data['ra'][idx_p], bins=100, color='red', alpha=0.3)
        ax[0, 0].axvline(self.cluster_pos_params[0], color='red', alpha=0.3)
        ax[0, 0].axvline(self.cluster_pos_params[0] + self.cluster_pos_params[2], color='red', ls='--', alpha=0.3)
        ax[0, 0].axvline(self.cluster_pos_params[0] - self.cluster_pos_params[2], color='red', ls='--', alpha=0.3)
        ax[0, 0].axvline(self.cluster_pos_params[0] + 2.*self.cluster_pos_params[2], color='red', ls='--', alpha=0.3)
        ax[0, 0].axvline(self.cluster_pos_params[0] - 2.*self.cluster_pos_params[2], color='red', ls='--', alpha=0.3)

        # ax[0, 1].hist(self.data['dec'][~idx_p], bins=100, color='black', alpha=0.3)
        ax[0, 1].hist(self.data['dec'][idx_p], bins=100, color='red', alpha=0.3)
        ax[0, 1].axvline(self.cluster_pos_params[1], color='red', alpha=0.3)
        ax[0, 1].axvline(self.cluster_pos_params[1] + self.cluster_pos_params[3], color='red', ls='--', alpha=0.3)
        ax[0, 1].axvline(self.cluster_pos_params[1] - self.cluster_pos_params[3], color='red', ls='--', alpha=0.3)
        ax[0, 1].axvline(self.cluster_pos_params[1] + 2.*self.cluster_pos_params[3], color='red', ls='--', alpha=0.3)
        ax[0, 1].axvline(self.cluster_pos_params[1] - 2.*self.cluster_pos_params[3], color='red', ls='--', alpha=0.3)

        # ax[0, 2].hist(self.data['parsec'][~idx_p], bins=100, color='black', alpha=0.3, range=self.parsec_lim)
        ax[0, 2].hist(self.data['parsec'][idx_p], bins=100, color='red', alpha=0.3, range=self.parsec_lim)
        ax[0, 2].axvline(self.cluster_dist_params[0], color='red', alpha=0.3)
        ax[0, 2].axvline(self.cluster_dist_params[0] + self.cluster_dist_params[1], color='red', ls='--', alpha=0.3)
        ax[0, 2].axvline(self.cluster_dist_params[0] - self.cluster_dist_params[1], color='red', ls='--', alpha=0.3)
        ax[0, 2].axvline(self.cluster_dist_params[0] + 2.*self.cluster_dist_params[1], color='red', ls='--', alpha=0.3)
        ax[0, 2].axvline(self.cluster_dist_params[0] - 2.*self.cluster_dist_params[1], color='red', ls='--', alpha=0.3)

        # ax[1, 0].hist(self.data['pmra'][~idx_p], bins=100, color='black', alpha=0.3, range=self.pl_xlim)
        ax[1, 0].hist(self.data['pmra'][idx_p], bins=100, color='red', alpha=0.3, range=self.pl_xlim)
        ax[1, 0].axvline(self.cluster_g2d_params[1], color='red', alpha=0.3)
        ax[1, 0].axvline(self.cluster_g2d_params[1] + self.cluster_g2d_params[3], color='red', ls='--', alpha=0.3)
        ax[1, 0].axvline(self.cluster_g2d_params[1] - self.cluster_g2d_params[3], color='red', ls='--', alpha=0.3)
        ax[1, 0].axvline(self.cluster_g2d_params[1] + 2.*self.cluster_g2d_params[3], color='red', ls='--', alpha=0.3)
        ax[1, 0].axvline(self.cluster_g2d_params[1] - 2.*self.cluster_g2d_params[3], color='red', ls='--', alpha=0.3)

        # ax[1, 1].hist(self.data['pmdec'][~idx_p], bins=100, color='black', alpha=0.3, range=self.pl_ylim)
        ax[1, 1].hist(self.data['pmdec'][idx_p], bins=100, color='red', alpha=0.3, range=self.pl_ylim)
        ax[1, 1].axvline(self.cluster_g2d_params[2], color='red', alpha=0.3)
        ax[1, 1].axvline(self.cluster_g2d_params[2] + self.cluster_g2d_params[4], color='red', ls='--', alpha=0.3)
        ax[1, 1].axvline(self.cluster_g2d_params[2] - self.cluster_g2d_params[4], color='red', ls='--', alpha=0.3)
        ax[1, 1].axvline(self.cluster_g2d_params[2] + 2.*self.cluster_g2d_params[4], color='red', ls='--', alpha=0.3)
        ax[1, 1].axvline(self.cluster_g2d_params[2] - 2.*self.cluster_g2d_params[4], color='red', ls='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()