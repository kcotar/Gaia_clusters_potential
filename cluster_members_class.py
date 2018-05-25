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


# -------------------------------
# other useful functions
# -------------------------------


# -------------------------------
# class functions
# -------------------------------


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
        self.selected_final = None
        self.selected = np.zeros(len(self.data))
        self.n_runs = 0

    def perform_selection(self, rad, bayesian_mixture=False, model_from_selection=False, max_com=12, covarinace='full'):
        idx_sel = self.data['center_sep'] <= rad
        data_cur = self.data[idx_sel]

        n_data_sel = len(data_cur)
        if n_data_sel < 5:
            print ' Not enough data points in selected radious'
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
        plt.scatter(self.ref['pmRAc'], self.ref['pmDEc'], lw=0, s=8, marker='*', c='red')
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

    def refine_distances_selection(self, out_plot=False, path='plot.png'):
        idx_c = self.get_cluster_members()
        data_c = self.data[idx_c]

        cluster_pc_medi = np.nanmedian(data_c['parsec'])
        if len(data_c) >= 3:
            cluster_pc_std = np.nanstd(data_c['parsec'])
        else:
            cluster_pc_std = 75.

        if out_plot:
            plt.hist(self.data['parsec'], range=self.parsec_lim, bins=100, color='black', alpha=0.3)
            plt.hist(data_c['parsec'], range=self.parsec_lim, bins=100, color='red', alpha=0.3)
            plt.axvline(x=cluster_pc_medi, color='red', ls='--')
            # plt.axvline(x=cluster_pc_medi + 0.5*cluster_pc_std, color='red', ls='--', alpha=0.6)
            # plt.axvline(x=cluster_pc_medi - 0.5*cluster_pc_std, color='red', ls='--', alpha=0.6)
            plt.axvline(x=cluster_pc_medi + cluster_pc_std, color='red', ls='--', alpha=0.4)
            plt.axvline(x=cluster_pc_medi - cluster_pc_std, color='red', ls='--', alpha=0.4)
            plt.axvline(x=cluster_pc_medi + 2.0*cluster_pc_std, color='red', ls='--', alpha=0.2)
            plt.axvline(x=cluster_pc_medi - 2.0*cluster_pc_std, color='red', ls='--', alpha=0.2)
            plt.axvline(x=self.ref['d'], color='black', ls='--')
            plt.tight_layout()
            plt.savefig(path, dpi=250)
            plt.close()

        if np.abs(self.ref['d'] - cluster_pc_medi) > 150:
            print ' Distance error is too large for this object'
            return False

        # else do the selection based on distance distribution
        std_multi_thr = 1.
        idx_bad_dist = np.abs(data_c['parsec'] - cluster_pc_medi) > std_multi_thr * cluster_pc_std
        n_nbad = np.sum(idx_bad_dist)

        print ' Inside dist:', np.std(data_c['parsec'][~idx_bad_dist])
        if np.std(data_c['parsec'][~idx_bad_dist]) > 75:
            print ' Distribution of distances inside cluster is large'
            return False

        if n_nbad > 0:
            mark_bad = np.where(idx_c)[0][idx_bad_dist]
            n_bef = np.sum(self.selected_final)
            self.selected_final[mark_bad] = False
            n_aft = np.sum(self.selected_final)
            print ' Removed by distance:', n_bef-n_aft

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

