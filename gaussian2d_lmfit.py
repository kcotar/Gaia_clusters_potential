import numpy as np
from time import time
from lmfit import minimize, Parameters, models


def single_gaussian2D(xv, yv, param_values, pde=False):
    # unpack parameters
    amp, xm, ym, xs, ys, th = param_values
    if pde:
        amp = 1.
    # first compute definitions that will be used in exponential function
    aa = (np.cos(th) ** 2) / (2. * xs ** 2) + (np.sin(th) ** 2) / (2. * ys ** 2)
    bb = (np.sin(2. * th)) / (2. * xs ** 2) - (np.sin(2. * th)) / (2. * ys ** 2)
    cc = (np.sin(th) ** 2) / (2. * xs ** 2) + (np.cos(th) ** 2) / (2. * ys ** 2)
    # the following equation is the same as in the case of the Astropy model fitting (Wikipedia source)
    ee = -1. * aa * (xv - xm) ** 2 - bb * (xv - xm) * (yv - ym) - cc * (yv - ym) ** 2
    # return (amp/(xs*ys*np.sqrt(2.*np.pi))) * np.exp(ee)
    return amp * np.exp(ee)


def eval_gaussian2D(params, xv, yv, zv, evaluate=True):
    # compute z values based on given parameters
    n_g = int((len(params)-1)/6)
    zn = np.full_like(xv, 0.)
    # add constant image offset level
    zn += params['offset']
    for i_g in range(n_g):
        su = str(i_g)
        zn += single_gaussian2D(xv, yv, [params['amp'+su], params['x'+su], params['y'+su], params['xs'+su], params['ys'+su], params['th'+su]])
        # aa = (np.cos(params['th'+su])**2) / (2.*params['xs'+su]**2) + (np.sin(params['th'+su])**2) / (2.*params['ys'+su]**2)
        # bb = (np.sin(2.*params['th'+su])) / (2.*params['xs'+su]**2) - (np.sin(2.*params['th'+su])) / (2.*params['ys'+su]**2)
        # cc = (np.sin(params['th'+su])**2) / (2.*params['xs'+su]**2) + (np.cos(params['th'+su])**2) / (2.*params['ys'+su]**2)
        # ee = -1.*aa*(xv-params['x'+su])**2 - bb*(xv-params['x'+su])*(yv-params['y'+su]) - cc*(yv-params['y'+su])**2
        # zn += params['amp'+su] * np.exp(ee)
    if evaluate:
        likelihood = np.power(zn - zv, 2)
        return likelihood
    else:
        return zn


def fit_multi_gaussian2d(z_vals, x_vals, y_vals, peaks_xy, peaks_val,
                         vary_position=True):
    # input values of image(z_values), x_vals and y_vals
    n_peaks = peaks_xy.shape[0]
    print '  Performing multi ('+str(n_peaks)+') 2D gaussian fit'
    fit_param = Parameters()
    fit_param.add('offset', value=0., min=-2., max=2., vary=True)
    xy_vary = 0.2
    for i_p in range(n_peaks):
        su = str(i_p)
        x_p = peaks_xy[i_p][0]
        y_p = peaks_xy[i_p][1]
        fit_param.add('amp'+su, value=peaks_val[i_p], min=0., vary=True)
        fit_param.add('x'+su, value=x_p, min=x_p-xy_vary, max=x_p+xy_vary, vary=vary_position)
        fit_param.add('y'+su, value=y_p, min=y_p-xy_vary, max=y_p+xy_vary, vary=vary_position)
        fit_param.add('xs'+su, value=1., min=0.1, max=15., vary=True)
        fit_param.add('ys'+su, value=1., min=0.1, max=15., vary=True)
        fit_param.add('th'+su, value=0., min=-np.pi/2., max=np.pi/2., vary=True)

    # perform the actual fit itself
    ts = time()
    fit_res = minimize(eval_gaussian2D, fit_param, args=(x_vals, y_vals, z_vals), method='leastsq')
    print '   - fit time {:.1f} min'.format((time()-ts)/60.)
    # fit_res.params.pretty_print()
    # report_fit(fit_res)
    # compute z values using current fitted parameters
    fitted_curve = eval_gaussian2D(fit_res.params, x_vals, y_vals, None, evaluate=False)

    return fitted_curve, fit_res.params



