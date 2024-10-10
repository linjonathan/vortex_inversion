#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Jonathan Lin
Vortex surgery library.
"""

import numpy as np
import pyamg
import scipy
import warnings
import sphere
from scipy.optimize import curve_fit
from pyshtools import SHCoeffs
from pyshtools.expand import SHExpandDH
from pyshtools.expand import MakeGridDH

"""
Returns the longitude and latitude vector given lonn/latn spacings in
the below range.
Longitude range is from [0, 360), and latitude range is from [90, -90).
"""
def lon_lat(lonn, latn):
    lon = np.linspace(0, 360, lonn+1)[0:-1]
    lat = np.linspace(90, -90, latn+1)[0:-1]
    return(lon, lat)

class VORTEX_LIB:
    """
    This is a class which implements the main methods for inverting a patch of
    vorticity on a sphere.
    """

    """
    lon and lat are the grid for inversion, should be np.arrays and in degrees.
    dlon and dlat are the grid spacings defined by lon and lat.
    num_inv describes the number of iterative inversions
    """
    def __init__(self, lon, lat, num_inv=3, xres=1):
        # 9/10/2020: Change xres=1 (instead of 2) since ensemble resolutions
        # are becoming better (0.5 degrees), and can perform the inversion
        # with high accuracy.

        self.lon = lon
        self.lat = lat
        self.dlon = np.abs(lon[1] - lon[0])
        self.dlat = np.abs(lat[1] - lat[0])

        self.lon_rad = np.deg2rad(self.lon)
        self.lat_rad = np.deg2rad(self.lat)

        self.LON, self.LAT = np.meshgrid(self.lon_rad, self.lat_rad)
        self.num_inv = num_inv

        self.xres = xres
        self.d_crit = 400
        self.n_deg = 40

    def func_powerlaw(self, x, m, c, c0):
        return c0 + x**m * c

    def wrap_lon_indexing(self, f, latidxs, lonidxs):
        return f[latidxs, :].take(lonidxs, axis=1, mode='wrap')

    # Returns the center of vortex calculated using gradient descent.
    def vortex_center(self, rv, slon, slat):
        lon, lat = lon_lat(rv.shape[1], rv.shape[0])
        LON, LAT = np.meshgrid(lon, lat)
        dists = sphere.haversine(slon, slat, LON, LAT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mask = dists < 400
        idx = np.argmax(rv[mask])
        return(LON[mask][idx], LAT[mask][idx])

    # Returns environmental and vortex relative vorticity for
    # points that are within d_crit of the center.
    def vortex_search_asym(self, rv, x, y):
        rv_env = np.copy(rv)
        rv_vort = np.zeros(rv.shape)
        lon, lat = lon_lat(rv.shape[1], rv.shape[0])
        LON, LAT = np.meshgrid(lon, lat)

        dists = sphere.haversine(x, y, LON, LAT)
        dists[np.isnan(dists)] = 1e10
        mask = dists <= self.d_crit

        rv_env[mask] = 0
        rv_vort[mask] = rv[mask]
        return(rv_env, rv_vort)

    # Cut the vorticity (rv) on grid points that are less than dist km
    # away from the center, (x, y). Set a cutoff vorticity of 1e-5.
    # Returns the filtered background vorticity, and the vortex's vorticity.
    def cut_vortex(self, rv, x, y):
        rv_env, rv_asym = self.vortex_search_asym(rv, x, y)

        return(rv_env, rv_asym)

    def super_resolution(self, f):
        if self.xres > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Expand into spherical harmonics
                f_coeffs = SHExpandDH(f, norm=4, sampling=2)
                f_coeffs = SHCoeffs.from_array(f_coeffs, normalization='ortho')
                ns = f_coeffs.coeffs.shape[1]
                ls = np.arange(ns, dtype='float')

                # Find the power spectrum fit at higher order SHs
                power_per_l = f_coeffs.spectrum(unit='per_l')
                target_func = self.func_powerlaw
                cfit = curve_fit(target_func, ls[1:], np.log(power_per_l[1:]),
                                 p0=np.array((0.288, -2, 4)))
                ls_hd = np.arange(ns*self.xres)
                f_power = np.exp(target_func(ls_hd, *cfit[0]))

                # Generate a non-unique set of SHs that obey the power spectrum
                # and set the lowest order ones to the original field.
                clm = SHCoeffs.from_random(f_power, normalization='ortho', exact_power='True')
                hres_coef = np.copy(clm.coeffs)
                hres_coef[:, 0:ns, 0:ns] = f_coeffs.coeffs
                f_hd = MakeGridDH(hres_coef, sampling=2, norm=4)

                # Original field aliased to higher orders
                hres_coef[:, (ns+1):, :] = 0
                f_lower = MakeGridDH(hres_coef, sampling=2, norm=4)

                # Non-unique field.
                hres_coef = np.copy(clm.coeffs)
                hres_coef[:, 0:ns, 0:ns] = 0
                f_higher = MakeGridDH(hres_coef, sampling=2, norm=4)

            return(f_hd, f_lower, f_higher)
        else:
            return(f, f, np.zeros(f.shape))

    # Recast a higher resolution wind field [lat, lon] back to lower resolution.
    # As in above, the vortex library must be consistent with the wnd array.
    def lower_res_f(self, f):
        if self.xres == 1:
            return f
        lon, lat = lon_lat(f.shape[1], f.shape[0])
        dlon = np.round(np.abs(lon[1] - lon[0]), 4)
        dlat = np.round(np.abs(lat[1] - lat[0]), 4)
        X, Y = np.meshgrid(lon, lat)
        lon_diff = np.abs(np.mod(np.round(X, 4), dlon*self.xres))
        lat_diff = np.abs(np.mod(np.round(Y, 4), dlat*self.xres))
        lonidx = np.logical_or(lon_diff <= 1e-5, np.abs(lon_diff - (dlon*self.xres)) < 1e-5)
        latidx = np.logical_or(lat_diff <= 1e-5, np.abs(lat_diff - (dlat*self.xres)) < 1e-5)
        lowres_idx = np.logical_and(lonidx, latidx)
        return(np.reshape(f[lowres_idx],
                          (int(len(lat) / self.xres),
                           int(len(lon) / self.xres))))

    # (2r+1)^2 point smoothing for all points specified in the mask, with a box
    # diameter of (2r+1), with a maximum number of ntimes.
    def smoother(self, f, mask, r, ntimes=0):
        f_smooth = np.copy(f)
        yidx, xidx = np.where(mask)
        temp = np.copy(f_smooth)
        if (np.sum(mask) > 0): # only if a vortex is identified
            niter = 0
            while True:
                for i in range(0, len(xidx)):
                    x = xidx[i]; y = yidx[i]

                    xidxs = range(x-r, x+r+1)
                    yidxs = range(y-r, y+r+1)
                    temp[y, x] = np.nanmean(self.wrap_lon_indexing(f_smooth, yidxs, xidxs))

                # Stop smoothing if convergence is reached.
                if (np.abs(np.nanmean(temp[mask] - f_smooth[mask])) < 8e-8 or
                    niter > ntimes):
                    break
                else:
                    f_smooth[mask] = temp[mask]
                niter = niter+1
        return(f_smooth)

    """
    Defines the 2nd order central difference stencil for the laplace equation on a sphere.
    nx and ny represent the size of the domain.
    """
    def laplace_stencil(self, nx, ny, lon, lat):
        n = (nx) * (ny)  # number of unknowns
        d = np.ones(n)  # diagonals
        dh = lon[1] - lon[0]

        # Since we are going by row-order first, constant latitude per nx.
        LON, LAT = np.meshgrid(lon, lat)

        d0 = np.multiply(d.copy(), -2 * (1 + np.divide(1, np.power(np.cos(np.deg2rad(LAT.flatten())), 2))))
        d1_lower = np.multiply(d.copy()[0:-1], np.divide(1, np.power(np.cos(np.deg2rad(LAT.flatten()[:-1])), 2)))
        d1_upper = np.multiply(d.copy()[0:-1], np.divide(1, np.power(np.cos(np.deg2rad(LAT.flatten()[1:])), 2)))

        dnx_lower = np.multiply(d.copy()[0:-nx], 1)  # + 0.5 * dh / 10 * np.tan(np.deg2rad(LAT.flatten()[nx:])))
        dnx_upper = np.multiply(d.copy()[0:-nx], 1)  # - 0.5 * dh /10 * np.tan(np.deg2rad(LAT.flatten()[:-nx])))

        d1_lower[nx - 1::nx] = 0  # every nx element on first diagonal is zero; starting from the nx-th element
        d1_upper[nx - 1::nx] = 0

        # Every nx element on first upper diagonal is two; stating from the first element.
        # d1_upper[::nx] = 2 * np.divide(1, np.power(np.cos(np.deg2rad(LAT.flatten()[1::nx])), 2))
        # Every nx element on first lower diagonal is two; stating from the first element.
        # d1_lower[(nx-2)::nx] = 2 * np.divide(1, np.power(np.cos(np.deg2rad(LAT.flatten()[(nx-1)::nx])), 2))

        # The first nx elements in the nx-th upper diagonal is two
        # dnx_upper[0:nx] = 2 * (1 - 0.5 * dh * np.tan(np.deg2rad(LAT.flatten()[nx:(2*nx)])))
        # The last nx elements in the nx-th lower diagonal is two;
        # dnx_lower[(dnx_lower.size - nx):] = 2 * (1 + 0.5 * dh * np.tan(np.deg2rad(LAT.flatten()[-nx:])))

        A = scipy.sparse.diags([d0, d1_upper, d1_lower, dnx_upper, dnx_lower], [0, 1, -1, nx, -nx], format='csr')
        return(A)

    """
    Extracts a box of 20 by 20 degrees from the vortex center x, y.
    """
    def get_box(self, lon, lat, x, y, F):
        # Wrap longitude to ensure the box is not limited by the grid coordinates.
        lon_wrap = np.concatenate((lon - 360, lon, lon + 360))
        lon_idx = np.argmin(np.abs(lon_wrap - x))
        lat_idx = np.argmin(np.abs(lat - y))
        n_lon_deg = self.n_deg
        n_lat_deg = self.n_deg
        dlon = np.round(np.abs(lon_wrap[1] - lon_wrap[0]), 4)
        dlat = np.round(np.abs(lat[1] - lat[0]), 4)
        nx = int(n_lon_deg / dlon)
        ny = int(n_lat_deg / dlat)
        lon_box = lon_wrap[(lon_idx - nx):(lon_idx + nx)]
        lat_box = lat[(lat_idx - ny):(lat_idx + ny)]

        F_wrap = np.hstack((F, F, F))
        f_box = F_wrap[(lat_idx - ny):(lat_idx + ny), (lon_idx - nx):(lon_idx + nx)]
        return (lon_box, lat_box, f_box)

    """
    Sets the values of F_box.
    """
    def set_box(self, lon, lat, x, y, F, F_box):
        # Wrap longitude to ensure the box is not limited by the grid coordinates.
        lon_wrap = np.concatenate((lon - 360, lon, lon + 360))
        lon_idx = np.argmin(np.abs(lon_wrap - x))
        lat_idx = np.argmin(np.abs(lat - y))
        n_lon_deg = self.n_deg
        n_lat_deg = self.n_deg
        dlon = np.round(np.abs(lon[1] - lon[0]), 4)
        dlat = np.round(np.abs(lat[1] - lat[0]), 4)
        nx = int(n_lon_deg / dlon)
        ny = int(n_lat_deg / dlat)
        F_wrap = np.hstack((F, F, F))
        Fs_wrap = np.copy(F_wrap)
        Fs_wrap[(lat_idx - ny):(lat_idx + ny), (lon_idx - nx):(lon_idx + nx)] -= F_box

        # Return in the original coordinates.
        Fs = np.copy(Fs_wrap[:, lon.size:(2 * lon.size)])
        if (lon_idx - nx) < lon.size:
            # Take the western points and move it to the right side.
            Fs[:, ((lon_idx - nx) - lon.size):] = Fs_wrap[:, (lon_idx - nx):lon.size]
        elif (lon_idx + nx) >= (2 * lon.size):
            # Take the eastern points and move it to the left side.
            Fs[:, :(lon_idx + nx - (2 * lon.size))] = Fs_wrap[:, (2 * lon.size):(lon_idx + nx)]
        return (Fs)

    # Vortex surgery on a wind field uwnd, vwnd.
    # slon and slat represent the guess point for the vortex that will be removed.
    # Returns the wind fields (uFilt, vFilt) of the filtered environment,
    # as well as the wind fields of the extracted vortex, (uVort, vVort).
    # Uses multi-grid methods.
    def vortex_surgery(self, u, v, slon, slat):
        # Super resolute the wind fields first.
        u_hd, u_l, u_h = self.super_resolution(u)
        v_hd, v_l, v_h = self.super_resolution(v)
        lon_hd, lat_hd = lon_lat(u_hd.shape[1], u_hd.shape[0])

        # Find the vortex center using the original, aliased field.
        rv = sphere.calc_rvor_sphere(lon_hd, lat_hd, u_l, v_l)
        div = sphere.calc_div_sphere(lon_hd, lat_hd, u_l, v_l)
        x, y = self.vortex_center(rv, slon, slat)
        rv_filt, rv_vort = self.cut_vortex(rv, x, y)
        div_filt, div_vort = self.cut_vortex(div, x, y)

        # Get relative voriticty and divergence over a box centered on vortex.
        lon_box, lat_box, rv_box = self.get_box(lon_hd, lat_hd, x, y, rv_vort)
        lon_box, lat_box, div_box = self.get_box(lon_hd, lat_hd, x, y, div_vort)

        # Invert the laplacian to calculate the stream function and velocity potential.
        Nx = rv_box.shape[1]
        Ny = rv_box.shape[0]
        A = self.laplace_stencil(Nx, Ny, lon_box, lat_box)
        ml = pyamg.ruge_stuben_solver(A)           # construct the multigrid hierarchy
        X = ml.solve(rv_box.flatten(), tol=1e-6)   # solve the linear system
        psi = -X.reshape(rv_box.shape)
        X = ml.solve(div_box.flatten(), tol=1e-6)  # solve the linear system
        phi = -X.reshape(div_box.shape)

        # Calculate velocities associated with Helmholtz decomposition.
        u_vort_rv, v_vort_rv = sphere.sf_wnd(lon_box, lat_box, psi)
        u_vort_div, v_vort_div = sphere.pot_wnd(lon_box, lat_box, phi)

        # Find the constant multiplicative scale for both
        # the stream function and velocity potential.
        rv_inv = sphere.calc_rvor_sphere(lon_box, lat_box, u_vort_rv, v_vort_rv)
        mask = np.abs(rv_box) > 1e-5
        c, _, _, _ = np.linalg.lstsq(np.expand_dims(rv_inv[mask], 1),
                                     np.expand_dims(rv_box[mask], 1), rcond=None)
        rv_scale = c.flatten()[0]

        div_inv = sphere.calc_div_sphere(lon_box, lat_box, u_vort_div, v_vort_div)
        mask = np.abs(div_box) > 1e-5
        c, _, _, _ = np.linalg.lstsq(np.expand_dims(div_inv[mask], 1),
                                     np.expand_dims(div_box[mask], 1), rcond=None)
        div_scale = c.flatten()[0]

        uf_s = self.set_box(lon_hd, lat_hd, x, y, u_l, u_vort_rv * rv_scale + u_vort_div * div_scale)
        vf_s = self.set_box(lon_hd, lat_hd, x, y, v_l, v_vort_rv * rv_scale + v_vort_div * div_scale)

        # Recast all the return fields into the original resolution.
        uf_s_orig = self.lower_res_f(uf_s)
        vf_s_orig = self.lower_res_f(vf_s)

        # Apply some smoothing.
        LON, LAT = np.meshgrid(self.lon, self.lat)
        dists = sphere.haversine(x, y, LON, LAT)
        dists[np.isnan(dists)] = 1e10
        uf_s_orig = self.smoother(uf_s_orig, dists < self.d_crit, 1, 20)
        vf_s_orig = self.smoother(vf_s_orig, dists < self.d_crit, 1, 20)

        return(x, y, uf_s_orig, vf_s_orig, uf_s, vf_s)
